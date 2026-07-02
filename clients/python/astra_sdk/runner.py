"""AstraRunner — serve a Astra-deployed artifact on your own hardware.

    from astra_sdk import AstraRunner

    runner = AstraRunner.from_deployment(
        base_url="https://app.example.com",
        deployment_id="dep_ab12cd34ef",
        api_key="astra_sk_live_…",
    )
    out = runner.run({"input": my_numpy_array})    # local onnxruntime inference
    runner.close()

The artifact is pulled once via the API-key-authed
GET /api/v1/artifacts/{deployment_id} and cached on disk keyed by its sha256,
so restarts don't re-download. Every run() is measured (pre/infer/post) and
shipped to the Astra dashboard by the background AstraTelemetryReporter, together
with periodic system snapshots and windowed input/output stats — a full
closed-loop, entirely on your own hardware.

Install:  pip install astra-ai-sdk   (onnxruntime + numpy come bundled)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ._http import HttpSession, resolve_base_url
from .telemetry import AstraTelemetryReporter

_DEFAULT_CACHE = "~/.cache/astra"


class AstraRunnerError(Exception):
    pass


def _require_runtime():
    # onnxruntime + numpy are core dependencies of astra-ai-sdk, so this normally
    # always passes; it only trips on a broken/partial install. Defensive.
    try:
        import numpy
        import onnxruntime
    except ImportError as exc:  # pragma: no cover
        raise AstraRunnerError(
            "onnxruntime + numpy are required to run models and ship with "
            "astra-ai-sdk — reinstall it: pip install --force-reinstall astra-ai-sdk"
        ) from exc
    return numpy, onnxruntime


class AstraRunner:
    """Local ONNX serving with built-in telemetry."""

    def __init__(
        self,
        model_path: str,
        *,
        reporter: AstraTelemetryReporter | None = None,
        providers: list[str] | None = None,
    ) -> None:
        np, ort = _require_runtime()
        self._np = np
        self._reporter = reporter
        self._session = ort.InferenceSession(
            model_path,
            providers=providers or ort.get_available_providers(),
        )
        self._input_meta = self._session.get_inputs()
        self._output_names = [o.name for o in self._session.get_outputs()]
        self.model_path = model_path

    @property
    def active_provider(self) -> str:
        """The execution provider ORT actually bound (first in get_providers)."""
        try:
            providers = self._session.get_providers()
            return providers[0] if providers else ""
        except Exception:  # noqa: BLE001
            return ""

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_deployment(
        cls,
        deployment_id: str,
        api_key: str,
        *,
        base_url: str | None = None,
        cache_dir: str = _DEFAULT_CACHE,
        report_telemetry: bool = True,
        providers: list[str] | None = None,
        timeout: float = 60.0,
    ) -> "AstraRunner":
        """Pull (or reuse) the deployed artifact and build a runner for it.

        base_url defaults to the hosted Astra origin (override with the
        ASTRA_BASE_URL env var or the base_url keyword)."""
        _require_runtime()
        base = resolve_base_url(base_url)
        path = pull_artifact(
            deployment_id, api_key, base_url=base,
            cache_dir=cache_dir, timeout=timeout,
        )
        from . import __version__

        # Build the session first so the reporter can record the provider ORT
        # actually selected — not just the first one available on the box.
        runner = cls(str(path), providers=providers)
        runner._reporter = AstraTelemetryReporter(
            base, deployment_id, api_key,
            sdk_version=__version__, enabled=report_telemetry,
            active_provider=runner.active_provider, cache_dir=cache_dir,
        )
        return runner

    @classmethod
    def from_file(
        cls,
        model_path: str,
        *,
        providers: list[str] | None = None,
        report_telemetry: bool = False,
        deployment_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> "AstraRunner":
        """Serve a model file you ALREADY have on disk — e.g. the one from the
        SDK Hub "Download Artifact" button. No deployment / network needed.
        Telemetry is OFF unless you also pass deployment_id + api_key (then local
        runs still report to that deployment)."""
        _require_runtime()
        runner = cls(model_path, providers=providers)
        want = report_telemetry or bool(deployment_id and api_key)
        if want and deployment_id and api_key:
            from . import __version__

            runner._reporter = AstraTelemetryReporter(
                resolve_base_url(base_url), deployment_id, api_key,
                sdk_version=__version__, enabled=True,
                active_provider=runner.active_provider,
            )
        return runner

    # ── serving ──────────────────────────────────────────────────────────────

    def run(
        self,
        inputs: dict[str, Any] | None = None,
        *,
        region: str = "local",
    ) -> dict[str, Any]:
        """One inference. `inputs` maps input name → ndarray/nested list; pass
        None to synthesize a random valid probe (smoke tests / benchmarks)."""
        np = self._np
        t0 = time.perf_counter()
        error_code: str | None = None
        feeds: dict[str, Any] = {}
        try:
            feeds = self._prepare(inputs)
            pre_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            raw = self._session.run(self._output_names, feeds)
            infer_ms = (time.perf_counter() - t1) * 1000

            t2 = time.perf_counter()
            outputs = [
                {"name": n, "shape": list(np.asarray(o).shape)}
                for n, o in zip(self._output_names, raw)
            ]
            first = raw[0] if raw else None
            if self._reporter is not None:
                self._reporter.observe(feeds, first)
            post_ms = (time.perf_counter() - t2) * 1000
        except Exception as exc:
            error_code = "bad_input" if isinstance(exc, (ValueError, KeyError)) else "inference_error"
            if self._reporter is not None:
                self._reporter.record_event(
                    latency_ms=0.0, success=False, error_code=error_code,
                    batch_size=self._batch_of(feeds), region=region,
                    input_sig=self._signature(feeds),
                )
            raise

        if self._reporter is not None:
            self._reporter.record_event(
                latency_ms=infer_ms, pre_ms=pre_ms, post_ms=post_ms,
                success=True, batch_size=self._batch_of(feeds), region=region,
                input_sig=self._signature(feeds),
            )
        return {
            "latencyMs": round(infer_ms, 3),
            "preMs": round(pre_ms, 3),
            "postMs": round(post_ms, 3),
            "outputs": outputs,
            "raw": raw,
        }

    def _prepare(self, inputs: dict[str, Any] | None) -> dict[str, Any]:
        np = self._np
        feeds: dict[str, Any] = {}
        if inputs is None:
            for meta in self._input_meta:
                shape = [d if isinstance(d, int) and d > 0 else 1 for d in meta.shape]
                feeds[meta.name] = np.random.default_rng().standard_normal(
                    shape).astype(np.float32)
            return feeds
        for meta in self._input_meta:
            if meta.name not in inputs:
                raise ValueError(f"missing input '{meta.name}'")
            arr = np.asarray(inputs[meta.name])
            if arr.dtype.kind == "f":
                arr = arr.astype(np.float32)
            feeds[meta.name] = arr
        return feeds

    def _batch_of(self, feeds: dict[str, Any]) -> int:
        for arr in feeds.values():
            shape = getattr(arr, "shape", None)
            if shape:
                return int(shape[0])
        return 1

    def _signature(self, feeds: dict[str, Any]) -> str | None:
        parts = []
        for name, arr in feeds.items():
            shape = getattr(arr, "shape", ())
            dtype = getattr(arr, "dtype", "")
            parts.append(f"{name}:{'x'.join(str(d) for d in shape)}:{dtype}")
        return ";".join(parts) or None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._reporter is not None:
            self._reporter.close()

    def __enter__(self) -> "AstraRunner":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def pull_artifact(
    deployment_id: str,
    api_key: str,
    *,
    base_url: str | None = None,
    cache_dir: str = _DEFAULT_CACHE,
    timeout: float = 60.0,
) -> Path:
    """Download the deployed artifact (sha256-cached under cache_dir).

    base_url defaults to the hosted Astra origin (override with ASTRA_BASE_URL)."""
    http = HttpSession(resolve_base_url(base_url), api_key, timeout=timeout)
    try:
        info = http.request(
            "GET", f"/api/v1/artifacts/{deployment_id}/info").json()
        sha = info["sha256"]
        root = Path(cache_dir).expanduser() / deployment_id
        root.mkdir(parents=True, exist_ok=True)
        target = root / f"{sha}{Path(info['fileName']).suffix or '.onnx'}"
        if target.exists() and target.stat().st_size == info["sizeBytes"]:
            return target
        resp = http.request("GET", f"/api/v1/artifacts/{deployment_id}")
        tmp = target.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.replace(target)
        (root / "meta.json").write_text(json.dumps(info, indent=2))
        return target
    finally:
        http.close()
