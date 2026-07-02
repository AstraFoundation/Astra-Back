"""AstraTelemetryReporter semantics against a fake in-process server."""

from __future__ import annotations

import sys
import threading
import time
import types

import httpx
import pytest

from astra_sdk.telemetry import AstraTelemetryReporter


def _fake_pynvml() -> types.ModuleType:
    """A minimal in-memory nvidia-ml-py so the GPU fingerprint path runs on a box
    with no NVIDIA card."""
    class _Mem:
        total = 40 * 1024**3
        used = 12 * 1024**3

    class _Util:
        gpu = 73

    m = types.ModuleType("pynvml")
    m.NVML_TEMPERATURE_GPU = 0
    m.nvmlInit = lambda: None
    m.nvmlDeviceGetCount = lambda: 1
    m.nvmlDeviceGetHandleByIndex = lambda i: object()
    m.nvmlDeviceGetName = lambda h: "NVIDIA A100-SXM4-40GB"
    m.nvmlDeviceGetMemoryInfo = lambda h: _Mem()
    m.nvmlSystemGetDriverVersion = lambda: "550.90.07"
    m.nvmlSystemGetCudaDriverVersion = lambda: 12040
    m.nvmlDeviceGetUtilizationRates = lambda h: _Util()
    m.nvmlDeviceGetTemperature = lambda h, s: 61
    return m


class FakeBackend:
    """Captures batches; can be told to fail N times."""

    def __init__(self) -> None:
        self.batches: list[dict] = []
        self.fail_next = 0
        self.lock = threading.Lock()

    def handler(self, request: httpx.Request) -> httpx.Response:
        with self.lock:
            if self.fail_next > 0:
                self.fail_next -= 1
                return httpx.Response(503, json={"detail": {"code": "unavailable"}})
            import json

            self.batches.append(json.loads(request.content))
        return httpx.Response(200, json={"accepted": {}, "dropped": 0})

    @property
    def events(self) -> list[dict]:
        return [e for b in self.batches for e in b.get("events", [])]


@pytest.fixture
def backend(monkeypatch):
    fake = FakeBackend()
    transport = httpx.MockTransport(fake.handler)
    original_init = httpx.Client.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = transport
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched)
    return fake


def _reporter(**kw) -> AstraTelemetryReporter:
    return AstraTelemetryReporter(
        "http://test", "dep_x", "astra_sk_test", sdk_version="0.2.0", **kw)


def test_events_flush_on_close(backend):
    rep = _reporter()
    for i in range(25):
        rep.record_event(latency_ms=float(i), pre_ms=0.1, post_ms=0.1)
    rep.close()
    assert len(backend.events) == 25
    ev = backend.events[0]
    assert {"ts", "latencyMs", "success", "batchSize", "region",
            "preMs", "postMs"} <= set(ev)


def test_recording_never_blocks_or_raises(backend):
    rep = _reporter()
    t0 = time.perf_counter()
    for _ in range(5000):
        rep.record_event(latency_ms=1.0)
    assert time.perf_counter() - t0 < 1.0, "hot path must be cheap"
    rep.close()


def test_failed_flush_requeues_then_recovers(backend):
    backend.fail_next = 1
    rep = _reporter()
    for i in range(10):
        rep.record_event(latency_ms=float(i))
    rep.close()  # close retries within its budget
    assert len(backend.events) == 10, "events must survive one failed flush"


def test_disabled_via_flag(backend):
    rep = _reporter(enabled=False)
    rep.record_event(latency_ms=1.0)
    rep.close()
    assert backend.batches == []


def test_disabled_via_env(backend, monkeypatch):
    monkeypatch.setenv("ASTRA_SDK_TELEMETRY", "0")
    rep = _reporter()
    rep.record_event(latency_ms=1.0)
    rep.close()
    assert backend.batches == []


def test_snapshot_carries_gpu_fingerprint_incl_driver_version(backend, monkeypatch):
    """A snapshot must ship the full hardware fingerprint, including every GPU
    field — driverVersion in particular, which the backend now persists."""
    import astra_sdk.system as S

    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml())
    monkeypatch.setattr(S, "_NVML", None, raising=False)  # reset cache → re-import fake

    rep = _reporter(active_provider="CUDAExecutionProvider")
    rep.record_event(latency_ms=1.0)
    rep.close()  # forces a final snapshot flush

    snaps = [s for b in backend.batches for s in b.get("snapshots", [])]
    assert snaps, "close() must flush a final system snapshot"
    s0 = snaps[0]
    # Static hardware identity is present.
    assert {"cpuModel", "cpuCores", "ramTotalMb", "availableProviders",
            "activeProvider", "sdkVersion", "os", "arch"} <= set(s0)
    assert s0["activeProvider"] == "CUDAExecutionProvider"
    # GPU identity — incl. the driverVersion that used to be dropped downstream.
    assert s0["gpuName"] == "NVIDIA A100-SXM4-40GB"
    assert s0["gpuCount"] == 1
    assert s0["cudaVersion"] == "12.4"
    assert s0["driverVersion"] == "550.90.07"
    # Dynamic accelerator sample too.
    assert s0["gpuUtilPct"] == 73.0
    assert s0["gpuMemUsedMb"] > 0


def test_window_stats_emitted_on_close(backend):
    np = pytest.importorskip("numpy")
    rep = _reporter()
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.standard_normal((1, 8)).astype(np.float32)
        logits = rng.standard_normal((1, 5))
        rep.observe({"input": x}, logits)
        rep.record_event(latency_ms=1.0)
    rep.close()
    windows = [w for b in backend.batches for w in b.get("windows", [])]
    assert windows, "close() must flush the open window"
    w = windows[0]
    assert w["n"] == 20
    assert "input" in w["inputs"]
    assert abs(w["inputs"]["input"]["mean"]) < 1.0
    assert "classDist" in w["output"]
    assert len(w["output"]["hist"]) == 16
