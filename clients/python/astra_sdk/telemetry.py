"""Background telemetry reporter — fault-tolerant, offline-durable by construction.

Design contract: NOTHING in this module may ever raise into the caller's serving
path. Events queue into a bounded in-memory deque; a daemon thread every few
seconds drains them into a durable on-disk *spool* (one JSON segment per batch)
BEFORE touching the network, then POSTs each pending segment to
POST /api/v1/telemetry/{deployment_id}/batch and deletes a segment only after the
server acks it (2xx). So the closed loop survives being offline: while the device
has no connectivity, segments accumulate on disk and survive process restarts;
when connectivity returns they flush oldest-first and are deleted. Each segment
carries a client-generated ``batchId`` that the server dedups on, so a re-send
after a crash never double-counts.

Disable telemetry entirely with report_telemetry=False / ASTRA_SDK_TELEMETRY=0.
Disable disk buffering (in-memory only, best-effort, lost on exit) with
ASTRA_SDK_SPOOL=0.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._http import AstraApiError, HttpSession
from .stats import WindowAggregator
from .system import runtime_fingerprint, system_sample


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


_QUEUE_MAX = 10_000
_BATCH_MAX = 450            # combined events+snapshots+windows — below the server cap
# 4xx statuses that are still worth retrying later: auth may be fixed out of
# band (401/403), a paused deployment can be resumed (409), and 408/429 are
# transient by definition. Any OTHER 4xx means the server will never accept
# this exact payload — retrying it forever would head-of-line block the spool.
_RETRY_LATER_STATUS = {401, 403, 408, 409, 429}
_MEM_PENDING_MAX = 64       # in-memory pending batches when the spool is off/unwritable
_FLUSH_INTERVAL_S = _env_float("ASTRA_SDK_FLUSH_INTERVAL_S", 5.0)
_SNAPSHOT_INTERVAL_S = _env_float("ASTRA_SDK_SNAPSHOT_INTERVAL_S", 30.0)
_WINDOW_INTERVAL_S = _env_float("ASTRA_SDK_WINDOW_INTERVAL_S", 60.0)
_WINDOW_MAX_REQUESTS = int(_env_float("ASTRA_SDK_WINDOW_MAX_REQUESTS", 200))
_SPOOL_MAX_BYTES = int(_env_float("ASTRA_SDK_SPOOL_MAX_MB", 32.0) * 1024 * 1024)
_DEFAULT_CACHE = "~/.cache/astra"
_ATEXIT_BUDGET_S = 3.0


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def telemetry_enabled(flag: bool | None = None) -> bool:
    if flag is not None and not flag:
        return False
    return os.environ.get("ASTRA_SDK_TELEMETRY", "1") not in ("0", "false", "no")


def _spool_enabled() -> bool:
    return os.environ.get("ASTRA_SDK_SPOOL", "1") not in ("0", "false", "no")


class AstraTelemetryReporter:
    """Collects events/snapshots/windows and ships them in the background, with a
    durable on-disk spool so nothing is lost while offline."""

    def __init__(
        self,
        base_url: str,
        deployment_id: str,
        api_key: str,
        *,
        sdk_version: str,
        enabled: bool = True,
        active_provider: str | None = None,
        cache_dir: str = _DEFAULT_CACHE,
    ) -> None:
        self.enabled = telemetry_enabled(enabled)
        self.client_id = f"sdk_{uuid.uuid4().hex[:10]}"
        self.deployment_id = deployment_id
        self._events: deque[dict] = deque(maxlen=_QUEUE_MAX)
        self._snapshots: deque[dict] = deque(maxlen=64)
        self._windows: deque[dict] = deque(maxlen=64)
        self._mem_pending: deque[dict] = deque(maxlen=_MEM_PENDING_MAX)
        self._dropped = 0
        self._sent_events = 0
        self._lock = threading.Lock()
        self._window_lock = threading.Lock()
        self._aggregator = WindowAggregator()
        self._window_requests = 0
        # Captures the ORT provider the serving session actually selected (vs the
        # first *available* one) so the dashboard attributes latency to real hw.
        self._fingerprint = runtime_fingerprint(sdk_version, active_provider)
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._http: HttpSession | None = None
        self._throughput_marker = (time.monotonic(), 0)
        self._seg_seq = 0

        # Durable spool dir: ~/.cache/astra/<deployment>/telemetry/ (alongside the
        # artifact cache). ASTRA_SDK_SPOOL_DIR overrides; ASTRA_SDK_SPOOL=0 disables.
        self._spool = self.enabled and _spool_enabled()
        override = os.environ.get("ASTRA_SDK_SPOOL_DIR")
        base = override or os.path.join(cache_dir, deployment_id, "telemetry")
        self._spool_dir = Path(os.path.expanduser(base))
        if self._spool:
            try:
                self._spool_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                self._spool = False  # unwritable → fall back to in-memory pending

        if self.enabled:
            self._http = HttpSession(
                base_url, api_key, timeout=10.0, max_attempts=2, max_backoff=30.0)
            self._thread = threading.Thread(
                target=self._loop, name="astra-telemetry", daemon=True)
            self._thread.start()
            atexit.register(self.close)

    # ── recording (hot path — must be cheap and never raise) ────────────────

    def record_event(
        self,
        *,
        latency_ms: float,
        pre_ms: float | None = None,
        post_ms: float | None = None,
        success: bool = True,
        error_code: str | None = None,
        batch_size: int = 1,
        region: str = "local",
        input_sig: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            event = {
                "id": uuid.uuid4().hex,
                "ts": _iso_now(),
                "latencyMs": round(float(latency_ms), 3),
                "success": success,
                "batchSize": int(batch_size),
                "region": region,
            }
            if pre_ms is not None:
                event["preMs"] = round(float(pre_ms), 3)
            if post_ms is not None:
                event["postMs"] = round(float(post_ms), 3)
            if error_code:
                event["errorCode"] = error_code
            if input_sig:
                event["inputSig"] = input_sig
            with self._lock:
                if len(self._events) == self._events.maxlen:
                    self._dropped += 1
                self._events.append(event)
                self._sent_events += 1
                self._window_requests += 1
                cut_window = self._window_requests >= _WINDOW_MAX_REQUESTS
            if cut_window:
                # Request-driven window cut: keeps window sizes deterministic
                # under burst load instead of waiting for the next loop tick.
                self._take_window()
        except Exception:
            pass

    def observe(self, inputs: dict[str, Any] | None, output: Any | None) -> None:
        if not self.enabled:
            return
        try:
            self._aggregator.observe(inputs, output)
        except Exception:
            pass

    # ── background loop ──────────────────────────────────────────────────────

    def _loop(self) -> None:
        # Startup recovery: flush any segments left by a prior (offline) session.
        try:
            self._send_pending()
        except Exception:
            pass
        last_snapshot = last_window = time.monotonic()
        while not self._closed.wait(_FLUSH_INTERVAL_S):
            now = time.monotonic()
            try:
                if now - last_snapshot >= _SNAPSHOT_INTERVAL_S:
                    self._take_snapshot()
                    last_snapshot = now
                window_due = (
                    now - last_window >= _WINDOW_INTERVAL_S
                    or self._window_requests >= _WINDOW_MAX_REQUESTS
                )
                if window_due:
                    self._take_window()
                    last_window = now
                self._flush()
            except Exception:
                pass  # the loop must survive anything

    def _take_snapshot(self) -> None:
        now = time.monotonic()
        marker_t, marker_n = self._throughput_marker
        with self._lock:
            sent = self._sent_events
            dropped = self._dropped
        elapsed_min = max(1e-6, (now - marker_t) / 60.0)
        rpm = (sent - marker_n) / elapsed_min
        self._throughput_marker = (now, sent)
        self._snapshots.append({
            "id": uuid.uuid4().hex,
            "ts": _iso_now(),
            **system_sample(),
            "throughputRpm": round(rpm, 2),
            "droppedEvents": dropped,
            **self._fingerprint,
        })

    def _take_window(self) -> None:
        # Serialized: callable from both the hot path (request-cap cut) and
        # the background loop (time-based cut).
        with self._window_lock:
            with self._lock:
                self._window_requests = 0
            window = self._aggregator.flush()
            if window:
                window.setdefault("id", uuid.uuid4().hex)
                self._windows.append(window)

    # ── flush: drain in-memory → durable spool → network (delete-after-ack) ──

    def _drain_to_batch(self) -> dict | None:
        # Snapshots/windows are few (deque maxlen 64 each); take them all, then
        # fill the REMAINING budget with events so the COMBINED batch stays
        # ≤ _BATCH_MAX — the server counts all three lists against its cap and
        # 422s an oversized batch outright. Leftover events wait for next flush.
        snapshots = [self._snapshots.popleft() for _ in range(len(self._snapshots))]
        windows = [self._windows.popleft() for _ in range(len(self._windows))]
        budget = max(0, _BATCH_MAX - len(snapshots) - len(windows))
        with self._lock:
            events = [self._events.popleft()
                      for _ in range(min(budget, len(self._events)))]
        if not events and not snapshots and not windows:
            return None
        return {
            "clientId": self.client_id,
            "batchId": uuid.uuid4().hex,
            "events": events,
            "snapshots": snapshots,
            "windows": windows,
        }

    def _flush(self) -> bool:
        if self._http is None:
            return True
        batch = self._drain_to_batch()
        if batch is not None and not (self._spool and self._spool_write(batch)):
            # Spool disabled or write failed → hold the batch in memory (bounded).
            if len(self._mem_pending) == self._mem_pending.maxlen:
                self._dropped += 1
            self._mem_pending.append(batch)
        return self._send_pending()

    def _send_pending(self) -> bool:
        """Send pending batches oldest-first; delete each only after a 2xx ack.
        Stops at the first failure (still offline) and returns False."""
        if self._http is None:
            return True
        if self._spool:
            for seg in self._spool_segments():
                try:
                    batch = json.loads(seg.read_text(encoding="utf-8"))
                except Exception:
                    self._unlink(seg)  # corrupt segment — drop it
                    continue
                if self._send_batch(batch):
                    self._unlink(seg)
                else:
                    return False
        while self._mem_pending:
            if self._send_batch(self._mem_pending[0]):
                self._mem_pending.popleft()
            else:
                return False
        return True

    def _send_batch(self, batch: dict) -> bool:
        """True → the batch is finished (acked, or permanently rejected and
        counted as dropped) and its segment may be deleted; False → transient
        failure, keep the segment and retry on a later flush."""
        try:
            self._http.request(
                "POST", f"/api/v1/telemetry/{self.deployment_id}/batch", json=batch)
            return True
        except AstraApiError as exc:
            if 400 <= exc.status < 500 and exc.status not in _RETRY_LATER_STATUS:
                # e.g. 422 batch_too_large, 404 deployment deleted: re-sending
                # the same payload can never succeed, and returning False would
                # head-of-line block every younger segment forever.
                with self._lock:
                    self._dropped += len(batch.get("events") or [])
                return True
            return False
        except Exception:
            return False

    # ── durable spool helpers ────────────────────────────────────────────────

    def _spool_segments(self) -> list[Path]:
        try:
            return sorted(self._spool_dir.glob("seg-*.json"))
        except Exception:
            return []

    def _spool_write(self, batch: dict) -> bool:
        try:
            self._enforce_spool_cap()
            self._seg_seq += 1
            stem = f"seg-{int(time.time() * 1000):013d}-{self._seg_seq:04d}"
            tmp = self._spool_dir / (stem + ".part")
            tmp.write_text(json.dumps(batch), encoding="utf-8")
            os.replace(tmp, self._spool_dir / (stem + ".json"))
            return True
        except Exception:
            return False

    def _enforce_spool_cap(self) -> None:
        # Bounded disk usage: drop oldest segments when over the cap.
        try:
            segs = self._spool_segments()
            total = sum(self._size(p) for p in segs)
            i = 0
            while total > _SPOOL_MAX_BYTES and i < len(segs):
                total -= self._size(segs[i])
                self._unlink(segs[i])
                self._dropped += 1
                i += 1
        except Exception:
            pass

    @staticmethod
    def _size(p: Path) -> int:
        try:
            return p.stat().st_size
        except Exception:
            return 0

    @staticmethod
    def _unlink(p: Path) -> None:
        try:
            p.unlink()
        except Exception:
            pass

    # ── shutdown ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Durably spool everything still in memory, best-effort flush within a
        small budget, and leave any unsent segments on disk for next time.
        Idempotent."""
        if not self.enabled or self._closed.is_set():
            return
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            # Always ship at least one snapshot per session — it carries the
            # runtime fingerprint the dashboard's client-hosts table shows
            # (short sessions would otherwise never hit the 30s cadence).
            self._take_snapshot()
            self._take_window()
            self._flush()  # drains remaining in-memory into the durable spool
            deadline = time.monotonic() + _ATEXIT_BUDGET_S
            while time.monotonic() < deadline:
                if self._send_pending():
                    break
                time.sleep(0.2)
        except Exception:
            pass
        finally:
            if self._http is not None:
                self._http.close()
