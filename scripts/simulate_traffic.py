#!/usr/bin/env python3
"""Simulate the astra-ai-sdk's closed-loop telemetry against a live deployment.

Astra is ON-DEVICE-ONLY: the server never runs a user's model. The only external
traffic a deployment ever sees is the on-device SDK shipping telemetry *batches* —
it downloads the compressed model, serves it locally, measures every inference,
and POSTs the results back through the closed loop. This script emulates exactly
that. It manufactures believable per-inference events (lognormal latency jitter),
buffers them into batches (each with a fresh idempotency batchId), and ships them
to

    POST /api/v1/telemetry/{deployment_id}/batch   (Authorization: Bearer <key>)

alongside a periodic host/system snapshot (~every 30s) and a windowed
input/output distribution report (~every 60s). With --incidents it injects a
latency-spike window and an error burst (success:false, errorCode:"inference_error")
so the Telemetry Dashboard lights up end-to-end — including the drift monitor's
error-rate alert.

It does NOT call any server-side inference endpoint: that path no longer exists.

Example:
    python scripts/simulate_traffic.py \
        --base-url http://localhost:8000 \
        --deployment dep_ab12cd34ef \
        --api-key astra_sk_live_xxxxxxxx \
        --rate 5 --duration 120 --incidents
"""

from __future__ import annotations

import argparse
import platform
import random
import sys
import time
import uuid
from datetime import datetime, timezone

import httpx

# The SDK buffers events durably and flushes on a size or time trigger.
BATCH_MAX_EVENTS = 200      # keep a batch well under the server's 500-item cap
FLUSH_SEC = 5.0             # ship at least this often, even at a low rate
SNAPSHOT_SEC = 30.0         # host snapshot cadence
WINDOW_SEC = 60.0           # input/output distribution window cadence
BASE_LATENCY_MS = 12.0      # nominal local inference latency
N_CLASSES = 10              # simulated classifier output classes


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Astra on-device SDK telemetry simulator")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--deployment", required=True, help="deployment id (dep_…)")
    p.add_argument("--api-key", required=True, help="bearer key (astra_sk_live_…)")
    p.add_argument("--rate", type=float, default=5.0, help="inferences per second")
    p.add_argument("--duration", type=float, default=120.0, help="seconds to run")
    p.add_argument("--incidents", action="store_true",
                   help="inject occasional latency/error incident bursts")
    return p.parse_args()


def _iso_ms(dt: datetime) -> str:
    """ISO8601 UTC with millisecond precision + Z, e.g. 2026-07-01T12:00:00.123Z."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{dt.microsecond // 1000:03d}Z"


def _make_event(in_incident: bool) -> tuple[dict, bool]:
    """One locally-served inference as the SDK would measure it."""
    latency = BASE_LATENCY_MS * random.lognormvariate(0.0, 0.35)
    success = True
    error = None
    if in_incident:
        latency *= random.uniform(4.0, 12.0)         # latency-spike window
        if random.random() < 0.4:                    # error burst
            success = False
            error = "inference_error"
    now = datetime.now(timezone.utc)
    ev = {
        "id": str(uuid.uuid4()),
        "ts": _iso_ms(now),
        "latencyMs": round(latency, 3),
        "preMs": round(latency * random.uniform(0.02, 0.06), 3),
        "postMs": round(latency * random.uniform(0.01, 0.04), 3),
        "success": success,
        "errorCode": error,
        "batchSize": 1,
        "region": "local",
        "inputSig": "input:1x3x224x224:float32",
    }
    return ev, success


def _make_snapshot(throughput_rpm: float) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "ts": _iso_ms(now),
        "cpuPct": round(random.uniform(20.0, 60.0), 1),
        "rssMb": round(random.uniform(350.0, 520.0), 1),
        "throughputRpm": round(throughput_rpm, 1),
        "droppedEvents": 0,
        "sdkVersion": "0.3.0",
        "pythonVersion": platform.python_version(),
        "ortVersion": "1.20.0",
        "os": platform.system() or "Linux",
        "arch": platform.machine() or "x86_64",
        "provider": "CPUExecutionProvider",
        "host": "sim-host",
    }


def _make_window(
    start: datetime, end: datetime, confidences: list[float], class_counts: dict[int, int],
) -> dict:
    n = len(confidences)
    total = max(1, sum(class_counts.values()))
    class_dist = {str(k): round(v / total, 4) for k, v in sorted(class_counts.items())}
    hist = [0] * 16
    for conf in confidences:
        hist[min(15, int(conf * 16))] += 1
    top1 = round(sum(confidences) / n, 4) if n else 0.0
    return {
        "windowStart": _iso_ms(start),
        "windowEnd": _iso_ms(end),
        "n": n,
        "inputs": {"input": {
            "mean": round(random.uniform(-0.2, 0.2), 4),
            "std": round(random.uniform(0.8, 1.2), 4),
            "min": -3.0, "max": 3.0, "nanPct": 0.0,
        }},
        "output": {
            "classDist": class_dist,
            "hist": hist,
            "entropyMean": round(random.uniform(0.8, 1.6), 3),
            "top1ConfMean": top1,
        },
    }


def _post_batch(
    client: httpx.Client, url: str, headers: dict, client_id: str,
    events: list[dict], snapshot: dict | None, window: dict | None,
) -> httpx.Response:
    body = {
        "clientId": client_id,
        "batchId": uuid.uuid4().hex,          # fresh idempotency key per batch
        "events": events,
        "snapshots": [snapshot] if snapshot else [],
        "windows": [window] if window else [],
    }
    return client.post(url, json=body, headers=headers)


def main() -> int:
    args = _parse_args()
    url = f"{args.base_url.rstrip('/')}/api/v1/telemetry/{args.deployment}/batch"
    headers = {"Authorization": f"Bearer {args.api_key}"}
    client_id = "sdk_" + uuid.uuid4().hex[:12]
    interval = 1.0 / args.rate if args.rate > 0 else 0.2

    start = time.monotonic()
    deadline = start + args.duration
    last_flush = last_snapshot = last_window = start
    snapshot_ref = start
    win_start_wall = datetime.now(timezone.utc)
    incident_until = 0.0
    in_incident = False

    pending: list[dict] = []
    confidences: list[float] = []
    class_counts: dict[int, int] = {}
    events_since_snapshot = 0

    sent_batches = accepted_events = total_events = total_fail = 0

    print(f"→ {url}\n  clientId={client_id} rate={args.rate}/s "
          f"duration={args.duration}s incidents={args.incidents}", flush=True)

    with httpx.Client(timeout=15.0) as client:
        while True:
            t = time.monotonic()
            over = t >= deadline
            if not over:
                if args.incidents and t > incident_until and random.random() < 0.01:
                    incident_until = t + random.uniform(5.0, 12.0)  # ~5–12s incident
                in_incident = t < incident_until
                ev, success = _make_event(in_incident)
                pending.append(ev)
                total_events += 1
                events_since_snapshot += 1
                if not success:
                    total_fail += 1
                # Accumulate the window's output signal (drift monitor reads this).
                cls = random.randint(0, N_CLASSES - 1)
                class_counts[cls] = class_counts.get(cls, 0) + 1
                confidences.append(random.uniform(0.5, 0.99))

            due = over or len(pending) >= BATCH_MAX_EVENTS or (t - last_flush) >= FLUSH_SEC
            if due and (pending or over):
                snapshot = window = None
                if over or (t - last_snapshot) >= SNAPSHOT_SEC:
                    elapsed = max(1e-6, t - snapshot_ref)
                    snapshot = _make_snapshot(events_since_snapshot / elapsed * 60.0)
                    last_snapshot = snapshot_ref = t
                    events_since_snapshot = 0
                if over or (t - last_window) >= WINDOW_SEC:
                    now_wall = datetime.now(timezone.utc)
                    window = _make_window(win_start_wall, now_wall, confidences, class_counts)
                    last_window = t
                    win_start_wall = now_wall
                    confidences, class_counts = [], {}
                if pending or snapshot or window:
                    r = _post_batch(client, url, headers, client_id, pending, snapshot, window)
                    sent_batches += 1
                    if r.status_code == 200:
                        accepted_events += r.json().get("accepted", {}).get("events", 0)
                    else:
                        print(f"  ! batch rejected {r.status_code}: {r.text[:140]}", flush=True)
                    pending = []
                    last_flush = t
                    print(f"  batch#{sent_batches} sent={total_events} failed={total_fail} "
                          f"accepted={accepted_events}"
                          + (" +snapshot" if snapshot else "")
                          + (" +window" if window else ""), flush=True)
            if over:
                break
            time.sleep(interval * (0.4 if in_incident else random.uniform(0.5, 1.5)))

    print(f"done — batches={sent_batches} events_sent={total_events} "
          f"failed={total_fail} accepted_events={accepted_events}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
