#!/usr/bin/env python3
"""End-to-end proof that the ON-DEVICE telemetry closed loop ACTUALLY works.

Astra is on-device-only: the server never runs a user's model. A deployment's
only external traffic is the astra-ai-sdk shipping telemetry *batches* to
POST /api/v1/telemetry/{deployment_id}/batch with a Bearer API key. This script
provisions a model end-to-end, then drives that public telemetry path and asserts
the whole dashboard/closed-loop reacts — using ONLY the public API.

Run against a live backend (default http://localhost:8000) started with the
inline drift monitor on a short interval, e.g.:

    ASTRA_DB_PATH=/tmp/astra-verify/db.sqlite \
    ASTRA_STORAGE_DIR=/tmp/astra-verify/storage \
    ASTRA_WORK_DIR=/tmp/astra-verify/work \
    ASTRA_FAST_PIPELINE=1 ASTRA_INLINE_JOBS=1 \
    ASTRA_MONITOR_INLINE_ENABLED=1 ASTRA_MONITOR_INTERVAL_SEC=5 \
    ASTRA_COOKIE_SECURE=0 ASTRA_RATE_LIMIT_ENABLED=0 \
    uvicorn app.main:app --port 8000

    python3 scripts/verify_closed_loop.py --base http://localhost:8000

Checklist proven (each step asserts against the public API only):
  1.  signup → session cookie
  2.  model import → real fast pipeline completes
  3.  deployment + API key minted
  4.  SDK telemetry batch (~60 on-device events incl. failed ones + a snapshot +
      a window) is accepted through POST /api/v1/telemetry/{dep}/batch (Bearer key)
  5.  telemetry flips to source=live; KPI/series/percentiles are consistent
  6.  SSE stream delivers a snapshot frame
  7.  the success:false events are recorded (live error rate > 0)
  8.  the drift monitor pass raises a REAL "5xx error spike" alert
  9.  deployment live metrics (qps / errorsPct / lastEventAt) are maintained
  10. re-POSTing the SAME batch (same batchId) is deduped — the client event
      count does NOT increase (idempotency)

Exit code 0 = closed loop verified; 1 = a step failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

import httpx

CHECK: list[tuple[str, bool]] = []


def step(name: str, ok: bool, detail: str = "") -> None:
    CHECK.append((name, ok))
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        finish()


def finish() -> None:
    failed = [n for n, ok in CHECK if not ok]
    print("\n" + ("CLOSED LOOP VERIFIED" if not failed else f"FAILED: {failed}"))
    sys.exit(1 if failed else 0)


def wait_until(fn, timeout: float, interval: float = 0.5):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = fn()
        if last:
            return last
        time.sleep(interval)
    return last


def _iso_ms(dt: datetime) -> str:
    """ISO8601 UTC ms + Z, e.g. 2026-07-01T12:00:00.123Z (server-accepted form)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{dt.microsecond // 1000:03d}Z"


def build_batch(client_id: str, n_events: int = 60, n_fail: int = 6) -> dict:
    """One believable on-device SDK batch: ~60 measured inferences (a handful
    failed with errorCode "inference_error"), a host snapshot, and a window of
    input/output distribution stats. Timestamps are recent so the batch lands
    inside the drift monitor's rolling window and none are dropped."""
    now = datetime.now(timezone.utc)
    events = []
    for i in range(n_events):
        ts = now - timedelta(seconds=(n_events - i) * 0.5)   # recent, spread out
        failed = i < n_fail
        events.append({
            "id": str(uuid.uuid4()),
            "ts": _iso_ms(ts),
            "latencyMs": round((45.0 + i) if failed else (8.0 + (i % 7) * 1.5), 3),
            "preMs": 0.4,
            "postMs": 0.2,
            "success": not failed,
            "errorCode": "inference_error" if failed else None,
            "batchSize": 1,
            "region": "local",
            "inputSig": "input:1x3x224x224:float32",
        })
    snapshot = {
        "ts": _iso_ms(now),
        "cpuPct": 34.0, "rssMb": 420.0, "throughputRpm": 120.0, "droppedEvents": 0,
        "sdkVersion": "0.3.0", "pythonVersion": "3.12.0", "ortVersion": "1.20.0",
        "os": "Linux", "arch": "x86_64", "provider": "CPUExecutionProvider",
        "host": "verify-host",
    }
    window = {
        "windowStart": _iso_ms(now - timedelta(seconds=60)),
        "windowEnd": _iso_ms(now),
        "n": n_events,
        "inputs": {"input": {"mean": 0.0, "std": 1.0, "min": -3.0, "max": 3.0, "nanPct": 0.0}},
        "output": {
            "classDist": {"3": 0.6, "7": 0.4},
            "hist": [0, 1, 2, 3, 5, 8, 10, 12, 10, 8, 5, 3, 2, 1, 0, 0],
            "entropyMean": 1.2, "top1ConfMean": 0.81,
        },
    }
    return {
        "clientId": client_id,
        "batchId": uuid.uuid4().hex,          # idempotency key (server dedups on it)
        "events": events,
        "snapshots": [snapshot],
        "windows": [window],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--monitor-interval", type=float, default=5.0,
                    help="ASTRA_MONITOR_INTERVAL_SEC the server was started with")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    c = httpx.Client(base_url=base, timeout=60.0)

    # 1 — auth
    email = f"verify+{uuid.uuid4().hex[:8]}@astra.dev"
    r = c.post("/api/auth/signup", json={
        "email": email, "password": "verify-pass-1234", "name": "Verify"})
    step("signup issues a session", r.status_code == 200, email)

    # 2 — import a model, real pipeline
    r = c.post("/api/models/import", json={"fileName": "closed-loop-verify.onnx"})
    step("model import accepted", r.status_code == 200)
    body = r.json()
    mid, rid = body["modelId"], body["runId"]

    deadline = time.time() + 180
    status = c.get(f"/api/models/{mid}/ingestion/{rid}").json()
    while status.get("status") == "streaming" and time.time() < deadline:
        time.sleep(0.5)
        status = c.get(f"/api/models/{mid}/ingestion/{rid}").json()
    step("pipeline completed", bool(status) and status["status"] == "completed",
         f"status={status.get('status') if status else 'n/a'}")
    c.post(f"/api/models/{mid}/ingestion/complete")

    # 3 — deployment + key
    r = c.post(f"/api/models/{mid}/deployments", json={"region": "ap-northeast-2"})
    step("deployment created + key minted", r.status_code == 200)
    dep = r.json()
    dep_id, api_key = dep["deployment"]["id"], dep["apiKey"]
    auth = {"Authorization": f"Bearer {api_key}"}

    # 4 — on-device SDK telemetry: ship one batch of ~60 measured inferences
    # (a few failed) plus a host snapshot and a distribution window. This is the
    # ONLY external traffic a deployment sees now — the SDK closing the loop.
    client_id = "sdk_" + uuid.uuid4().hex[:12]
    batch = build_batch(client_id, n_events=60, n_fail=6)
    rr = c.post(f"/api/v1/telemetry/{dep_id}/batch", headers=auth, json=batch)
    step("telemetry batch accepted (Bearer key)", rr.status_code == 200,
         f"http {rr.status_code}")
    acc = rr.json()
    step("batch accepted 60 events + 1 snapshot + 1 window, 0 dropped",
         acc["accepted"]["events"] == 60 and acc["accepted"]["snapshots"] == 1
         and acc["accepted"]["windows"] == 1 and acc["dropped"] == 0,
         f"accepted={acc['accepted']} dropped={acc['dropped']}")

    # 5 — live aggregation
    meta = c.get(f"/api/models/{mid}/telemetry/meta").json()
    step("telemetry source flips to live", meta["source"] == "live",
         f"sources={meta.get('sources')}")
    kpi = c.get(f"/api/models/{mid}/telemetry/kpi?range=1h").json()
    step("KPI requests/min > 0", kpi["requestsPerMin"]["value"] > 0,
         f"req/min={kpi['requestsPerMin']['value']}")
    series = c.get(f"/api/models/{mid}/telemetry/series?range=1h").json()
    step("series has non-empty buckets", any(p["requests"] > 0 for p in series))
    pct = c.get(f"/api/models/{mid}/telemetry/percentiles?range=1h").json()
    v = pct["values"]
    step("percentiles ordered p50<=p95<=p99",
         v["p50"] <= v["p95"] <= v["p99"],
         f"p50={v['p50']} p95={v['p95']} p99={v['p99']}")

    # 6 — SSE
    got_snapshot = False
    try:
        with c.stream("GET", f"/api/models/{mid}/telemetry/stream",
                      timeout=15.0) as resp:
            event = None
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and event == "snapshot":
                    payload = json.loads(line.split(":", 1)[1])
                    got_snapshot = "source" in payload
                    break
    except httpx.HTTPError:
        pass
    step("SSE stream delivers a snapshot frame", got_snapshot)

    # 7 — the success:false events landed as real failed events
    step("failed on-device events recorded (error rate > 0)",
         kpi["errorRate"]["value"] > 0, f"errorRate={kpi['errorRate']['value']}")

    # 8 — the monitor pass raises a real alert (error rate > threshold)
    def find_alert():
        alerts = c.get(f"/api/models/{mid}/telemetry/alerts").json()
        return [a for a in alerts if a["title"] == "5xx error spike"] or None

    alerts = wait_until(find_alert, timeout=args.monitor_interval * 4 + 10)
    step("drift monitor raised '5xx error spike'", bool(alerts),
         alerts[0]["body"][:70] if alerts else "no alert within window")

    # 9 — live deployment metrics maintained by the monitor
    deps = c.get(f"/api/models/{mid}/telemetry/deployments").json()
    d = deps[0] if deps else {}
    step("deployment live metrics updated",
         bool(d) and d["qps"] > 0 and d["errorsPct"] > 1.0,
         f"qps={d.get('qps')} errors%={d.get('errorsPct')}")

    # 10 — idempotency: re-POST the IDENTICAL batch (same batchId). The server
    # dedups on batchId, so the client event count must NOT increase.
    def client_events() -> int:
        return c.get(f"/api/models/{mid}/telemetry/meta").json()["sources"]["client"]

    before = client_events()
    rr2 = c.post(f"/api/v1/telemetry/{dep_id}/batch", headers=auth, json=batch)
    acc2 = rr2.json()
    after = client_events()
    step("re-POST of same batchId is deduped (no double-count)",
         rr2.status_code == 200 and after == before
         and acc2["accepted"]["events"] == acc["accepted"]["events"],
         f"before={before} after={after} reAccepted={acc2['accepted']['events']}")

    finish()


if __name__ == "__main__":
    main()
