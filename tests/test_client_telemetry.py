"""Client-telemetry ingestion (astra-ai-sdk path): batch endpoint, key-authed
artifact pull, drift detection from window stats, and aggregation parity."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.config import iso


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _event(ts: datetime, latency: float = 5.0, **kw) -> dict:
    return {
        "ts": iso(ts), "latencyMs": latency, "preMs": 0.4, "postMs": 0.2,
        "success": kw.get("success", True), "errorCode": kw.get("errorCode"),
        "batchSize": 1, "region": "local",
        "inputSig": "input:1x3x8x8:float32",
    }


def _snapshot(ts: datetime) -> dict:
    return {
        "ts": iso(ts), "cpuPct": 31.5, "rssMb": 412.0, "throughputRpm": 120.0,
        "droppedEvents": 0, "sdkVersion": "0.2.0", "pythonVersion": "3.13.1",
        "ortVersion": "1.26.0", "os": "Darwin", "arch": "arm64",
        "provider": "CPUExecutionProvider", "host": "test-host",
    }


def _window(start: datetime, class_dist: dict, mean: float = 0.0) -> dict:
    return {
        "windowStart": iso(start), "windowEnd": iso(start + timedelta(minutes=1)),
        "n": 50,
        "inputs": {"input": {"mean": mean, "std": 1.0, "min": -3.0, "max": 3.0,
                             "nanPct": 0.0}},
        "output": {"classDist": class_dist, "hist": [0, 1, 2, 5, 9, 14, 10, 5, 2, 1, 1,
                                                     0, 0, 0, 0, 0],
                   "entropyMean": 1.2, "top1ConfMean": 0.81},
    }


def _deploy(client, model_id: str) -> tuple[str, str]:
    r = client.post(f"/api/models/{model_id}/deployments",
                    json={"region": "ap-northeast-2"})
    assert r.status_code == 200, r.text
    data = r.json()
    return data["deployment"]["id"], data["apiKey"]


@pytest.fixture(scope="module")
def live_dep(client, make_live_model):
    body = make_live_model("client-telemetry-fixture.onnx")
    dep_id, api_key = _deploy(client, body["modelId"])
    return {"model_id": body["modelId"], "dep_id": dep_id, "key": api_key}


def test_batch_happy_path(client, live_dep):
    now = _now()
    r = client.post(
        f"/api/v1/telemetry/{live_dep['dep_id']}/batch",
        headers={"Authorization": f"Bearer {live_dep['key']}"},
        json={
            "clientId": "sdk_test01",
            "events": [_event(now - timedelta(seconds=i)) for i in range(10)],
            "snapshots": [_snapshot(now)],
            "windows": [_window(now - timedelta(minutes=1), {"3": 0.6, "7": 0.4})],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] == {"events": 10, "snapshots": 1, "windows": 1}
    assert body["dropped"] == 0

    meta = client.get(f"/api/models/{live_dep['model_id']}/telemetry/meta").json()
    assert meta["source"] == "live"
    assert meta["sources"]["client"] >= 10
    assert meta["lastSnapshotAt"] is not None


def test_batch_auth_failures(client, live_dep):
    r = client.post(
        f"/api/v1/telemetry/{live_dep['dep_id']}/batch",
        headers={"Authorization": "Bearer astra_sk_live_wrong"},
        json={"clientId": "x", "events": []},
    )
    assert r.status_code == 401
    r = client.post(
        f"/api/v1/telemetry/dep_nonexistent/batch",
        headers={"Authorization": f"Bearer {live_dep['key']}"},
        json={"clientId": "x", "events": []},
    )
    assert r.status_code == 404


def test_batch_ts_clamping(client, live_dep):
    now = _now()
    r = client.post(
        f"/api/v1/telemetry/{live_dep['dep_id']}/batch",
        headers={"Authorization": f"Bearer {live_dep['key']}"},
        json={
            "clientId": "sdk_test01",
            "events": [
                _event(now),
                _event(now - timedelta(days=40)),     # older than 30d clamp → dropped
                _event(now + timedelta(hours=2)),     # future → dropped
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"]["events"] == 1
    assert body["dropped"] == 2


def test_batch_too_large(client, live_dep):
    now = _now()
    events = [_event(now) for _ in range(501)]
    r = client.post(
        f"/api/v1/telemetry/{live_dep['dep_id']}/batch",
        headers={"Authorization": f"Bearer {live_dep['key']}"},
        json={"clientId": "sdk_test01", "events": events},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "batch_too_large"


def test_batch_idempotent_resend(client, live_dep):
    """A durable-spool segment re-sent after a reconnect/restart carries the same
    batchId — the server must dedup it, never double-counting."""
    now = _now()
    url = f"/api/v1/telemetry/{live_dep['dep_id']}/batch"
    hdr = {"Authorization": f"Bearer {live_dep['key']}"}
    batch = {
        "clientId": "sdk_idem",
        "batchId": "batch_idem_001",
        "events": [_event(now - timedelta(seconds=i)) for i in range(6)],
    }

    def client_count() -> int:
        meta = client.get(
            f"/api/models/{live_dep['model_id']}/telemetry/meta").json()
        return meta["sources"]["client"]

    before = client_count()
    r1 = client.post(url, headers=hdr, json=batch)
    assert r1.status_code == 200, r1.text
    assert r1.json()["accepted"]["events"] == 6
    after_first = client_count()
    assert after_first - before == 6

    # Re-send the identical batch (same batchId) — deduped, acked, not re-inserted.
    r2 = client.post(url, headers=hdr, json=batch)
    assert r2.status_code == 200, r2.text
    assert r2.json()["accepted"]["events"] == 6      # prior accepted counts echoed
    assert client_count() == after_first, "re-sent batch must not double-count"


def test_client_events_feed_kpi(client, live_dep):
    kpi = client.get(
        f"/api/models/{live_dep['model_id']}/telemetry/kpi?range=1h").json()
    assert kpi["requestsPerMin"]["value"] > 0


def test_artifact_pull_with_key(client, live_dep):
    info = client.get(
        f"/api/v1/artifacts/{live_dep['dep_id']}/info",
        headers={"Authorization": f"Bearer {live_dep['key']}"},
    )
    assert info.status_code == 200, info.text
    meta = info.json()
    assert meta["sizeBytes"] > 0 and len(meta["sha256"]) == 64

    r = client.get(
        f"/api/v1/artifacts/{live_dep['dep_id']}",
        headers={"Authorization": f"Bearer {live_dep['key']}"},
    )
    assert r.status_code == 200
    assert len(r.content) == meta["sizeBytes"]
    assert r.headers["ETag"] == f'"{meta["sha256"]}"'

    cookie_r = client.get(f"/api/models/{live_dep['model_id']}/artifact")
    assert cookie_r.status_code == 200
    assert r.content == cookie_r.content, "key-authed bytes must equal cookie-authed bytes"

    not_mod = client.get(
        f"/api/v1/artifacts/{live_dep['dep_id']}",
        headers={
            "Authorization": f"Bearer {live_dep['key']}",
            "If-None-Match": r.headers["ETag"],
        },
    )
    assert not_mod.status_code == 304


def test_artifact_pull_requires_key(client, live_dep):
    r = client.get(f"/api/v1/artifacts/{live_dep['dep_id']}")
    assert r.status_code == 401


def test_breakdown_and_output_stats(client, live_dep):
    r = client.get(
        f"/api/models/{live_dep['model_id']}/telemetry/breakdown?range=24h").json()
    assert r["points"], "client events with pre/post latencies must produce points"
    pt = r["points"][0]
    assert pt["preprocessMs"] > 0 and pt["inferenceMs"] > 0

    stats = client.get(
        f"/api/models/{live_dep['model_id']}/telemetry/output-stats?range=24h").json()
    assert stats["windows"] >= 1
    assert stats["meanConfidence"] == pytest.approx(0.81, abs=0.01)
    assert any(b["count"] > 0 for b in stats["bins"])

    hosts = client.get(
        f"/api/models/{live_dep['model_id']}/telemetry/clients").json()
    assert hosts and hosts[0]["host"] == "test-host"
    assert hosts[0]["sdkVersion"] == "0.2.0"


def test_prediction_and_input_drift(client, make_live_model):
    """Seed reference windows, then shifted windows → monitor raises drift alerts."""
    from app.db import open_session
    from app.services.drift_monitor import drift_monitor_pass

    body = make_live_model("drift-fixture.onnx")
    dep_id, api_key = _deploy(client, body["modelId"])
    now = _now()

    # 5 reference windows: stable class distribution + input mean ~0.
    ref_windows = [
        _window(now - timedelta(minutes=30 - i), {"1": 0.5, "2": 0.5}, mean=0.0)
        for i in range(5)
    ]
    # Latest window: flipped distribution + input mean shifted by 5 sigma.
    shifted = _window(now - timedelta(minutes=1), {"9": 0.9, "1": 0.1}, mean=5.0)
    r = client.post(
        f"/api/v1/telemetry/{dep_id}/batch",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"clientId": "sdk_drift", "windows": ref_windows + [shifted]},
    )
    assert r.status_code == 200, r.text

    with open_session() as s:
        drift_monitor_pass(s)

    alerts = client.get(f"/api/models/{body['modelId']}/telemetry/alerts").json()
    titles = {a["title"] for a in alerts}
    assert "prediction drift" in titles, titles
    assert "input distribution shift" in titles, titles


def test_gpu_snapshot_driver_version_round_trips(client, live_dep):
    """A GPU host's snapshot carrying driverVersion is accepted through the real
    batch endpoint and surfaced by /clients. Regression: driverVersion was absent
    from the ClientSnapshot schema, so Pydantic silently dropped it at ingest."""
    now = _now()
    snap = _snapshot(now)
    snap.update({
        "os": "Linux", "arch": "x86_64",
        "provider": "CUDAExecutionProvider", "activeProvider": "CUDAExecutionProvider",
        "availableProviders": "CUDAExecutionProvider,CPUExecutionProvider",
        "cpuModel": "Intel Xeon Platinum 8259CL", "cpuCores": 8, "ramTotalMb": 32000.0,
        "gpuName": "NVIDIA A100-SXM4-40GB", "gpuCount": 1, "gpuMemTotalMb": 40000.0,
        "cudaVersion": "12.4", "driverVersion": "550.90.07",
        "gpuUtilPct": 68.0, "gpuMemUsedMb": 12000.0, "gpuTempC": 60.0,
    })
    r = client.post(
        f"/api/v1/telemetry/{live_dep['dep_id']}/batch",
        headers={"Authorization": f"Bearer {live_dep['key']}"},
        json={"clientId": "sdk_gpu01", "snapshots": [snap]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["accepted"]["snapshots"] == 1

    hosts = client.get(f"/api/models/{live_dep['model_id']}/telemetry/clients").json()
    host = next((h for h in hosts if h["clientId"] == "sdk_gpu01"), None)
    assert host is not None, "GPU snapshot host not found in /clients"
    assert host["driverVersion"] == "550.90.07"      # previously-dropped field survives
    assert host["gpuName"] == "NVIDIA A100-SXM4-40GB"
    assert host["cudaVersion"] == "12.4"
