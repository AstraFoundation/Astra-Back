"""Live telemetry aggregation + empty (no-traffic) parity.

Once a deployed model serves traffic, the telemetry endpoints aggregate the real
events; a model that has never been served returns empty shapes (NOT the
benchmark) — covered by test_contract_telemetry and re-asserted here at the
source level.
"""

from __future__ import annotations


def _simulate(client, mid: str, **body) -> dict:
    r = client.post(f"/api/models/{mid}/telemetry/simulate", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_simulate_makes_dashboard_live(make_live_model, deploy_model, client):
    mid = make_live_model("live-a.onnx")["modelId"]
    deploy_model(mid)

    summary = _simulate(client, mid, count=300, hours=6, incidents=False)
    assert summary["events"] >= 200              # synthesized on-device (client) traffic
    assert "monitor" in summary                  # a monitor pass ran

    assert client.get(f"/api/models/{mid}/telemetry/meta").json()["source"] == "live"

    kpi = client.get(f"/api/models/{mid}/telemetry/kpi").json()
    assert kpi["requestsPerMin"]["value"] > 0
    assert kpi["p95LatencyMs"]["value"] > 0
    assert set(kpi["accuracyDrift"]) == {"value", "note"}

    series = client.get(f"/api/models/{mid}/telemetry/series").json()
    assert len(series) == 48                      # 24h → 48 buckets
    assert any(p["requests"] > 0 for p in series)

    pct = client.get(f"/api/models/{mid}/telemetry/percentiles").json()
    assert len(pct["p50"]) == len(pct["p95"]) == len(pct["p99"]) == 48
    v = pct["values"]
    assert v["p50"] <= v["p95"] <= v["p99"]


def test_range_param_controls_buckets(make_live_model, deploy_model, client):
    mid = make_live_model("live-b.onnx")["modelId"]
    deploy_model(mid)
    _simulate(client, mid, count=200, hours=1, incidents=False)

    assert len(client.get(f"/api/models/{mid}/telemetry/series?range=1h").json()) == 60
    assert len(client.get(f"/api/models/{mid}/telemetry/series?range=6h").json()) == 72
    assert len(client.get(f"/api/models/{mid}/telemetry/series?range=7d").json()) == 84


def test_unserved_model_is_empty_not_benchmark(real_model, client):
    """A model with no traffic returns empty telemetry (source "none"), never the
    benchmark reshaped to look like live data."""
    mid = real_model["modelId"]
    assert client.get(f"/api/models/{mid}/telemetry/meta").json()["source"] == "none"

    kpi = client.get(f"/api/models/{mid}/telemetry/kpi").json()
    assert kpi["requestsPerMin"]["value"] == 0.0
    assert kpi["p95LatencyMs"]["value"] == 0.0
    assert client.get(f"/api/models/{mid}/telemetry/series").json() == []
