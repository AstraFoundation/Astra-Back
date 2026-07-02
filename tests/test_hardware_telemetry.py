"""Hardware-aware telemetry — per-hardware inference speed, GPU/CPU resource
time-series, enriched fleet inventory, and the cost/efficiency lens.

The fleet simulator injects a believable multi-accelerator serving fleet (A10G,
T4, Apple CoreML, Qualcomm NPU, hosted x86 CPU) so these views have data on a box
without a GPU; the aggregation treats those rows exactly like real astra-ai-sdk
telemetry.
"""

from __future__ import annotations

from app.services import hardware


def _simulate_fleet(client, mid: str) -> dict:
    r = client.post(
        f"/api/models/{mid}/telemetry/simulate",
        json={"count": 480, "hours": 6, "incidents": False, "fleet": True},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_cost_model_is_single_stream_dollars_per_million():
    # 1M inferences at 5 ms single-stream on a $0.526/hr T4 ≈ $0.73.
    assert hardware.est_cost_per_million(5.0, 0.526) == round(0.526 * 5.0 / 3.6, 4)
    assert hardware.est_cost_per_million(0.0, 1.0) == 0.0      # no latency → no cost
    assert hardware.est_cost_per_million(10.0, 0.0) == 0.0     # on-device → no cost


def test_classify_accelerators():
    assert hardware.classify(
        {"gpuName": "NVIDIA T4", "activeProvider": "CUDAExecutionProvider",
         "gpuCount": 1})["accelerator"] == "gpu"
    assert hardware.classify(
        {"activeProvider": "CoreMLExecutionProvider", "cpuModel": "Apple M3",
         "arch": "arm64"})["accelerator"] == "coreml"
    cpu = hardware.classify({"activeProvider": "CPUExecutionProvider", "arch": "x86_64"})
    assert cpu["accelerator"] == "cpu" and cpu["hourlyUsd"] > 0


def test_classify_npu_providers():
    """Dedicated NPUs (Qualcomm/Intel/AMD-XDNA) must classify as `npu`, not CPU —
    the pre-fix behavior collapsed every non-Apple accelerator into a CPU bucket."""
    qnn = hardware.classify(
        {"activeProvider": "QNNExecutionProvider", "arch": "aarch64",
         "cpuModel": "Qualcomm Snapdragon 8 Gen 3"})
    assert qnn["accelerator"] == "npu"
    assert qnn["hourlyUsd"] == 0.0                      # on-device edge → no cloud $
    assert hardware.classify(
        {"activeProvider": "OpenVINOExecutionProvider", "arch": "x86_64"}
    )["accelerator"] == "npu"
    assert hardware.classify(
        {"activeProvider": "VitisAIExecutionProvider", "arch": "x86_64"}
    )["accelerator"] == "npu"


def test_classify_rocm_and_directml_gpus():
    """A ROCm/AMD GPU whose name pynvml can't resolve (no NVML) still classifies
    as `gpu`, not CPU; DirectML (a GPU abstraction) is a GPU too."""
    assert hardware.classify(
        {"activeProvider": "ROCMExecutionProvider", "arch": "x86_64"}  # no gpuName
    )["accelerator"] == "gpu"
    named = hardware.classify(
        {"gpuName": "AMD Instinct MI210", "activeProvider": "ROCMExecutionProvider"})
    assert named["accelerator"] == "gpu" and named["deviceClass"] == "AMD Instinct MI210"
    assert hardware.classify(
        {"activeProvider": "DmlExecutionProvider", "arch": "x86_64"}
    )["accelerator"] == "gpu"


def test_fleet_simulate_summary(make_live_model, deploy_model, client):
    mid = make_live_model("hw-fleet.onnx")["modelId"]
    deploy_model(mid)
    summary = _simulate_fleet(client, mid)
    fleet = summary["fleet"]
    assert fleet["hosts"] == 5          # A10G, T4, CoreML, Qualcomm NPU, x86 CPU
    assert fleet["gpuHosts"] == 2
    assert fleet["events"] >= 400


def test_hardware_breakdown_groups_by_accelerator(make_live_model, deploy_model, client):
    mid = make_live_model("hw-breakdown.onnx")["modelId"]
    deploy_model(mid)
    _simulate_fleet(client, mid)

    groups = client.get(f"/api/models/{mid}/telemetry/hardware").json()
    accels = {g["accelerator"] for g in groups}
    assert {"gpu", "coreml"} <= accels                # at least GPU + CoreML present
    assert len(groups) >= 4                           # 4 fleet hosts (+ maybe hosted)

    by_class = {g["deviceClass"]: g for g in groups}
    gpu_groups = [g for g in groups if g["accelerator"] == "gpu"]
    cpu_groups = [g for g in groups if g["accelerator"] in ("cpu", "hosted")]
    assert gpu_groups and cpu_groups

    # The core promise: the GPU serves the SAME artifact faster than the CPU.
    assert min(g["p95"] for g in gpu_groups) < max(g["p95"] for g in cpu_groups)
    # Every group carries a cost estimate and a throughput capacity proxy.
    for g in groups:
        assert g["throughputPerSec"] > 0
        assert g["estCostPer1M"] >= 0
        assert g["samples"] > 0
    # GPU groups expose live accelerator utilization.
    assert any(g["avgGpuUtilPct"] and g["avgGpuUtilPct"] > 0 for g in gpu_groups)
    assert "NVIDIA" in "".join(g["gpuName"] for g in gpu_groups)


def test_resource_series_has_gpu_when_fleet_present(make_live_model, deploy_model, client):
    mid = make_live_model("hw-resources.onnx")["modelId"]
    deploy_model(mid)
    _simulate_fleet(client, mid)

    res = client.get(f"/api/models/{mid}/telemetry/resources").json()
    assert res["hasGpu"] is True
    assert res["points"], "expected bucketed resource points"
    # CPU% is always present; GPU util appears on at least one bucket.
    assert all("cpuPct" in p and "memMb" in p for p in res["points"])
    assert any(p.get("gpuUtilPct") for p in res["points"])


def test_clients_enriched_with_hardware(make_live_model, deploy_model, client):
    mid = make_live_model("hw-clients.onnx")["modelId"]
    deploy_model(mid)
    _simulate_fleet(client, mid)

    clients = client.get(f"/api/models/{mid}/telemetry/clients").json()
    assert clients
    keys = set(clients[0])
    assert {"gpuName", "cpuModel", "cpuCores", "activeProvider", "gpuUtilPct"} <= keys
    # A GPU host reports its accelerator name + a live util sample.
    gpu_hosts = [c for c in clients if c["gpuName"]]
    assert gpu_hosts and any(c["gpuUtilPct"] for c in gpu_hosts)


def test_npu_host_grouped_as_npu(make_live_model, deploy_model, client):
    """The Qualcomm NPU serving host surfaces as its own `npu` accelerator group —
    before the fix it was mislabeled a CPU host."""
    mid = make_live_model("hw-npu.onnx")["modelId"]
    deploy_model(mid)
    _simulate_fleet(client, mid)

    groups = client.get(f"/api/models/{mid}/telemetry/hardware").json()
    npu = [g for g in groups if g["accelerator"] == "npu"]
    assert npu, f"expected an npu group, got {[g['accelerator'] for g in groups]}"
    g = npu[0]
    assert "NPU" in g["deviceClass"]
    assert g["provider"] == "QNNExecutionProvider"
    assert g["throughputPerSec"] > 0 and g["samples"] > 0
    assert g["estCostPer1M"] == 0.0                    # on-device → no cloud cost


def test_driver_version_persisted_and_returned(make_live_model, deploy_model, client):
    """`driverVersion` used to be silently dropped at ingest (absent from the
    snapshot schema). It must now round-trip SDK → ingest → /clients."""
    mid = make_live_model("hw-driver.onnx")["modelId"]
    deploy_model(mid)
    _simulate_fleet(client, mid)

    clients = client.get(f"/api/models/{mid}/telemetry/clients").json()
    assert "driverVersion" in clients[0]               # field is in the read contract
    gpu_hosts = [c for c in clients if c["gpuName"]]
    assert gpu_hosts, "expected simulated GPU hosts"
    # At least one GPU host carries the NVIDIA driver version end-to-end.
    assert any(c["driverVersion"] for c in gpu_hosts), \
        f"driverVersion lost: {[(c['gpuName'], c['driverVersion']) for c in gpu_hosts]}"
    a10g = next((c for c in gpu_hosts if "A10G" in c["gpuName"]), None)
    assert a10g and a10g["driverVersion"] == "550.90.07"


def test_no_traffic_returns_empty_hardware_views(real_model, client):
    """A never-served model has no fleet — endpoints return empty, not errors."""
    mid = real_model["modelId"]
    assert client.get(f"/api/models/{mid}/telemetry/hardware").json() == []
    res = client.get(f"/api/models/{mid}/telemetry/resources").json()
    assert res == {"points": [], "hasGpu": False}
