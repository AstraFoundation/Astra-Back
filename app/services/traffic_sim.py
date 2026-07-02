"""In-process traffic generator — make the dashboard live without real users yet.

A demo burst synthesizes a believable recent history (a diurnal-ish request curve
with lognormal latency jitter and the occasional incident — a latency spike or an
error burst) so the charts, percentiles, and drift alerts have something
real-shaped to show immediately.

Everything is written as ordinary InferenceEventRow rows with source="client" —
identical to what the on-device SDK's closed-loop telemetry ships to
/api/v1/telemetry/{id}/batch — so the aggregation + drift monitor treat simulated
and organic on-device traffic exactly the same. (Astra never runs user models
server-side, so there is no server-measured latency to seed from.)
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from app.config import iso
from app.dbmodels import (
    DeploymentRow,
    InferenceEventRow,
    ModelRow,
    ResultCacheRow,
    TelemetrySnapshotRow,
    TelemetryWindowStatsRow,
)


def _benchmark_p50(session: Session, model_id: str) -> float | None:
    row = session.exec(
        select(ResultCacheRow).where(
            ResultCacheRow.model_id == model_id, ResultCacheRow.kind == "benchmark",
        )
    ).first()
    if not row:
        return None
    try:
        return float(json.loads(row.payload)["compressed"]["p50"])
    except (KeyError, ValueError, TypeError):
        return None


def _pick_incidents(now: datetime, hours: int) -> list[tuple[datetime, datetime, str]]:
    """0–2 short incident windows somewhere in the last `hours`."""
    windows: list[tuple[datetime, datetime, str]] = []
    for kind in random.sample(["latency", "errors"], k=random.randint(0, 2)):
        start_h = random.uniform(0.2, hours - 0.6)
        dur_min = random.uniform(12, 30)
        w0 = now - timedelta(hours=hours) + timedelta(hours=start_h)
        windows.append((w0, w0 + timedelta(minutes=dur_min), kind))
    return windows


def _diurnal_weight(dt: datetime) -> float:
    """A gentle day/night demand shape (busier midday, quieter at night)."""
    h = dt.hour + dt.minute / 60.0
    return 0.45 + 0.55 * (0.5 - 0.5 * math.cos((h - 3) / 24.0 * 2 * math.pi))


def simulate_burst(
    session: Session,
    dep: DeploymentRow,
    model: ModelRow,
    *,
    count: int = 240,
    hours: int = 6,
    incidents: bool = True,
) -> dict:
    """Generate `count` events for a deployment over the last `hours`. Returns a
    small summary. Commits once (bulk insert) for speed."""
    now = datetime.now(timezone.utc)
    rows: list[InferenceEventRow] = []

    # Base latency comes from the on-device benchmark (measured by the SDK /
    # pipeline), never from a server-side inference — Astra doesn't run user
    # models. Fall back to a small default when no benchmark exists yet.
    base_latency = _benchmark_p50(session, model.id) or 5.0

    # Synthesize the recent history with jitter + incidents.
    windows = _pick_incidents(now, hours) if incidents else []
    errors = 0
    for _ in range(max(0, count)):
        # Weight timestamps toward busier hours so the request curve looks real.
        frac = random.random()
        ts_dt = now - timedelta(hours=hours) + timedelta(hours=hours * frac)
        if random.random() > _diurnal_weight(ts_dt):
            ts_dt = now - timedelta(hours=hours) + timedelta(hours=hours * random.random())
        lat = base_latency * math.exp(random.gauss(0.0, 0.28))
        success, err_code = True, None
        for w0, w1, kind in windows:
            if w0 <= ts_dt < w1:
                if kind == "latency":
                    lat *= random.uniform(3.0, 6.0)
                elif kind == "errors" and random.random() < 0.3:
                    success, err_code, lat = False, "inference_error", 0.0
        if not success:
            errors += 1
        rows.append(InferenceEventRow(
            user_id=model.user_id, model_id=model.id, deployment_id=dep.id,
            ts=iso(ts_dt), latency_ms=round(lat, 3), success=success,
            error_code=err_code, region=dep.region, source="client",
        ))

    session.add_all(rows)
    session.commit()
    return {
        "deploymentId": dep.id,
        "events": len(rows),
        "errors": errors,
        "incidents": [k for _, _, k in windows],
    }


# ── hardware-aware fleet simulation ──────────────────────────────────────────
# Demo only (gated by ASTRA_TELEMETRY_SIM_ENABLED): on a machine without an
# NVIDIA GPU a single dev box can't produce GPU/multi-hardware data, so this
# injects a believable on-device fleet — the same compressed artifact running on a
# T4, an A10G, an Apple CoreML box and an x86 CPU host — so the per-hardware
# speed + GPU resource views are visible and verifiable. Rows are written as
# ordinary client telemetry (events + snapshots + windows), identical in shape
# to what astra-ai-sdk ships, so aggregation treats them exactly like real fleets.

# Each profile: relative inference cost + the hardware identity the dashboard
# attributes it to. Latencies are deliberately ordered GPU < CoreML < CPU so the
# per-hardware comparison and the $/1M cost lens tell a real story.
_FLEET: list[dict] = [
    {
        "host": "ip-10-0-3-12.gpu", "infer": 1.9, "cpu": (8, 16),
        "runtime": {
            "os": "Linux", "arch": "x86_64", "provider": "CUDAExecutionProvider",
            "activeProvider": "CUDAExecutionProvider",
            "availableProviders": "TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider",
            "cpuModel": "Intel Xeon Platinum 8259CL", "cpuCores": 8, "ramTotalMb": 32000.0,
            "gpuName": "NVIDIA A10G", "gpuCount": 1, "gpuMemTotalMb": 24000.0,
            "cudaVersion": "12.4", "driverVersion": "550.90.07",
            "ort": "1.18.1", "python": "3.11.9",
        },
        "gpu": {"util": (62, 84), "mem": (2600, 3400), "temp": (54, 68)},
    },
    {
        "host": "ip-10-0-4-21.gpu", "infer": 3.2, "cpu": (10, 18),
        "runtime": {
            "os": "Linux", "arch": "x86_64", "provider": "CUDAExecutionProvider",
            "activeProvider": "CUDAExecutionProvider",
            "availableProviders": "CUDAExecutionProvider,CPUExecutionProvider",
            "cpuModel": "Intel Xeon E5-2686 v4", "cpuCores": 4, "ramTotalMb": 16000.0,
            "gpuName": "NVIDIA T4", "gpuCount": 1, "gpuMemTotalMb": 16000.0,
            "cudaVersion": "12.2", "driverVersion": "535.161.08",
            "ort": "1.18.1", "python": "3.10.14",
        },
        "gpu": {"util": (48, 72), "mem": (1900, 2600), "temp": (49, 63)},
    },
    {
        "host": "edge-snapdragon.local", "infer": 4.6, "cpu": (18, 34),
        "runtime": {
            "os": "Linux", "arch": "aarch64", "provider": "QNNExecutionProvider",
            "activeProvider": "QNNExecutionProvider",
            "availableProviders": "QNNExecutionProvider,CPUExecutionProvider",
            "cpuModel": "Qualcomm Snapdragon 8 Gen 3", "cpuCores": 8, "ramTotalMb": 12000.0,
            "ort": "1.18.1", "python": "3.11.8",
        },
        "gpu": None,
    },
    {
        "host": "edge-m3.local", "infer": 6.8, "cpu": (22, 38),
        "runtime": {
            "os": "Darwin", "arch": "arm64", "provider": "CoreMLExecutionProvider",
            "activeProvider": "CoreMLExecutionProvider",
            "availableProviders": "CoreMLExecutionProvider,CPUExecutionProvider",
            "cpuModel": "Apple M3", "cpuCores": 8, "ramTotalMb": 16000.0,
            "ort": "1.18.0", "python": "3.12.4",
        },
        "gpu": None,
    },
    {
        "host": "ip-10-0-1-9.cpu", "infer": 27.5, "cpu": (55, 80),
        "runtime": {
            "os": "Linux", "arch": "x86_64", "provider": "CPUExecutionProvider",
            "activeProvider": "CPUExecutionProvider",
            "availableProviders": "CPUExecutionProvider",
            "cpuModel": "Intel Xeon Platinum 8375C", "cpuCores": 2, "ramTotalMb": 8000.0,
            "ort": "1.18.1", "python": "3.11.9",
        },
        "gpu": None,
    },
]

_SDK_VERSION = "0.4.0"


def _fleet_output() -> dict:
    """A believable classifier output summary for a window (top-k class share +
    confidence histogram), matching the SDK's window stat shape."""
    classes = random.sample(range(1000), 6)
    shares = sorted((random.random() for _ in classes), reverse=True)
    total = sum(shares) or 1.0
    class_dist = {
        str(c): round(s / total, 4)
        for c, s in zip(classes, shares, strict=True)
    }
    hist = [0] * 16
    for _ in range(64):
        b = min(15, int(abs(random.gauss(11, 3))))
        hist[b] += 1
    return {
        "classDist": class_dist,
        "hist": hist,
        "top1ConfMean": round(random.uniform(0.62, 0.92), 4),
        "entropyMean": round(random.uniform(0.4, 1.6), 4),
    }


def simulate_fleet(
    session: Session,
    dep: DeploymentRow,
    model: ModelRow,
    *,
    count: int = 480,
    hours: int = 6,
) -> dict:
    """Inject a multi-hardware serving fleet (events + GPU snapshots + windows)
    for a deployment. Returns a small summary. Demo affordance."""
    now = datetime.now(timezone.utc)
    rows: list = []
    per_host = max(1, count // len(_FLEET))
    snaps_per_host = 8
    last_event_ts: str | None = None

    for fi, profile in enumerate(_FLEET):
        client_id = f"fleet_{fi}_{profile['host'].split('.')[0]}"
        rt = profile["runtime"]
        base = profile["infer"]

        # 1) Inference events spread over the window (lognormal jitter + rare error).
        for _ in range(per_host):
            frac = random.random()
            ts_dt = now - timedelta(hours=hours) + timedelta(hours=hours * frac)
            if random.random() > _diurnal_weight(ts_dt):
                ts_dt = now - timedelta(hours=hours) + timedelta(
                    hours=hours * random.random())
            infer = base * math.exp(random.gauss(0.0, 0.22))
            success = random.random() > 0.004
            ts = iso(ts_dt)
            rows.append(InferenceEventRow(
                user_id=model.user_id, model_id=model.id, deployment_id=dep.id,
                ts=ts, latency_ms=round(infer if success else 0.0, 3),
                success=success, error_code=None if success else "inference_error",
                region=dep.region, source="client",
                latency_pre_ms=round(random.uniform(0.2, 0.7), 3),
                latency_post_ms=round(random.uniform(0.1, 0.4), 3),
                client_id=client_id, input_sig="input:1x3x224x224:float32",
            ))
            if last_event_ts is None or ts > last_event_ts:
                last_event_ts = ts

        # 2) Periodic system snapshots — carry the hardware fingerprint + GPU sample.
        rpm = per_host / max(1e-6, hours * 60.0)
        for si in range(snaps_per_host):
            ts_dt = now - timedelta(hours=hours) + timedelta(
                hours=hours * (si + 0.5) / snaps_per_host)
            cpu_lo, cpu_hi = profile["cpu"]
            gpu = profile["gpu"]
            snap = TelemetrySnapshotRow(
                user_id=model.user_id, model_id=model.id, deployment_id=dep.id,
                client_id=client_id, ts=iso(ts_dt),
                cpu_pct=round(random.uniform(cpu_lo, cpu_hi), 2),
                rss_mb=round(random.uniform(220, 680), 1),
                throughput_rpm=round(rpm * random.uniform(0.8, 1.2), 2),
                dropped_events=0, sdk_version=_SDK_VERSION,
                runtime_json=json.dumps({**rt, "host": profile["host"]}),
            )
            if gpu is not None:
                snap.gpu_util_pct = round(random.uniform(*gpu["util"]), 2)
                snap.gpu_mem_used_mb = round(random.uniform(*gpu["mem"]), 1)
                snap.gpu_temp_c = round(random.uniform(*gpu["temp"]), 1)
            rows.append(snap)

        # 3) A couple of window stats so output distribution views have fleet data.
        for wi in range(3):
            w0 = now - timedelta(hours=hours) + timedelta(hours=hours * wi / 3)
            rows.append(TelemetryWindowStatsRow(
                user_id=model.user_id, model_id=model.id, deployment_id=dep.id,
                client_id=client_id, window_start=iso(w0),
                window_end=iso(w0 + timedelta(minutes=2)), n=per_host // 3,
                input_stats_json=json.dumps({"input": {
                    "mean": round(random.uniform(-0.1, 0.1), 4),
                    "std": round(random.uniform(0.9, 1.1), 4),
                    "min": -4.0, "max": 4.0, "nanPct": 0.0}}),
                output_json=json.dumps(_fleet_output()),
            ))

    if last_event_ts and (dep.last_event_at is None or last_event_ts > dep.last_event_at):
        dep.last_event_at = last_event_ts
        session.add(dep)
    session.add_all(rows)
    session.commit()
    return {
        "deploymentId": dep.id,
        "hosts": len(_FLEET),
        "events": per_host * len(_FLEET),
        "snapshots": snaps_per_host * len(_FLEET),
        "gpuHosts": sum(1 for p in _FLEET if p["gpu"] is not None),
        "devices": [p["runtime"].get("gpuName") or p["runtime"]["cpuModel"]
                    for p in _FLEET],
    }


def first_live_deployment(session: Session, model_id: str) -> DeploymentRow | None:
    rows = session.exec(
        select(DeploymentRow).where(DeploymentRow.model_id == model_id)
    ).all()
    rows.sort(key=lambda d: (d.status == "paused", d.created_at))
    return rows[0] if rows else None
