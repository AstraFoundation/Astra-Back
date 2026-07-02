"""Ingest astra-ai-sdk telemetry batches into the live-telemetry tables.

Client request events land in `inference_events` with source="client", so the
EXISTING aggregation (KPIs, series, percentiles, rollups, p95/error alerts)
consumes SDK traffic with zero changes. Snapshots and window stats get their
own tables; the drift monitor reads windows for prediction/input drift.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from app.config import get_settings, iso
from app.dbmodels import (
    DeploymentRow,
    InferenceEventRow,
    ModelRow,
    ProcessedBatchRow,
    TelemetrySnapshotRow,
    TelemetryWindowStatsRow,
)
from app.schemas.client_telemetry import TelemetryBatch

# The SDK buffers telemetry durably on disk while offline, so accept batches that
# were spooled for a while — but still reject wildly stale/future timestamps.
_PAST_CLAMP = timedelta(days=30)
_FUTURE_CLAMP = timedelta(minutes=5)


def _valid_ts(ts: str, now: datetime) -> bool:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return now - _PAST_CLAMP <= dt <= now + _FUTURE_CLAMP


def ingest_batch(
    session: Session,
    dep: DeploymentRow,
    model: ModelRow,
    batch: TelemetryBatch,
) -> tuple[dict[str, int], int]:
    """Bulk-insert one SDK batch. Returns (accepted counts, dropped count)."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    dropped = 0
    accepted = {"events": 0, "snapshots": 0, "windows": 0}

    total_items = len(batch.events) + len(batch.snapshots) + len(batch.windows)
    if total_items > settings.telemetry_batch_max:
        raise ValueError(
            f"batch exceeds {settings.telemetry_batch_max} items ({total_items})")

    # Idempotency: a durable-spool segment carries the same batchId across retries.
    # If we've already ingested this batch (reconnect or process restart re-sent
    # it), ack it WITHOUT re-inserting so the SDK safely deletes its segment and
    # the dashboard never double-counts.
    if batch.batchId:
        prior = session.get(ProcessedBatchRow, batch.batchId)
        if prior is not None:
            return ({
                "events": prior.accepted_events,
                "snapshots": prior.accepted_snapshots,
                "windows": prior.accepted_windows,
            }, prior.dropped)

    last_event_ts: str | None = None
    rows: list = []

    for ev in batch.events:
        if not _valid_ts(ev.ts, now):
            dropped += 1
            continue
        rows.append(InferenceEventRow(
            user_id=model.user_id,
            model_id=model.id,
            deployment_id=dep.id,
            ts=ev.ts,
            latency_ms=round(float(ev.latencyMs), 3),
            success=ev.success,
            error_code=ev.errorCode,
            batch_size=ev.batchSize,
            region=ev.region or "local",
            source="client",
            latency_pre_ms=round(float(ev.preMs), 3) if ev.preMs is not None else None,
            latency_post_ms=round(float(ev.postMs), 3) if ev.postMs is not None else None,
            client_id=batch.clientId,
            input_sig=ev.inputSig,
        ))
        accepted["events"] += 1
        if last_event_ts is None or ev.ts > last_event_ts:
            last_event_ts = ev.ts

    for snap in batch.snapshots:
        if not _valid_ts(snap.ts, now):
            dropped += 1
            continue
        rows.append(TelemetrySnapshotRow(
            user_id=model.user_id,
            model_id=model.id,
            deployment_id=dep.id,
            client_id=batch.clientId,
            ts=snap.ts,
            cpu_pct=round(float(snap.cpuPct), 2),
            rss_mb=round(float(snap.rssMb), 2),
            throughput_rpm=round(float(snap.throughputRpm), 2),
            dropped_events=snap.droppedEvents,
            sdk_version=snap.sdkVersion[:32],
            runtime_json=json.dumps({
                "python": snap.pythonVersion[:32],
                "ort": snap.ortVersion[:32],
                "os": snap.os[:32],
                "arch": snap.arch[:32],
                "provider": snap.provider[:64],
                "host": snap.host[:128],
                # Hardware identity (static).
                "cpuModel": snap.cpuModel[:96],
                "cpuCores": int(snap.cpuCores),
                "ramTotalMb": round(float(snap.ramTotalMb), 1),
                "availableProviders": snap.availableProviders[:256],
                "activeProvider": (snap.activeProvider or snap.provider)[:64],
                "gpuName": snap.gpuName[:96],
                "gpuCount": int(snap.gpuCount),
                "gpuMemTotalMb": round(float(snap.gpuMemTotalMb), 1),
                "cudaVersion": snap.cudaVersion[:32],
                "driverVersion": snap.driverVersion[:32],
            }),
            gpu_util_pct=(
                round(float(snap.gpuUtilPct), 2) if snap.gpuUtilPct is not None else None),
            gpu_mem_used_mb=(
                round(float(snap.gpuMemUsedMb), 1) if snap.gpuMemUsedMb is not None else None),
            gpu_temp_c=(
                round(float(snap.gpuTempC), 1) if snap.gpuTempC is not None else None),
        ))
        accepted["snapshots"] += 1

    for win in batch.windows:
        if not _valid_ts(win.windowStart, now):
            dropped += 1
            continue
        rows.append(TelemetryWindowStatsRow(
            user_id=model.user_id,
            model_id=model.id,
            deployment_id=dep.id,
            client_id=batch.clientId,
            window_start=win.windowStart,
            window_end=win.windowEnd,
            n=win.n,
            input_stats_json=json.dumps(
                {name: stat.model_dump() for name, stat in win.inputs.items()}),
            output_json=json.dumps(win.output),
        ))
        accepted["windows"] += 1

    session.add_all(rows)

    if last_event_ts and (dep.last_event_at is None or last_event_ts > dep.last_event_at):
        dep.last_event_at = last_event_ts
        session.add(dep)

    # Record the idempotency key in the same transaction as the rows, so a retry
    # of this exact batch is recognised as a duplicate.
    if batch.batchId:
        session.add(ProcessedBatchRow(
            batch_id=batch.batchId,
            deployment_id=dep.id,
            received_at=iso(now),
            accepted_events=accepted["events"],
            accepted_snapshots=accepted["snapshots"],
            accepted_windows=accepted["windows"],
            dropped=dropped,
        ))

    try:
        session.commit()
    except IntegrityError:
        # A concurrent request committed the same batchId first — treat as the
        # idempotent duplicate it is and return that batch's accepted counts.
        session.rollback()
        if batch.batchId:
            prior = session.get(ProcessedBatchRow, batch.batchId)
            if prior is not None:
                return ({
                    "events": prior.accepted_events,
                    "snapshots": prior.accepted_snapshots,
                    "windows": prior.accepted_windows,
                }, prior.dropped)
        raise
    return accepted, dropped


def source_counts(session: Session, model_id: str) -> dict[str, int]:
    """How many raw events each source contributed for a model (honest labeling
    of where 'live' data comes from)."""
    from sqlalchemy import func
    from sqlmodel import select

    rows = session.exec(
        select(InferenceEventRow.source, func.count())
        .where(InferenceEventRow.model_id == model_id)
        .group_by(InferenceEventRow.source)
    ).all()
    counts = {"server": 0, "client": 0}
    for source, n in rows:
        counts[source or "server"] = int(n)
    return counts
