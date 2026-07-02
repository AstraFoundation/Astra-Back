"""Telemetry data retention — bound the append-only fact tables so the primary
Postgres volume can't grow without limit (a full disk stops ingestion, dashboards
and every shared write). Runs from the drift-monitor tick; each DELETE is a cheap
no-op once the backlog is trimmed because every timestamp column is indexed.

Timestamps are ISO-8601 UTC ("…Z") strings, so a lexical `< cutoff` comparison is
a correct chronological filter.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete
from sqlmodel import Session

from app.config import get_settings, iso
from app.dbmodels import (
    ActivityRow,
    AlertRow,
    InferenceEventRow,
    ProcessedBatchRow,
    TelemetrySnapshotRow,
    TelemetryWindowStatsRow,
)

log = logging.getLogger("astra")


def purge_expired(session: Session) -> dict[str, int]:
    """Delete rows older than their configured horizon. Returns {table: deleted}.
    Never raises into the caller (the monitor loop) — a purge failure is logged."""
    settings = get_settings()
    if not settings.retention_enabled:
        return {}

    now = datetime.now(timezone.utc)
    tel_cut = iso(now - timedelta(days=settings.telemetry_retention_days))
    pb_cut = iso(now - timedelta(days=settings.processed_batch_retention_days))
    act_cut = iso(now - timedelta(days=settings.activity_retention_days))

    # (model, timestamp column, cutoff)
    targets = [
        (InferenceEventRow, InferenceEventRow.ts, tel_cut),
        (TelemetrySnapshotRow, TelemetrySnapshotRow.ts, tel_cut),
        (TelemetryWindowStatsRow, TelemetryWindowStatsRow.window_start, tel_cut),
        (ProcessedBatchRow, ProcessedBatchRow.received_at, pb_cut),
        (AlertRow, AlertRow.at, act_cut),
        (ActivityRow, ActivityRow.timestamp, act_cut),
    ]
    deleted: dict[str, int] = {}
    try:
        for model, col, cutoff in targets:
            res = session.execute(delete(model).where(col < cutoff))
            deleted[model.__tablename__] = int(res.rowcount or 0)
        session.commit()
    except Exception:  # noqa: BLE001 — retention must never break the monitor
        session.rollback()
        log.exception("telemetry retention purge failed")
        return {}

    total = sum(deleted.values())
    if total:
        log.info("telemetry retention purged %d rows: %s", total, deleted)
    return deleted
