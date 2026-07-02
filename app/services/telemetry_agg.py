"""Telemetry aggregation — turn raw inference_events into the dashboard contract.

The dashboard shows ONLY real deployment traffic:
  • LIVE  — the model has real inference events → aggregate them (KPIs carry a
            real window-over-window delta; series/percentiles are time-bucketed).
  • EMPTY — no real traffic yet → the endpoints return empty/zero shapes (NOT the
            post-compression benchmark). The SPA gates on /meta and renders its
            "deploy / waiting for traffic" empty states instead of fabricated
            numbers. `has_any_events` is the single pivot.

≤24h ranges read raw events (exact). 7d/30d read the per-minute TelemetryRollupRow
the drift monitor maintains, downsampled to the range's buckets (cheap at scale),
falling back to raw when no rollups exist yet.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from app.config import iso
from app.dbmodels import DeploymentRow, InferenceEventRow, TelemetryRollupRow
from app.schemas.telemetry import (
    DeltaKpi,
    DriftKpi,
    Percentiles,
    PercentileValues,
    TelemetryKpi,
    TelemetryPoint,
)

# range → (window duration, bucket count)
_RANGE: dict[str, tuple[timedelta, int]] = {
    "1h": (timedelta(hours=1), 60),    # 1-min buckets
    "6h": (timedelta(hours=6), 72),    # 5-min buckets
    "24h": (timedelta(hours=24), 48),  # 30-min buckets
    "7d": (timedelta(days=7), 84),     # 2-hr buckets
    "30d": (timedelta(days=30), 60),   # 12-hr buckets
}

_DRIFT_NOTE = "output divergence vs original (DFCV, measured)"


def _window(range_str: str) -> tuple[timedelta, int]:
    return _RANGE.get(range_str, _RANGE["24h"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _pct(values: list[float], q: float) -> float:
    import numpy as np

    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def _delta_pct(prev: float, cur: float) -> float:
    if prev <= 0:
        return 0.0
    return round((cur - prev) / prev * 100, 1)


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


# ── source selection ─────────────────────────────────────────────────────────


def has_any_events(session: Session, model_id: str) -> bool:
    """True once a model has served at least one real inference (→ live path)."""
    return session.exec(
        select(InferenceEventRow.pk).where(InferenceEventRow.model_id == model_id).limit(1)
    ).first() is not None


# Memory guard: cap how many raw events one aggregation pulls into Python. Above
# this the newest N are used (a bounded sample) instead of scanning a whole M-row
# window into RAM per dashboard load. Normal windows are far below the cap, so this
# never changes their results; the proper long-term fix is SQL-side aggregation.
_MAX_EVENTS = 200_000


def _fetch_events(
    session: Session, model_id: str, start: datetime, end: datetime,
) -> list[tuple[datetime, float, bool]]:
    # Select only the 3 columns used (not whole rows), newest-first with a cap.
    rows = session.exec(
        select(InferenceEventRow.ts, InferenceEventRow.latency_ms, InferenceEventRow.success)
        .where(
            InferenceEventRow.model_id == model_id,
            InferenceEventRow.ts >= iso(start),
            InferenceEventRow.ts < iso(end),
        )
        .order_by(InferenceEventRow.ts.desc())  # type: ignore[attr-defined]
        .limit(_MAX_EVENTS)
    ).all()
    out: list[tuple[datetime, float, bool]] = []
    for ts, latency_ms, success in rows:
        try:
            out.append((_parse(ts), latency_ms, success))
        except ValueError:
            continue
    out.sort(key=lambda e: e[0])
    return out


def _deployment_ids(session: Session, model_id: str) -> list[str]:
    return list(session.exec(
        select(DeploymentRow.id).where(DeploymentRow.model_id == model_id)
    ).all())


# ── LIVE: KPI ────────────────────────────────────────────────────────────────


def kpi_live(
    session: Session, model_id: str, bench: dict | None, range_str: str = "24h",
) -> TelemetryKpi:
    window, _ = _window(range_str)
    now = _now()
    cur = _fetch_events(session, model_id, now - window, now)
    prev = _fetch_events(session, model_id, now - 2 * window, now - window)
    win_min = window.total_seconds() / 60.0

    def rpm(evts: list) -> float:
        return len(evts) / win_min if win_min > 0 else 0.0

    def p95(evts: list) -> float:
        return _pct([lat for _, lat, ok in evts if ok], 95)

    def err(evts: list) -> float:
        return 100.0 * sum(1 for _, _, ok in evts if not ok) / len(evts) if evts else 0.0

    cur_rpm, prev_rpm = rpm(cur), rpm(prev)
    cur_p95, prev_p95 = p95(cur), p95(prev)
    cur_err, prev_err = err(cur), err(prev)
    divergence = round(100.0 - bench["agreementPct"], 2) if bench else 0.0
    return TelemetryKpi(
        requestsPerMin=DeltaKpi(value=round(cur_rpm, 1), deltaPct=_delta_pct(prev_rpm, cur_rpm)),
        p95LatencyMs=DeltaKpi(value=round(cur_p95, 3), deltaPct=_delta_pct(prev_p95, cur_p95)),
        errorRate=DeltaKpi(value=round(cur_err, 3), deltaPct=_delta_pct(prev_err, cur_err)),
        accuracyDrift=DriftKpi(value=divergence, note=_DRIFT_NOTE),
    )


# ── LIVE: series / percentiles ───────────────────────────────────────────────


def _bucket_bounds(range_str: str) -> tuple[datetime, timedelta, int]:
    window, n = _window(range_str)
    now = _now()
    start = now - window
    return start, window / n, n


def _bucket_raw(
    session: Session, model_id: str, range_str: str,
) -> list[tuple[datetime, list[float], int]]:
    """Returns per-bucket (start, successful-latencies, total-count)."""
    start, size, n = _bucket_bounds(range_str)
    events = _fetch_events(session, model_id, start, start + size * n)
    buckets: list[tuple[datetime, list[float], list[int]]] = [
        (start + size * i, [], [0]) for i in range(n)
    ]
    size_sec = size.total_seconds()
    for dt, lat, ok in events:
        idx = int((dt - start).total_seconds() / size_sec) if size_sec > 0 else 0
        if 0 <= idx < n:
            _, lats, cnt = buckets[idx]
            cnt[0] += 1
            if ok:
                lats.append(lat)
    return [(b0, lats, cnt[0]) for b0, lats, cnt in buckets]


def _use_rollups(range_str: str) -> bool:
    return range_str in ("7d", "30d")


def _bucket_rollups(
    session: Session, model_id: str, range_str: str,
) -> list[tuple[datetime, dict, int]] | None:
    """Per-bucket aggregate from per-minute rollups; None when none exist yet."""
    dep_ids = _deployment_ids(session, model_id)
    if not dep_ids:
        return None
    start, size, n = _bucket_bounds(range_str)
    rows = session.exec(
        select(TelemetryRollupRow).where(
            TelemetryRollupRow.deployment_id.in_(dep_ids),  # type: ignore[attr-defined]
            TelemetryRollupRow.bucket_ts >= iso(start),
            TelemetryRollupRow.bucket_ts < iso(start + size * n),
        )
    ).all()
    if not rows:
        return None
    size_sec = size.total_seconds()
    agg: list[dict] = [
        {"count": 0, "errors": 0, "wp50": 0.0, "wp95": 0.0, "wp99": 0.0}
        for _ in range(n)
    ]
    for r in rows:
        try:
            dt = _parse(r.bucket_ts)
        except ValueError:
            continue
        idx = int((dt - start).total_seconds() / size_sec) if size_sec > 0 else 0
        if not (0 <= idx < n):
            continue
        a = agg[idx]
        a["count"] += r.count
        a["errors"] += r.errors
        # count-weighted percentile downsample (exact for one deployment/minute).
        a["wp50"] += r.p50 * r.count
        a["wp95"] += r.p95 * r.count
        a["wp99"] += r.p99 * r.count
    out: list[tuple[datetime, dict, int]] = []
    for i, a in enumerate(agg):
        c = a["count"] or 1
        out.append((start + size * i, {
            "p50": a["wp50"] / c, "p95": a["wp95"] / c, "p99": a["wp99"] / c,
        }, a["count"]))
    return out


def series_live(session: Session, model_id: str, range_str: str = "24h") -> list[TelemetryPoint]:
    _, size, _ = _bucket_bounds(range_str)
    size_min = size.total_seconds() / 60.0
    if _use_rollups(range_str):
        roll = _bucket_rollups(session, model_id, range_str)
        if roll is not None:
            return [
                TelemetryPoint(
                    t=iso(b0),
                    requests=round(cnt / size_min, 1) if size_min > 0 else 0.0,
                    p95=round(pct["p95"], 3),
                )
                for b0, pct, cnt in roll
            ]
    return [
        TelemetryPoint(
            t=iso(b0),
            requests=round(cnt / size_min, 1) if size_min > 0 else 0.0,
            p95=round(_pct(lats, 95), 3),
        )
        for b0, lats, cnt in _bucket_raw(session, model_id, range_str)
    ]


def percentiles_live(session: Session, model_id: str, range_str: str = "24h") -> Percentiles:
    window, _ = _window(range_str)
    now = _now()
    overall = [lat for _, lat, ok in _fetch_events(session, model_id, now - window, now) if ok]
    values = PercentileValues(
        p50=round(_pct(overall, 50), 3),
        p95=round(_pct(overall, 95), 3),
        p99=round(_pct(overall, 99), 3),
    )
    if _use_rollups(range_str):
        roll = _bucket_rollups(session, model_id, range_str)
        if roll is not None:
            return Percentiles(
                p50=[round(p["p50"], 3) for _, p, _ in roll],
                p95=[round(p["p95"], 3) for _, p, _ in roll],
                p99=[round(p["p99"], 3) for _, p, _ in roll],
                values=values,
            )
    raw = _bucket_raw(session, model_id, range_str)
    return Percentiles(
        p50=[round(_pct(lats, 50), 3) for _, lats, _ in raw],
        p95=[round(_pct(lats, 95), 3) for _, lats, _ in raw],
        p99=[round(_pct(lats, 99), 3) for _, lats, _ in raw],
        values=values,
    )


# ── EMPTY (no real traffic yet) ──────────────────────────────────────────────


def kpi_empty() -> TelemetryKpi:
    """Zeroed KPIs for a deployed-but-untrafficked model. The SPA renders "—"
    (driven by /meta source != "live"), so these values are never shown."""
    return TelemetryKpi(
        requestsPerMin=DeltaKpi(value=0.0, deltaPct=0.0),
        p95LatencyMs=DeltaKpi(value=0.0, deltaPct=0.0),
        errorRate=DeltaKpi(value=0.0, deltaPct=0.0),
        accuracyDrift=DriftKpi(value=0.0, note=_DRIFT_NOTE),
    )


def percentiles_empty() -> Percentiles:
    return Percentiles(
        p50=[], p95=[], p99=[],
        values=PercentileValues(p50=0.0, p95=0.0, p99=0.0),
    )
