"""Cost & savings lens schemas.

Mirrors Astra-Front/src/features/telemetry/types.ts (ModelCostSummary) and
src/features/dashboard/types.ts (WorkspaceCostSavings). Nullable fields are
honest absences — a monthly $ is null until real traffic is measured, an
original cost is null when no benchmark exists to derive the counterfactual.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HardwareCostRow(BaseModel):
    """One accelerator class's $/1M for the compressed artifact, with the
    counterfactual original cost (compressed × benchmarked latency ratio)."""

    key: str
    label: str
    accelerator: Literal["gpu", "coreml", "npu", "cpu", "hosted"]
    p95: float
    throughputPerSec: float
    compressedPer1M: float
    originalPer1M: float | None = None
    savingsPer1M: float | None = None


class ModelCostSummary(BaseModel):
    # "live" = derived from measured serving traffic; "none" = no real traffic yet,
    # so the cost lens is empty (no benchmark-derived numbers are surfaced).
    source: Literal["live", "none"]
    compressedPer1M: float
    originalPer1M: float | None = None
    savingsPer1M: float | None = None
    savingsPct: float | None = None
    # original.p95 / compressed.p95 — the counterfactual basis disclosed in the UI.
    assumedLatencyRatio: float | None = None
    measuredQps: float = 0.0
    # Monthly $ — asserted ONLY when measuredQps > 0 (real traffic). Null otherwise.
    monthlyCompressed: float | None = None
    monthlyOriginal: float | None = None
    monthlySavings: float | None = None
    # Projection at a caller-supplied target QPS (labeled a projection in the UI).
    projected: bool = False
    projectedMonthlyCompressed: float | None = None
    projectedMonthlyOriginal: float | None = None
    projectedMonthlySavings: float | None = None
    perHardware: list[HardwareCostRow] = []


class WorkspaceCostSavings(BaseModel):
    """Workspace-wide $ rollup. Both `monthly*` and `avgSavingsPct` are asserted
    only across models with real serving traffic (null before any are used)."""

    hasLiveTraffic: bool
    monthlyCompressed: float | None = None
    monthlyOriginal: float | None = None
    monthlySavings: float | None = None
    avgSavingsPct: float | None = None
    modelCount: int
    liveModelCount: int
