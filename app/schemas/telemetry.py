"""Mirrors Astra-Front/src/features/telemetry/types.ts.

Note: `accuracyDrift` is `{value, note}` — unlike the other three KPI blocks
which are `{value, deltaPct}`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class DeltaKpi(BaseModel):
    value: float
    deltaPct: float


class DriftKpi(BaseModel):
    value: float
    note: str


class TelemetryKpi(BaseModel):
    requestsPerMin: DeltaKpi
    p95LatencyMs: DeltaKpi
    errorRate: DeltaKpi
    accuracyDrift: DriftKpi


class TelemetryPoint(BaseModel):
    t: str
    requests: float
    p95: float


class PercentileValues(BaseModel):
    p50: float
    p95: float
    p99: float


class Percentiles(BaseModel):
    p50: list[float]
    p95: list[float]
    p99: list[float]
    values: PercentileValues


class Deployment(BaseModel):
    id: str                 # deployment id (e.g. dep_ab12cd34ef) — what AstraRunner.from_deployment() takes
    region: str
    qps: float
    p95: float
    errorsPct: float
    status: Literal["live", "canary", "paused"]


class Alert(BaseModel):
    id: str
    level: Literal["warning", "danger"]
    title: str
    body: str
    at: str
