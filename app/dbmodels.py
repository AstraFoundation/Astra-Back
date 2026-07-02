"""SQLModel tables (snake_case DB layer — API responses are mapped separately)."""

from __future__ import annotations

from sqlalchemy import Index, UniqueConstraint
from sqlmodel import Field, SQLModel


class UserRow(SQLModel, table=True):
    """A registered account. Owns models and all derived rows (multi-tenancy)."""

    __tablename__ = "users"

    id: str = Field(primary_key=True)
    email: str = Field(index=True, unique=True)   # stored lowercased
    password_hash: str | None = None              # None for OAuth-only accounts
    name: str
    created_at: str                               # ISO-8601
    role: str = "user"                            # user | admin (future)
    auth_provider: str = "password"               # password | google
    google_sub: str | None = Field(default=None, index=True)  # Google stable id
    # Session-revocation epoch. Embedded as a JWT claim; a token whose `tv` != this
    # is rejected. Bumped on password change and "log out everywhere" so a
    # leaked/stolen 7-day token can actually be evicted.
    token_version: int = 0


class ModelRow(SQLModel, table=True):
    __tablename__ = "models"

    id: str = Field(primary_key=True)
    user_id: str = Field(index=True, foreign_key="users.id")
    name: str
    type_full: str
    type_short: str
    format: str                       # frontend ModelFormat literal
    last_learned_at: str              # ISO-8601
    last_optimized_at: str | None = None
    status: str                       # frontend ModelStatus literal
    best_accuracy: float | None = None
    is_deployed: bool = False
    description: str | None = None
    analysis_run_id: str | None = None
    family: str = "cnn"               # han|lstm|diffusion-t|cnn|tree — filename hint metadata
    weights_only: bool = False        # raw state_dict checkpoint (no executable graph)
    source: str = "pipeline"          # pipeline (real uploads/imports only)
    # Object-storage keys (set by the worker / upload handler). None until present.
    artifact_key: str | None = None   # compressed artifact key
    source_key: str | None = None     # user-uploaded source key
    # Persisted artifact identity so the SDK-facing /artifacts endpoints serve the
    # ETag/size from metadata instead of reading the whole object + re-hashing it
    # on every call (incl. 304s). Set at pipeline completion; lazily backfilled.
    artifact_sha256: str | None = None
    artifact_size_bytes: int | None = None


class RunRow(SQLModel, table=True):
    """Dashboard optimization runs."""

    __tablename__ = "runs"

    id: str = Field(primary_key=True)
    user_id: str = Field(default="", index=True)
    model_id: str = Field(index=True)
    name: str
    status: str                       # running|queued|done|failed
    progress_pct: float = 0
    iter: str = "0 / 0"
    best_acc: float = 0
    delta_acc: float = 0
    created_at: str = ""


class IngestionRunRow(SQLModel, table=True):
    __tablename__ = "ingestion_runs"

    id: str = Field(primary_key=True)
    user_id: str = Field(default="", index=True)
    model_id: str = Field(index=True)
    file_name: str
    started_at: str
    status: str = "streaming"         # streaming|completed|failed
    progress: int = 0                 # 0..100
    error: str | None = None
    finished_at: str | None = None


class IngestionLogRow(SQLModel, table=True):
    __tablename__ = "ingestion_logs"

    pk: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    seq: int
    ts: str
    level: str                        # INFO|WARN|ERROR|DEBUG
    message: str


class ResultCacheRow(SQLModel, table=True):
    """Cached per-model JSON payloads (architecture / pareto / benchmark)."""

    __tablename__ = "result_cache"

    pk: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(default="", index=True)
    model_id: str = Field(index=True)
    kind: str                         # architecture | pareto | benchmark
    payload: str                      # JSON, already in frontend shape


class DeploymentRow(SQLModel, table=True):
    __tablename__ = "deployments"

    pk: int | None = Field(default=None, primary_key=True)
    # Stable business id (dep_<token>) — used in the inference URL and to link
    # api_keys / inference_events. `pk` stays the auto-increment surrogate.
    id: str = Field(default="", index=True)
    user_id: str = Field(default="", index=True)
    model_id: str = Field(default="", index=True)
    name: str = ""
    # User-editable note describing what this endpoint is for (which server,
    # which environment, …). Free text, empty when unset.
    description: str = ""
    endpoint: str
    region: str
    # Live metrics — maintained by the drift monitor from a rolling window of
    # inference_events. Zero until the first real traffic arrives.
    qps: float = 0.0
    p95: float = 0.0
    errors_pct: float = 0.0
    accuracy_drift: float = 0.0       # pts vs original (benchmark baseline)
    status: str                       # live|canary|paused
    created_at: str = ""
    last_event_at: str | None = None  # ISO-8601 of the most recent inference


class InferenceEventRow(SQLModel, table=True):
    """One on-device inference measured by the SDK — the raw telemetry fact.
    Astra never runs the model server-side; these arrive via the SDK's
    closed-loop telemetry batches (/api/v1/telemetry/{id}/batch)."""

    __tablename__ = "inference_events"
    __table_args__ = (
        Index("ix_infer_model_ts", "model_id", "ts"),
        Index("ix_infer_deployment_ts", "deployment_id", "ts"),
    )

    pk: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(default="", index=True)
    model_id: str = Field(index=True)
    deployment_id: str = Field(default="", index=True)
    ts: str = Field(index=True)               # ISO-8601 UTC, millisecond precision
    latency_ms: float
    success: bool = True
    error_code: str | None = None             # set on a failed inference
    batch_size: int = 1
    region: str = ""
    # On-device SDK telemetry always tags "client". The column is retained (with a
    # now-vestigial "server" value) for schema stability; there is no server-side
    # inference path anymore.
    source: str = Field(default="client", index=True)
    latency_pre_ms: float | None = None       # client-side input preparation
    latency_post_ms: float | None = None      # client-side output handling
    client_id: str | None = None              # stable random id per SDK process
    input_sig: str | None = None              # "input:1x3x224x224:float32" shape signature


class TelemetrySnapshotRow(SQLModel, table=True):
    """Periodic system snapshot from a astra-ai-sdk client serving a deployment
    locally — host resource usage + runtime fingerprint (every ~30s)."""

    __tablename__ = "telemetry_snapshots"
    __table_args__ = (
        Index("ix_snap_dep_ts", "deployment_id", "ts"),
        # The hardware/resource dashboards + _client_hardware filter by model_id
        # and order by ts; without this they scan the model's whole snapshot history.
        Index("ix_snap_model_ts", "model_id", "ts"),
    )

    pk: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(default="", index=True)
    model_id: str = Field(default="", index=True)
    deployment_id: str = Field(default="", index=True)
    client_id: str = Field(default="", index=True)
    ts: str = Field(index=True)
    cpu_pct: float = 0.0
    rss_mb: float = 0.0
    throughput_rpm: float = 0.0
    dropped_events: int = 0
    sdk_version: str = ""
    # {"python","ort","os","arch","provider","host", + hardware identity:
    #  "cpuModel","cpuCores","ramTotalMb","availableProviders","activeProvider",
    #  "gpuName","gpuCount","gpuMemTotalMb","cudaVersion","driverVersion"}
    runtime_json: str = "{}"
    # Dynamic accelerator sample (NULL when the host has no NVIDIA GPU) — kept as
    # real columns (not in runtime_json) so the resource time-series query is cheap.
    gpu_util_pct: float | None = None
    gpu_mem_used_mb: float | None = None
    gpu_temp_c: float | None = None


class TelemetryWindowStatsRow(SQLModel, table=True):
    """Windowed input/output distribution stats from a astra-ai-sdk client —
    the raw signal for prediction/input drift detection (every ~60s)."""

    __tablename__ = "telemetry_window_stats"
    __table_args__ = (
        Index("ix_winstats_dep_start", "deployment_id", "window_start"),
    )

    pk: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(default="", index=True)
    model_id: str = Field(default="", index=True)
    deployment_id: str = Field(default="", index=True)
    client_id: str = Field(default="", index=True)
    window_start: str = Field(index=True)
    window_end: str = ""
    n: int = 0
    input_stats_json: str = "{}"   # {input_name: {mean,std,min,max,nanPct}}
    output_json: str = "{}"        # {"classDist":{...},"hist":[...],"entropyMean":x,"top1ConfMean":x}


class ApiKeyRow(SQLModel, table=True):
    """A bearer key for a deployment's inference endpoint. Only the sha256 hash
    is stored; the plaintext is shown once at creation and never persisted."""

    __tablename__ = "api_keys"

    id: str = Field(primary_key=True)
    user_id: str = Field(default="", index=True)
    deployment_id: str = Field(default="", index=True)
    key_hash: str = Field(index=True)         # sha256 hex of the plaintext key
    prefix: str                               # display only: astra_sk_live_3i7c…b71c
    created_at: str
    last_used_at: str | None = None
    revoked: bool = False


class TelemetryRollupRow(SQLModel, table=True):
    """Per-deployment, per-minute pre-aggregate. The drift monitor upserts these
    so 7d/30d chart ranges read cheap rollups instead of scanning raw events."""

    __tablename__ = "telemetry_rollup"
    __table_args__ = (
        UniqueConstraint("deployment_id", "bucket_ts", name="uq_rollup_dep_bucket"),
    )

    pk: int | None = Field(default=None, primary_key=True)
    deployment_id: str = Field(default="", index=True)
    bucket_ts: str = Field(index=True)        # minute-truncated ISO-8601 UTC
    count: int = 0
    errors: int = 0
    sum_latency: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


class ProcessedBatchRow(SQLModel, table=True):
    """Idempotency ledger for the SDK's closed-loop telemetry. The SDK gives each
    durable-spool segment a stable `batchId`; recording it here (in the same
    transaction as the batch's rows) lets a re-sent batch — after an offline
    reconnect or a process restart — be recognised as a duplicate and acked
    without double-inserting, so dashboard KPIs never inflate."""

    __tablename__ = "processed_batches"

    batch_id: str = Field(primary_key=True)
    deployment_id: str = Field(default="", index=True)
    received_at: str = ""             # ISO-8601 UTC
    accepted_events: int = 0
    accepted_snapshots: int = 0
    accepted_windows: int = 0
    dropped: int = 0


class AlertRow(SQLModel, table=True):
    __tablename__ = "alerts"

    id: str = Field(primary_key=True)
    user_id: str = Field(default="", index=True)
    model_id: str = Field(default="", index=True)
    level: str                        # warning|danger
    title: str
    body: str
    at: str


class ActivityRow(SQLModel, table=True):
    __tablename__ = "activity_events"

    id: str = Field(primary_key=True)
    user_id: str = Field(default="", index=True)
    kind: str
    text: str
    timestamp: str


class FeedbackRow(SQLModel, table=True):
    """In-app feedback / feature-request submission. The durable record; a GitHub
    issue is opened from it as a best-effort side effect (github_issue_* set then)."""

    __tablename__ = "feedback"

    id: str = Field(primary_key=True)
    user_id: str = Field(default="", index=True)
    email: str = ""                    # submitter snapshot (from the session)
    name: str = ""
    kind: str = "feature"             # feature|bug|question|other
    message: str
    page: str | None = None           # route the user was on at submit time
    locale: str | None = None         # UI locale at submit time
    status: str = "open"              # open|triaged|closed (future)
    github_issue_number: int | None = None
    github_issue_url: str | None = None
    created_at: str = ""               # ISO-8601
    # Optional screenshot attachment. attachment_key is the object-storage key;
    # attachment_name preserves the original filename for the download/serve.
    attachment_key: str | None = None
    attachment_name: str | None = None


class RecipeRow(SQLModel, table=True):
    __tablename__ = "recipes"

    id: str = Field(primary_key=True)
    title: str
    description: str
    language: str
    steps_json: str = "[]"


class SdkSnippetRow(SQLModel, table=True):
    __tablename__ = "sdk_snippets"

    language: str = Field(primary_key=True)
    filename: str
    code: str
