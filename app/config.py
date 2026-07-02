"""Application settings — all configurable via ASTRA_* environment variables."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache
from urllib.parse import urlparse

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("astra")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASTRA_", env_file=".env", extra="ignore")

    # --- persistence ---
    db_path: str = "astra.db"
    # When set (e.g. postgresql://user:pass@host/db) this wins over the SQLite
    # db_path. Tests/local-dev leave it empty → SQLite.
    database_url: str | None = None
    storage_dir: str = "storage"

    # --- object storage ---
    storage_backend: str = "local"        # local | s3
    s3_endpoint_url: str | None = None     # set for MinIO; None → real AWS
    s3_bucket: str = "astra-artifacts"
    s3_region: str = "us-east-1"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_force_path_style: bool = True       # MinIO needs path-style addressing

    # --- job queue / worker ---
    redis_url: str = "redis://localhost:6379/0"
    # Run pipelines synchronously in-process instead of enqueuing to Redis.
    # Tests & single-box local dev set this so no broker/worker is needed.
    inline_jobs: bool = False
    work_dir: str = "/tmp/astra-work"      # worker scratch (engine stages here)

    # --- server ---
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # Public origin browsers use to reach the app (e.g. https://app.example.com).
    # Used to render correct base URLs in generated SDK snippets. When unset, the
    # base URL is derived per-request from the incoming Host (good enough for dev
    # and single-origin deploys); set it explicitly when behind a proxy/CDN.
    public_origin: str | None = None

    # --- auth ---
    jwt_secret: str = "dev-insecure-change-me"
    jwt_ttl_min: int = 60 * 24 * 7         # 7 days
    cookie_secure: bool = True             # set 0 for plain-HTTP local dev
    cookie_samesite: str = "lax"
    cookie_domain: str | None = None
    signup_enabled: bool = True

    # --- Google OAuth (optional sign-in method) ---
    google_client_id: str | None = None
    google_client_secret: str | None = None
    # Leave unset in production: the callback is then derived from public_origin
    # (see effective_google_redirect_uri) so it can never drift to a dev host.
    google_redirect_uri: str | None = None
    post_login_path: str = "/dashboard"    # where the SPA lands after OAuth

    # --- feedback widget (in-app → GitHub Issues) ---
    # When BOTH are set, each submitted feedback also opens a GitHub issue in the
    # target repo. Recommended: a PRIVATE repo + a fine-grained PAT scoped to
    # Issues: Read and write on that one repo. Unset → feedback is stored in the
    # DB only (no issue is created).
    feedback_github_token: str | None = None
    feedback_github_repo: str | None = None    # "owner/repo"
    # Image attachments on feedback submissions — screenshots, not model files,
    # so a much smaller cap and an image-only extension allowlist (separate from
    # the model-upload allowlist below).
    feedback_image_max_mb: int = 10
    feedback_image_exts: str = ".png,.jpg,.jpeg,.gif,.webp"

    # --- uploads / limits ---
    max_upload_mb: int = 2048
    allowed_upload_exts: str = (
        ".onnx,.pt,.pth,.bin,.ckpt,.safetensors,.h5,.keras,.pb,.tflite,"
        ".mlmodel,.pkl,.joblib,.gguf"
    )
    rate_limit_enabled: bool = True
    rate_limit_upload: str = "20/minute"
    rate_limit_import: str = "60/minute"
    rate_limit_auth: str = "30/minute"
    # Served inference is high-volume by design (a deployed model under load) —
    # generous so the traffic simulator and real bursts aren't throttled.
    rate_limit_infer: str = "1200/minute"

    # --- real inference serving + live telemetry ---
    inference_cache_size: int = 8         # warm ORT sessions kept in the LRU
    max_infer_batch: int = 64             # cap on synthesized batch size
    # Drift monitor: rolling window + thresholds (README closed-loop policy).
    monitor_interval_sec: int = 60        # how often the monitor pass runs
    monitor_window_min: int = 10          # rolling window for live deployment metrics
    drift_p95_pct: float = 10.0           # alert when p95 rises > +10% vs baseline
    drift_error_pct: float = 1.0          # alert when error rate exceeds 1%
    drift_acc_pts: float = 0.5            # alert when accuracy drifts > 0.5 pts
    # Demo affordance: when on, the dashboard "Generate traffic" button + the
    # /telemetry/simulate endpoint are enabled (off in production by default).
    telemetry_sim_enabled: bool = False
    # Run the drift monitor from inside the API process. Defaults ON so the
    # closed loop works out of the box on a single box (inline_jobs deploys
    # have no arq cron); scaled deploys with a worker may turn this off and
    # rely on the worker's cron instead.
    monitor_inline_enabled: bool = True
    # SSE dashboard refresh cadence. Each tick runs a DB round-trip per open tab;
    # 4s hammered the worker, so default to a calmer interval (still "live" enough).
    stream_interval_sec: int = 8
    # Client-telemetry (SDK) ingestion + drift thresholds on SDK-shipped stats.
    rate_limit_telemetry: str = "600/minute"
    rate_limit_artifact: str = "60/minute"  # artifact pulls (heavier; per-deployment)
    telemetry_batch_max: int = 500         # max items per /telemetry batch POST
    telemetry_body_max_mb: int = 8         # hard cap on the telemetry batch body (OOM guard)
    drift_psi: float = 0.2                 # prediction-drift PSI warning level
    drift_input_z: float = 3.0             # input-mean shift alert (z-score)

    # --- telemetry retention (unbounded fact tables → disk-fill outage) ---------
    # A periodic purge (run from the drift-monitor tick) deletes rows older than
    # these horizons so the primary Postgres volume can't grow without bound.
    retention_enabled: bool = True
    telemetry_retention_days: int = 90     # inference_events / snapshots / window_stats
    processed_batch_retention_days: int = 35  # idempotency ledger (> 30d spool horizon)
    activity_retention_days: int = 180     # activity_events / alerts

    # --- observability ---
    log_level: str = "INFO"
    log_json: bool = True

    # --- determinism ---
    seed: int = 42
    # Fixed reference instant for all generated timestamps/series. When unset,
    # the process start time is used (stable within one server process).
    ref_date: str | None = None

    # --- real compression pipeline (astra) ---
    fast_pipeline: bool = False           # tiny model + few trials (tests/CI)
    # The Pareto search optimizes two DETERMINISTIC objectives (accuracy ↑,
    # size ↓); latency is measured per trial for reporting but kept out of the
    # TPE objective (its wall-clock noise otherwise fed back into the sampler and
    # made results non-reproducible). With a deterministic objective the trial
    # budget scales with each model's real search dimensionality D instead of a
    # fixed guess: n = clamp(per_dim·D + startup, min, max). Empirically the
    # frontier converges by ~140 trials even at D~24 and <10 for trivial models,
    # so 150 stays the ceiling while simple models run far fewer trials.
    pareto_adaptive: bool = True          # ASTRA_PARETO_ADAPTIVE=0 → fixed budget
    pareto_trials: int = 150              # ceiling (also the fixed count if !adaptive)
    pareto_trials_per_dim: int = 10       # trials added per live search dimension
    pareto_startup_trials: int = 10       # constant budget term (TPE random startup)
    pareto_min_trials: int = 30           # lower clamp on the adaptive budget
    # HV-plateau early stop: OFF by default. The frontier shows long plateaus
    # then unpredictable late jumps, so a plateau stop cannot preserve quality —
    # the D-scaled budget is the reliable adaptive mechanism.
    pareto_early_stop: bool = False
    pareto_hv_patience: int = 20
    pareto_hv_epsilon: float = 1e-3
    n_probes: int = 16                    # synthetic calibration probes
    max_compressible_ops: int = 24        # UOSA builds one ORT session per op — cap it
    job_timeout_sec: int = 900
    job_workers: int = 2
    # Guarantee-by-construction backbone: the served artifact must pass the
    # pooled-probe OFS >= tau gate (the configuration validated in the paper
    # experiments). A Pareto candidate is served only when it passes the same
    # gate AND is smaller than the ladder's certified candidate.
    guarantee_mode: bool = True           # ASTRA_GUARANTEE_MODE=0 to disable
    tau: float = 0.95                     # fidelity floor for the gate
    # Per-trial Pareto export: refuse to materialize artifacts for source
    # models larger than this (transform memory is ~2x model size).
    trial_export_max_mb: int = 500

    # ── derived helpers ──────────────────────────────────────────────────────

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_upload_ext_set(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_upload_exts.split(",") if e.strip()}

    @property
    def feedback_image_ext_set(self) -> set[str]:
        return {e.strip().lower() for e in self.feedback_image_exts.split(",") if e.strip()}

    @property
    def effective_database_url(self) -> str:
        """The SQLAlchemy URL actually used. database_url wins; else SQLite file."""
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.db_path}"

    @property
    def is_sqlite(self) -> bool:
        return self.effective_database_url.startswith("sqlite")

    @property
    def google_enabled(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)

    @property
    def github_feedback_enabled(self) -> bool:
        """True when feedback submissions should also open a GitHub issue."""
        return bool(self.feedback_github_token and self.feedback_github_repo)

    @property
    def effective_google_redirect_uri(self) -> str:
        """OAuth callback URL. Prefers an explicit ASTRA_GOOGLE_REDIRECT_URI;
        otherwise derives it from public_origin so a production deploy only has
        to set ASTRA_PUBLIC_ORIGIN. Falls back to the localhost dev default."""
        if self.google_redirect_uri:
            return self.google_redirect_uri
        if self.public_origin:
            return f"{self.public_origin.rstrip('/')}/api/auth/google/callback"
        return "http://localhost:5173/api/auth/google/callback"

    def validate_runtime(self) -> list[str]:
        """Fail-fast checks at startup. Returns a list of warnings (non-fatal);
        raises ValueError on misconfiguration that would break the app."""
        warnings: list[str] = []
        # A JWT secret is unsafe if it's a known placeholder or too short to
        # resist offline brute force. The guard used to only match the code
        # default and only warn — so the compose fallback ("change-me-in-production")
        # and the .env.example placeholder both slipped through and booted with a
        # repo-public HS256 key. Now: warn in dev (SQLite), FAIL CLOSED in a
        # production-like deploy (Postgres) so a missing/placeholder secret aborts
        # startup instead of silently signing forgeable session cookies.
        _KNOWN_WEAK = {
            "dev-insecure-change-me",
            "change-me-in-production",
            "change-me-please-openssl-rand-hex-32",
        }
        jwt_weak = self.jwt_secret in _KNOWN_WEAK or len(self.jwt_secret) < 32
        is_prod_like = not self.is_sqlite  # prod hard-wires Postgres; dev/test = SQLite
        if jwt_weak and is_prod_like:
            raise ValueError(
                "ASTRA_JWT_SECRET is a placeholder or too short (<32 chars) in a "
                "production deploy — refusing to boot with a forgeable session key. "
                "Set a strong secret: openssl rand -hex 32"
            )
        if jwt_weak:
            warnings.append(
                "ASTRA_JWT_SECRET is an insecure placeholder/short value — set a "
                "strong secret (openssl rand -hex 32) before deploying."
            )
        if bool(self.google_client_id) != bool(self.google_client_secret):
            warnings.append(
                "Google OAuth is half-configured — set BOTH ASTRA_GOOGLE_CLIENT_ID and "
                "ASTRA_GOOGLE_CLIENT_SECRET (or neither). Google sign-in stays disabled."
            )
        if bool(self.feedback_github_token) != bool(self.feedback_github_repo):
            warnings.append(
                "Feedback→GitHub is half-configured — set BOTH ASTRA_FEEDBACK_GITHUB_TOKEN "
                "and ASTRA_FEEDBACK_GITHUB_REPO (or neither). Feedback is still stored in "
                "the DB; no issue is opened."
            )
        # An HTTPS public origin must serve a Secure session cookie; otherwise the
        # cookie lacks the Secure attribute and could leak over a downgraded request.
        if (self.public_origin and self.public_origin.startswith("https://")
                and not self.cookie_secure):
            warnings.append(
                "ASTRA_PUBLIC_ORIGIN is HTTPS but ASTRA_COOKIE_SECURE=0 — the session "
                "cookie will lack the Secure attribute. Set ASTRA_COOKIE_SECURE=1 in production."
            )
        # The OAuth callback must live on the public origin, or Google redirects
        # users to an unreachable host (e.g. a leftover localhost dev value).
        if self.google_enabled and self.public_origin:
            want = urlparse(self.public_origin).netloc
            got = urlparse(self.effective_google_redirect_uri).netloc
            if want and got and want != got:
                warnings.append(
                    f"ASTRA_GOOGLE_REDIRECT_URI host ({got}) does not match "
                    f"ASTRA_PUBLIC_ORIGIN host ({want}) — Google sign-in would redirect "
                    "users to the wrong host. Unset ASTRA_GOOGLE_REDIRECT_URI to derive it "
                    "from ASTRA_PUBLIC_ORIGIN, or set it to the public callback URL."
                )
        if self.storage_backend == "s3":
            missing = [
                name for name, val in (
                    ("ASTRA_S3_BUCKET", self.s3_bucket),
                    ("ASTRA_S3_ACCESS_KEY", self.s3_access_key),
                    ("ASTRA_S3_SECRET_KEY", self.s3_secret_key),
                ) if not val
            ]
            if missing:
                raise ValueError(
                    f"storage_backend=s3 requires {', '.join(missing)}"
                )
        elif self.storage_backend != "local":
            raise ValueError(f"unknown storage_backend: {self.storage_backend!r}")
        if not self.is_sqlite and self.inline_jobs:
            warnings.append(
                "inline_jobs=1 with a non-SQLite DB runs pipelines inside the API "
                "process — fine for a single box, but use a worker for production scale."
            )
        return warnings


_PROCESS_START = datetime.now(timezone.utc).replace(microsecond=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def ref_now() -> datetime:
    """The reference 'now' for every generated timestamp.

    Defaults to process start so repeated requests are byte-identical within a
    server lifetime; pin ASTRA_REF_DATE for cross-process determinism.
    """
    s = get_settings()
    if s.ref_date:
        dt = datetime.fromisoformat(s.ref_date.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return _PROCESS_START


def iso(dt: datetime) -> str:
    """ISO-8601 UTC with Z suffix and millisecond precision (JS-compatible)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
