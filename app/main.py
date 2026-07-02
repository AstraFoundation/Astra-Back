"""Astra backend — FastAPI app factory.

All routes mount under `/api` to match the SPA's fetch('/api' + path). In
production the frontend is served from the same origin behind a reverse proxy so
the httpOnly session cookie is sent with every request.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.auth.dependencies import get_current_user
from app.config import get_settings
from app.db import get_engine, init_db
from app.errors import install_error_handlers
from app.middleware import RequestContextMiddleware, configure_logging
from app.seed import seed_if_empty
from app.services.limits import limiter

log = logging.getLogger("astra")


async def _inline_monitor_loop() -> None:
    """Single-box drift monitor (the closed loop's detection half). In scaled
    deploys the arq worker runs the monitor via its cron instead; here we tick
    it from the API process. Gated on monitor_inline_enabled (default ON; the
    test suite pins it off for determinism)."""
    from app.services.queue import run_monitor_once

    settings = get_settings()
    while True:
        await asyncio.sleep(max(5, settings.monitor_interval_sec))
        try:
            await asyncio.get_event_loop().run_in_executor(None, run_monitor_once)
        except Exception:  # noqa: BLE001 — a bad pass must not kill the loop
            log.exception("inline drift monitor pass failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings)
    for warning in settings.validate_runtime():  # raises on misconfig (fail fast)
        log.warning(warning)

    # Proactive feedback→GitHub reachability probe. The create-issue side effect
    # swallows failures (a GitHub outage must never drop feedback), so a wrong repo
    # or a revoked/insufficient token would otherwise fail silently forever — the
    # exact class of bug a renamed issues repo caused. Surface it loudly at boot.
    try:
        from app.services.github_feedback import check_feedback_github_reachable

        gh_warning = check_feedback_github_reachable()
        if gh_warning:
            log.warning(gh_warning)
        elif settings.github_feedback_enabled:
            log.info("Feedback→GitHub repo %s reachable", settings.feedback_github_repo)
    except Exception:  # noqa: BLE001 — a probe must never block startup
        log.exception("feedback→GitHub reachability probe failed")

    # Postgres schema is owned by Alembic; create_all is the SQLite/test path.
    if settings.is_sqlite:
        init_db()
    from sqlmodel import Session

    with Session(get_engine()) as session:
        seed_if_empty(session)
        # Self-heal the deploy badge invariant (is_deployed ⟹ status "deployed").
        # Wrapped so a failure can never block startup. See repositories for the
        # narrow, idempotent transition rules.
        try:
            from app.repositories import reconcile_deploy_status

            flipped = reconcile_deploy_status(session)
            log.info(
                "deploy-status reconcile: flipped %d model(s) to 'deployed': %s",
                len(flipped), flipped,
            )
        except Exception:  # noqa: BLE001 — reconcile must never block startup
            log.exception("deploy-status reconcile failed")

        # Fail any ingestion run left 'streaming' with no live worker (e.g. a
        # worker that died mid-pipeline) so models don't read 'analyzing' forever.
        try:
            from app.repositories import reconcile_orphaned_runs

            reaped = reconcile_orphaned_runs(session, settings.job_timeout_sec + 300)
            if reaped:
                log.warning(
                    "orphaned-run reaper: failed %d stuck run(s): %s", len(reaped), reaped
                )
        except Exception:  # noqa: BLE001 — reaper must never block startup
            log.exception("orphaned-run reconcile failed")
    log.info("Astra backend ready (db=%s storage=%s inline_jobs=%s)",
             "sqlite" if settings.is_sqlite else "postgres",
             settings.storage_backend, settings.inline_jobs)

    monitor_task: asyncio.Task | None = None
    if settings.monitor_inline_enabled:
        monitor_task = asyncio.create_task(_inline_monitor_loop())
    try:
        yield
    finally:
        if monitor_task is not None:
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor_task


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Astra Backend",
        description=(
            "Sensitivity-Guided Pareto Search + Surrogate Model + Real Inference "
            "Benchmarks — backend for the Astra on-device AI compression service"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    install_error_handlers(app)

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    from app.routers import (
        architecture,
        auth,
        client_telemetry,
        dashboard,
        deployments,
        feedback,
        ingestion,
        meta,
        models,
        pareto,
        sdk,
        telemetry,
    )

    api = APIRouter(prefix="/api")
    # Public — issues/clears the session cookie.
    api.include_router(auth.router)
    # Public — static capability matrix the upload/marketing UI reads pre-login.
    api.include_router(meta.router)
    # Public — the on-device SDK's telemetry ingestion + artifact pull. These
    # authenticate with a deployment API key (Authorization: Bearer …), NOT the
    # browser session cookie, so they sit outside the cookie gate. Astra never
    # runs user models server-side: inference happens on-device via the SDK's
    # AstraRunner; the server only serves the compressed artifact + collects
    # closed-loop telemetry.
    api.include_router(client_telemetry.router)
    # Every other router requires a valid session. Handlers that scope by owner
    # also inject CurrentUser; this router-level gate is defense-in-depth so a
    # newly added endpoint can never be unintentionally public.
    protected = [Depends(get_current_user)]
    api.include_router(dashboard.router, dependencies=protected)
    api.include_router(models.router, dependencies=protected)
    api.include_router(deployments.router, dependencies=protected)
    api.include_router(feedback.router, dependencies=protected)
    api.include_router(ingestion.router, dependencies=protected)
    api.include_router(architecture.router, dependencies=protected)
    api.include_router(pareto.router, dependencies=protected)
    api.include_router(telemetry.router, dependencies=protected)
    api.include_router(sdk.router, dependencies=protected)
    app.include_router(api)

    @app.get("/healthz")
    def healthz() -> dict:
        """Liveness — cheap, never touches dependencies."""
        return {"ok": True, "fastPipeline": settings.fast_pipeline}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        """Readiness — DB + Redis (when used) + object storage reachable."""
        checks: dict[str, str] = {}
        ok = True

        try:
            from sqlalchemy import text
            from sqlmodel import Session

            with Session(get_engine()) as s:
                s.exec(text("SELECT 1"))
            checks["db"] = "ok"
        except Exception:  # noqa: BLE001
            # Coarse status only — /readyz is publicly reachable, so never leak the
            # raw driver exception (internal hostnames, DB user/db, endpoints).
            log.exception("readyz: db check failed")
            checks["db"] = "error"
            ok = False

        if not settings.inline_jobs:
            try:
                import redis

                redis.Redis.from_url(settings.redis_url, socket_timeout=2).ping()
                checks["redis"] = "ok"
            except Exception:  # noqa: BLE001
                log.exception("readyz: redis check failed")
                checks["redis"] = "error"
                ok = False

        try:
            from app.services.storage import get_storage

            get_storage().ping()  # type: ignore[attr-defined]
            checks["storage"] = "ok"
        except Exception:  # noqa: BLE001
            log.exception("readyz: storage check failed")
            checks["storage"] = "error"
            ok = False

        if ok:
            return JSONResponse({"ok": True, "checks": checks})
        return JSONResponse(
            status_code=503,
            content={"detail": {"code": "not_ready", "message": "Dependencies unavailable.",
                                "checks": checks}},
        )

    return app


app = create_app()
