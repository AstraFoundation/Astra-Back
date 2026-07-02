"""Job dispatch — arq (Redis) in production, inline daemon threads otherwise.

The API enqueues via `enqueue_pipeline`. In `inline_jobs` mode (tests / single
box with no broker) it spawns a bounded daemon thread that runs
`execute_pipeline` in-process — preserving the live SSE/poll experience. In
scaled mode it pushes an arq job keyed by run_id; a separate worker process
(`worker.py`) consumes it and runs the same `execute_pipeline` in a thread
executor, with cancellation + timeout wired through the Redis-backed predicate.
"""

from __future__ import annotations

import asyncio
import threading
import time

from app.config import get_settings
from app.services.cancel import clear_cancel, make_should_cancel
from app.services.jobs import execute_pipeline

# ── inline (thread) dispatch ──────────────────────────────────────────────────

_inline_slots: threading.Semaphore | None = None
_inline_lock = threading.Lock()


def _slots() -> threading.Semaphore:
    global _inline_slots
    if _inline_slots is None:
        with _inline_lock:
            if _inline_slots is None:
                _inline_slots = threading.Semaphore(max(1, get_settings().job_workers))
    return _inline_slots


def _run_inline(payload: dict) -> None:
    settings = get_settings()
    run_id = payload["run_id"]
    deadline = time.time() + settings.job_timeout_sec
    should_cancel = make_should_cancel(run_id, deadline)

    def _target() -> None:
        with _slots():
            try:
                execute_pipeline(should_cancel=should_cancel, **payload)
            finally:
                clear_cancel(run_id)

    threading.Thread(target=_target, daemon=True, name=f"astra-inline-{run_id}").start()


# ── arq (Redis) dispatch ──────────────────────────────────────────────────────

_pool = None
_pool_lock = asyncio.Lock()


def _redis_settings():
    from arq.connections import RedisSettings

    return RedisSettings.from_dsn(get_settings().redis_url)


async def _get_pool():
    global _pool
    if _pool is None:
        async with _pool_lock:
            if _pool is None:
                from arq import create_pool

                _pool = await create_pool(_redis_settings())
    return _pool


async def enqueue_pipeline(
    *,
    run_id: str,
    model_id: str,
    user_id: str,
    model_name: str,
    file_name: str,
    source_key: str | None,
    input_shape: list[int] | None,
    declared_format: str,
) -> None:
    payload = {
        "run_id": run_id,
        "model_id": model_id,
        "user_id": user_id,
        "model_name": model_name,
        "file_name": file_name,
        "source_key": source_key,
        "input_shape": input_shape,
        "declared_format": declared_format,
    }
    if get_settings().inline_jobs:
        _run_inline(payload)
        return
    pool = await _get_pool()
    # _job_id=run_id makes enqueue idempotent and lets us reference the job later.
    await pool.enqueue_job("run_pipeline_task", _job_id=run_id, **payload)


# ── arq worker task + settings ────────────────────────────────────────────────

async def run_pipeline_task(ctx: dict, **payload) -> None:
    """arq entrypoint — runs the sync CPU pipeline in a thread so the worker's
    event loop stays responsive (heartbeats, concurrent jobs up to max_jobs)."""
    settings = get_settings()
    run_id = payload["run_id"]
    deadline = time.time() + settings.job_timeout_sec
    should_cancel = make_should_cancel(run_id, deadline)
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None, lambda: execute_pipeline(should_cancel=should_cancel, **payload),
        )
    finally:
        clear_cancel(run_id)


def run_monitor_once() -> dict:
    """One synchronous drift-monitor pass (shared by the arq cron + inline loop).
    Also runs the telemetry retention purge so the fact tables stay bounded."""
    from app.db import open_session
    from app.services.drift_monitor import drift_monitor_pass
    from app.services.retention import purge_expired

    with open_session() as session:
        result = drift_monitor_pass(session)
    # Separate session so a purge issue can't roll back the monitor's writes.
    with open_session() as session:
        purge_expired(session)
    return result


async def run_drift_monitor(ctx: dict) -> None:
    """arq cron entrypoint — refresh live deployment metrics + raise drift alerts."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_monitor_once)


async def _worker_startup(ctx: dict) -> None:
    # The worker is a fresh process: make sure the DB schema + storage exist.
    import logging

    from app.db import get_engine, init_db, open_session
    from app.services.storage import get_storage

    get_engine()
    if get_settings().is_sqlite:
        init_db()
    get_storage()

    # A restarted worker (e.g. after a native crash) abandoned whatever it was
    # running — arq does not resume in-flight jobs. Fail any ingestion run too old
    # to still be alive so its model doesn't read 'analyzing' forever.
    try:
        from app.repositories import reconcile_orphaned_runs

        with open_session() as session:
            reaped = reconcile_orphaned_runs(session, get_settings().job_timeout_sec + 300)
        if reaped:
            logging.getLogger("astra").warning(
                "worker-startup orphaned-run reaper: failed %d run(s): %s", len(reaped), reaped)
    except Exception:  # noqa: BLE001 — reaper must never block worker startup
        logging.getLogger("astra").exception("worker-startup reaper failed")


def _cron_jobs():
    # Default `second=0` → fires once per minute; refreshes live deployment
    # metrics and raises drift alerts from real inference_events.
    from arq import cron

    return [cron(run_drift_monitor, run_at_startup=True)]


class WorkerSettings:
    """`arq app.services.queue.WorkerSettings` — the worker process entrypoint."""

    functions = [run_pipeline_task]
    cron_jobs = _cron_jobs()
    on_startup = _worker_startup
    redis_settings = _redis_settings()
    max_jobs = get_settings().job_workers
    # arq's own timeout guards a hung coroutine; our deadline handles the sync
    # engine cooperatively, so set arq's a bit higher to avoid double-firing.
    job_timeout = get_settings().job_timeout_sec + 120
    keep_result = 0
    max_tries = 1  # pipelines are not idempotent to retry blindly
