"""Ingestion run status + real log streaming (SSE) with a polling fallback.

The SPA currently simulates the log stream client-side; these endpoints are the
server-side replacement, emitting the same `IngestionLog {ts, level, message}`
shape so `startIngestionStream` can be swapped for an EventSource."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse

from app.auth.dependencies import CurrentUser, current_user_id_for_stream
from app.db import get_session, open_session
from app.dbmodels import IngestionLogRow, IngestionRunRow
from app.schemas.models import IngestionLog, IngestionRun

router = APIRouter(prefix="/models/{model_id}/ingestion", tags=["ingestion"])


def _get_run(session: Session, model_id: str, run_id: str, user_id: str) -> IngestionRunRow:
    run = session.get(IngestionRunRow, run_id)
    if run is None or run.model_id != model_id or run.user_id != user_id:
        raise HTTPException(status_code=404, detail="ingestion run not found")
    return run


@router.get("/{run_id}")
def run_status(
    model_id: str,
    run_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> dict:
    run = _get_run(session, model_id, run_id, current_user.id)
    payload = IngestionRun(
        id=run.id, modelId=run.model_id, fileName=run.file_name,
        startedAt=run.started_at, status=run.status,  # type: ignore[arg-type]
    ).model_dump()
    payload["progress"] = run.progress
    if run.error:
        payload["error"] = run.error
    return payload


@router.get("/{run_id}/logs")
def run_logs(
    model_id: str,
    run_id: str,
    current_user: CurrentUser,
    after: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
) -> dict:
    run = _get_run(session, model_id, run_id, current_user.id)
    rows = session.exec(
        select(IngestionLogRow)
        .where(IngestionLogRow.run_id == run_id, IngestionLogRow.seq > after)
        .order_by(IngestionLogRow.seq)  # type: ignore[arg-type]
    ).all()
    return {
        "logs": [
            {"seq": r.seq, **IngestionLog(ts=r.ts, level=r.level, message=r.message).model_dump()}  # type: ignore[arg-type]
            for r in rows
        ],
        "done": run.status != "streaming",
        "status": run.status,
        "progress": run.progress,
    }


@router.get("/{run_id}/stream")
async def run_stream(
    model_id: str,
    run_id: str,
    request: Request,
    user_id: str = Depends(current_user_id_for_stream),
) -> EventSourceResponse:
    with open_session() as s:
        _get_run(s, model_id, run_id, user_id)

    async def event_source():
        last_seq = 0
        # -1 forces an immediate emit on connect, so a mid-run refresh (or a
        # completed-run revisit) gets the current percent before any `done`.
        last_progress = -1
        while True:
            if await request.is_disconnected():
                return
            with open_session() as s:
                rows = s.exec(
                    select(IngestionLogRow)
                    .where(IngestionLogRow.run_id == run_id, IngestionLogRow.seq > last_seq)
                    .order_by(IngestionLogRow.seq)  # type: ignore[arg-type]
                ).all()
                run = s.get(IngestionRunRow, run_id)
            for r in rows:
                last_seq = r.seq
                yield {
                    "event": "log",
                    "data": json.dumps(
                        IngestionLog(ts=r.ts, level=r.level, message=r.message).model_dump()  # type: ignore[arg-type]
                    ),
                }
            if run is not None and run.progress != last_progress:
                last_progress = run.progress
                yield {"event": "progress", "data": json.dumps({"progress": run.progress})}
            if run is not None and run.status != "streaming":
                yield {"event": "done", "data": json.dumps({"status": run.status})}
                return
            await asyncio.sleep(0.3)

    return EventSourceResponse(event_source())
