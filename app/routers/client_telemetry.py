"""Public API-key-authed endpoints for the astra-ai-sdk package (on-device SDK).

  POST /api/v1/telemetry/{deployment_id}/batch   — ship closed-loop telemetry
  GET  /api/v1/artifacts/{deployment_id}         — pull the deployed artifact
  GET  /api/v1/artifacts/{deployment_id}/info    — artifact metadata (ETag basis)

These sit OUTSIDE the session-cookie gate: callers are on-device SDK processes
holding a deployment API key, not browsers. Astra serves the compressed artifact
for the SDK to run locally and ingests its telemetry — it never runs the model
server-side.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from slowapi.util import get_remote_address
from sqlmodel import Session, select

from app.config import get_settings
from app.db import get_session
from app.dbmodels import ApiKeyRow, DeploymentRow, ModelRow
from app.schemas.client_telemetry import BatchAccepted, TelemetryBatch
from app.services import apikeys
from app.services.client_telemetry import ingest_batch
from app.services.limits import limiter
from app.services.storage import StorageError, get_storage

router = APIRouter(prefix="/v1", tags=["client-telemetry"])


def _deployment_key(request: Request) -> str:
    """Rate-limit key for the API-key-authed SDK endpoints: bucket by the target
    deployment (path param), NOT the client IP. IP keying is both spoofable
    (X-Forwarded-For) and wrong here — a whole NAT'd fleet shares one IP, and with
    trusted-proxy restrictions every client collapses to the frontend's IP. Keying
    by deployment_id gives each deployment its own honest quota."""
    return request.path_params.get("deployment_id") or get_remote_address(request)


def _bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return authorization.strip()


def _resolve(
    deployment_id: str, authorization: str | None, session: Session,
) -> tuple[ApiKeyRow, DeploymentRow, ModelRow]:
    """Authenticate a deployment API key (Authorization: Bearer astra_sk_…) and
    resolve its deployment + model. Used by the on-device SDK's telemetry and
    artifact-pull endpoints — the only remaining API-key-authed surface."""
    key = apikeys.resolve_key(session, _bearer(authorization))
    if key is None:
        raise HTTPException(status_code=401, detail={
            "code": "invalid_api_key", "message": "Missing or invalid API key.",
        })
    # 404 (not 403) on mismatch — never confirm a deployment the key can't reach.
    dep = session.exec(
        select(DeploymentRow).where(DeploymentRow.id == deployment_id)
    ).first()
    if dep is None or key.deployment_id != deployment_id:
        raise HTTPException(status_code=404, detail={
            "code": "deployment_not_found", "message": "Deployment not found.",
        })
    if dep.status == "paused":
        raise HTTPException(status_code=409, detail={
            "code": "deployment_paused", "message": "This deployment is paused.",
        })
    model = session.get(ModelRow, dep.model_id)
    if model is None or not model.artifact_key:
        raise HTTPException(status_code=404, detail={
            "code": "no_artifact", "message": "Deployment has no servable artifact.",
        })
    return key, dep, model


@router.post("/telemetry/{deployment_id}/batch")
@limiter.limit(get_settings().rate_limit_telemetry, key_func=_deployment_key)
def telemetry_batch(
    deployment_id: str,
    body: TelemetryBatch,
    request: Request,
    authorization: str | None = Header(default=None),
    session: Session = Depends(get_session),
) -> BatchAccepted:
    key, dep, model = _resolve(deployment_id, authorization, session)
    apikeys.touch_key(session, key)
    try:
        accepted, dropped = ingest_batch(session, dep, model, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={
            "code": "batch_too_large", "message": str(exc),
        }) from exc
    return BatchAccepted(accepted=accepted, dropped=dropped)


def _artifact_meta(session: Session, model) -> tuple[str, int, str]:
    """(sha256, size_bytes, key) for the deployed artifact, from PERSISTED metadata.

    Avoids reading the whole object + re-hashing on every /info, ETag and 304 —
    which spiked the shared worker to N×artifact-size RAM under a re-pulling fleet.
    Artifacts deployed before the metadata columns existed are backfilled once
    (a single read), then served from the row forever after."""
    key = model.artifact_key
    if not key:
        raise HTTPException(status_code=404, detail={
            "code": "no_artifact", "message": "Deployment has no servable artifact.",
        })
    if model.artifact_sha256 and model.artifact_size_bytes is not None:
        return model.artifact_sha256, model.artifact_size_bytes, key
    # One-time lazy backfill for pre-existing artifacts.
    try:
        data = get_storage().read_bytes(key)
    except StorageError:
        raise HTTPException(status_code=404, detail={
            "code": "no_artifact", "message": "Deployment has no servable artifact.",
        }) from None
    sha = hashlib.sha256(data).hexdigest()
    size = len(data)
    model.artifact_sha256 = sha
    model.artifact_size_bytes = size
    session.add(model)
    session.commit()
    return sha, size, key


@router.get("/artifacts/{deployment_id}/info")
@limiter.limit(get_settings().rate_limit_artifact, key_func=_deployment_key)
def artifact_info(
    deployment_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
    session: Session = Depends(get_session),
) -> dict:
    _key, _dep, model = _resolve(deployment_id, authorization, session)
    sha, size, key = _artifact_meta(session, model)
    return {
        "fileName": Path(key).name,
        "sizeBytes": size,
        "sha256": sha,
        "kind": "onnx" if key.endswith(".onnx") else "npz",
    }


@router.get("/artifacts/{deployment_id}")
@limiter.limit(get_settings().rate_limit_artifact, key_func=_deployment_key)
def artifact_download(
    deployment_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
    if_none_match: str | None = Header(default=None),
    session: Session = Depends(get_session),
) -> Response:
    """Stream the deployed artifact to the SDK. ETag = sha256 (from metadata) so
    clients cache on disk and re-download only when the artifact changed; a 304
    never touches the object bytes, and a real download STREAMS from storage
    instead of buffering the whole file into the API's memory."""
    _key, _dep, model = _resolve(deployment_id, authorization, session)
    sha, size, key = _artifact_meta(session, model)
    etag = f'"{sha}"'
    if if_none_match and if_none_match.strip() == etag:
        return Response(status_code=304, headers={"ETag": etag})
    try:
        stream, stream_size = get_storage().open_stream(key)
    except StorageError:
        raise HTTPException(status_code=404, detail={
            "code": "no_artifact", "message": "Deployment has no servable artifact.",
        }) from None
    return StreamingResponse(
        stream,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{Path(key).name}"',
            "Content-Length": str(stream_size or size),
            "ETag": etag,
        },
    )
