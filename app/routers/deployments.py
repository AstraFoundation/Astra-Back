"""Deployment lifecycle (cookie-authed dashboard side).

Deploying a compressed model mints a DeploymentRow + an API key. Astra never runs
the model server-side — inference is on-device via the SDK's AstraRunner — so the
returned endpoint URL is the real, callable artifact-pull path
`/api/v1/artifacts/{id}` that `AstraRunner.from_deployment(id, key)` fetches
(and caches) before serving locally. Live metrics on the row stay zero until the
drift monitor fills them from the SDK's closed-loop telemetry (inference_events).
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, select

from app.auth.dependencies import CurrentUser
from app.config import get_settings, iso
from app.db import get_session
from app.dbmodels import ActivityRow, ApiKeyRow, DeploymentRow, ModelRow
from app.repositories import owned_model
from app.schemas.deployments import (
    CreatedDeployment,
    CreateDeploymentRequest,
    DeploymentItem,
    RotatedKey,
    UpdateDeploymentRequest,
)
from app.services import apikeys
from app.services.formats import is_executable

router = APIRouter(tags=["deployments"])

# Bounds on the user-editable label fields (mirrors the frontend input limits).
_MAX_NAME = 80
_MAX_DESC = 280

# Models are served on-device by the SDK's AstraRunner (onnxruntime), not in a
# cloud region — so we label the deployment with its real serving runtime
# instead of a fictional AWS region.
_SERVING_RUNTIME = "ONNX Runtime (on-device)"
# Legacy rows / older demo seeds stored a fake AWS-style region (e.g.
# "ap-northeast-2"); detect those so the list view shows the real runtime.
_CLOUD_REGION_RE = re.compile(r"^[a-z]{2}-[a-z]+-\d+$")


def _runtime_label(stored: str | None) -> str:
    """Display label for where a deployment runs. Empty or cloud-region-looking
    values are treated as legacy and replaced with the real serving runtime."""
    if stored and not _CLOUD_REGION_RE.match(stored):
        return stored
    return _SERVING_RUNTIME


def _clean_name(name: str, stored_region: str | None) -> str:
    """Strip the legacy ' · <cloud-region>' suffix old rows baked into the name."""
    if stored_region and _CLOUD_REGION_RE.match(stored_region):
        suffix = f" · {stored_region}"
        if name.endswith(suffix):
            return name[: -len(suffix)] or name
    return name


def _now() -> str:
    return iso(datetime.now(timezone.utc))


def _base_url(request: Request) -> str:
    return (get_settings().public_origin or str(request.base_url)).rstrip("/")


def _owned_deployment(session: Session, deployment_id: str, user_id: str) -> DeploymentRow:
    dep = session.exec(
        select(DeploymentRow).where(DeploymentRow.id == deployment_id)
    ).first()
    if dep is None or dep.user_id != user_id:
        raise HTTPException(status_code=404, detail="deployment not found")
    return dep


def _key_prefix(session: Session, deployment_id: str) -> str | None:
    rows = session.exec(
        select(ApiKeyRow).where(
            ApiKeyRow.deployment_id == deployment_id,
            ApiKeyRow.revoked == False,  # noqa: E712
        )
    ).all()
    if not rows:
        return None
    rows.sort(key=lambda r: r.created_at, reverse=True)
    return rows[0].prefix


def _to_item(session: Session, dep: DeploymentRow) -> DeploymentItem:
    return DeploymentItem(
        id=dep.id,
        name=_clean_name(dep.name or dep.id, dep.region),
        description=dep.description,
        endpoint=dep.endpoint,
        region=_runtime_label(dep.region),
        status=dep.status,  # type: ignore[arg-type]
        qps=dep.qps,
        p95=dep.p95,
        errorsPct=dep.errors_pct,
        accuracyDrift=dep.accuracy_drift,
        keyPrefix=_key_prefix(session, dep.id),
        createdAt=dep.created_at,
        lastEventAt=dep.last_event_at,
    )


@router.get("/models/{model_id}/deployments")
def list_deployments(
    model_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> list[DeploymentItem]:
    owned_model(session, model_id, current_user.id)
    rows = session.exec(
        select(DeploymentRow).where(DeploymentRow.model_id == model_id)
    ).all()
    rows.sort(key=lambda r: r.created_at)
    return [_to_item(session, d) for d in rows]


@router.post("/models/{model_id}/deployments")
def create_deployment(
    model_id: str,
    body: CreateDeploymentRequest,
    request: Request,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> CreatedDeployment:
    model = owned_model(session, model_id, current_user.id)
    if not is_executable(model.artifact_key):
        raise HTTPException(status_code=409, detail={
            "code": "not_servable",
            "message": "Only compressed ONNX models can be deployed. This model "
                       "has no executable artifact (weights-only or not yet "
                       "compressed).",
        })

    dep_id = f"dep_{uuid.uuid4().hex[:10]}"
    now = _now()
    # `region` is an optional free-form serving-target override; default to the
    # real runtime rather than a fictional cloud region.
    runtime = (body.region or _SERVING_RUNTIME).strip()
    dep = DeploymentRow(
        id=dep_id,
        user_id=current_user.id,
        model_id=model_id,
        name=body.name or model.name,
        endpoint=f"{_base_url(request)}/api/v1/artifacts/{dep_id}",
        region=runtime,
        qps=0.0, p95=0.0, errors_pct=0.0, accuracy_drift=0.0,
        status=body.status,
        created_at=now,
        last_event_at=None,
    )
    session.add(dep)
    model.is_deployed = True
    # Surface the deploy in the model's lifecycle status too, so the AI Models
    # list shows the green "Deployed" badge — not the stale "draft". `is_deployed`
    # alone never drove the badge (the list renders from `status`).
    model.status = "deployed"
    session.add(model)
    session.add(ActivityRow(
        id=f"act_dep_{dep_id}", user_id=current_user.id, kind="deploy_promoted",
        text=f"Deployment promoted — {model.name} → {_runtime_label(runtime)}", timestamp=now,
    ))
    session.commit()

    _, plaintext = apikeys.issue_key(
        session, user_id=current_user.id, deployment_id=dep_id,
    )
    session.refresh(dep)
    return CreatedDeployment(deployment=_to_item(session, dep), apiKey=plaintext)


@router.post("/deployments/{deployment_id}/rotate-key")
def rotate_key(
    deployment_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> RotatedKey:
    _owned_deployment(session, deployment_id, current_user.id)
    row, plaintext = apikeys.rotate_key(
        session, user_id=current_user.id, deployment_id=deployment_id,
    )
    return RotatedKey(apiKey=plaintext, keyPrefix=row.prefix)


@router.post("/deployments/{deployment_id}/pause")
def pause_deployment(
    deployment_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> DeploymentItem:
    dep = _owned_deployment(session, deployment_id, current_user.id)
    dep.status = "paused"
    session.add(dep)
    session.commit()
    return _to_item(session, dep)


@router.post("/deployments/{deployment_id}/resume")
def resume_deployment(
    deployment_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> DeploymentItem:
    dep = _owned_deployment(session, deployment_id, current_user.id)
    dep.status = "live"
    session.add(dep)
    session.commit()
    return _to_item(session, dep)


@router.patch("/deployments/{deployment_id}")
def update_deployment(
    deployment_id: str,
    body: UpdateDeploymentRequest,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> DeploymentItem:
    """Rename a deployment and/or set its description (partial update)."""
    dep = _owned_deployment(session, deployment_id, current_user.id)
    if body.name is not None:
        name = body.name.strip()[:_MAX_NAME]
        if name:  # ignore a blank name — keep the existing one
            dep.name = name
    if body.description is not None:
        dep.description = body.description.strip()[:_MAX_DESC]  # may clear to ""
    session.add(dep)
    session.commit()
    return _to_item(session, dep)


@router.delete("/deployments/{deployment_id}")
def delete_deployment(
    deployment_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> dict:
    dep = _owned_deployment(session, deployment_id, current_user.id)
    model_id = dep.model_id
    apikeys.revoke_deployment_keys(session, deployment_id)
    session.delete(dep)
    session.commit()
    # Reflect deploy state on the model when its last live deployment is gone.
    remaining = session.exec(
        select(DeploymentRow).where(DeploymentRow.model_id == model_id)
    ).all()
    if not remaining:
        model = session.get(ModelRow, model_id)
        if model is not None:
            model.is_deployed = False
            # Mirror of create_deployment: undeploying returns the model to its
            # ready-to-deploy "draft" state (the compressed artifact still exists).
            # Guarded so a re-optimization mid-flight (analyzing/optimizing) or a
            # failed model isn't clobbered.
            if model.status == "deployed":
                model.status = "draft"
            session.add(model)
            session.commit()
    return {"ok": True}
