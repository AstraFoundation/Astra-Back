"""In-app feedback / feature-request intake (cookie-authed dashboard side).

A submission is always persisted — the feedback row is the record. When a target
GitHub repo is configured it's also opened as an issue via a background task, so
the POST stays fast and a GitHub outage can never drop feedback. The submitter
identity is taken from the session (CurrentUser), never trusted from the client.

The submission is multipart so an optional screenshot can ride along: the image
is streamed into object storage and served back (session-gated) from a sibling
GET, and linked from the GitHub issue body.
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from app.auth.dependencies import CurrentUser
from app.config import get_settings, iso
from app.db import get_session
from app.dbmodels import FeedbackRow
from app.schemas.feedback import FeedbackKind, FeedbackResult
from app.services.github_feedback import create_issue_for_feedback
from app.services.limits import enforce_image_size, validate_feedback_image
from app.services.storage import StorageError, feedback_attachment_key, get_storage

router = APIRouter(tags=["feedback"])

_MAX_MESSAGE = 4000
_MAX_PAGE = 300
_MAX_LOCALE = 16

# Suffix → response media type for serving a stored attachment back inline.
_IMAGE_MEDIA = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@router.post("/feedback")
async def submit_feedback(
    request: Request,
    current_user: CurrentUser,
    background: BackgroundTasks,
    kind: FeedbackKind = Form("feature"),
    message: str = Form(...),
    page: str | None = Form(None),
    locale: str | None = Form(None),
    image: UploadFile | None = File(None),
    session: Session = Depends(get_session),
) -> FeedbackResult:
    clean = message.strip()[:_MAX_MESSAGE]
    if not clean:
        raise HTTPException(status_code=422, detail={
            "code": "empty_message",
            "message": "Feedback message must not be empty.",
        })

    feedback_id = f"fb_{uuid.uuid4().hex[:10]}"

    # Optional screenshot: validate the type, then stream into object storage
    # (counting bytes for the size cap). The temp file is always cleaned up; the
    # storage object is only written after the whole image is staged, so a size
    # trip mid-stream leaves nothing behind.
    attachment_key: str | None = None
    attachment_name: str | None = None
    if image is not None and image.filename:
        validate_feedback_image(image)
        attachment_name = image.filename
        attachment_key = feedback_attachment_key(feedback_id, image.filename)
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp_path = tmp.name
        total = 0
        try:
            while chunk := await image.read(1 << 20):
                total += len(chunk)
                enforce_image_size(total)
                tmp.write(chunk)
            tmp.close()
            get_storage().upload_file(tmp_path, attachment_key)
        finally:
            tmp.close()
            Path(tmp_path).unlink(missing_ok=True)

    row = FeedbackRow(
        id=feedback_id,
        user_id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        kind=kind,
        message=clean,
        page=page.strip()[:_MAX_PAGE] if page else None,
        locale=locale.strip()[:_MAX_LOCALE] if locale else None,
        status="open",
        created_at=iso(datetime.now(timezone.utc)),
        attachment_key=attachment_key,
        attachment_name=attachment_name,
    )
    session.add(row)
    session.commit()

    # Open the GitHub issue after the response is sent — keeps submit snappy, and
    # a GitHub failure can never fail the user's submission (the row is saved).
    # public_origin gives a stable absolute base for the attachment link; fall
    # back to the request host for dev / single-origin deploys.
    base_url = (get_settings().public_origin or str(request.base_url)).rstrip("/")
    background.add_task(create_issue_for_feedback, row.id, base_url)

    return FeedbackResult(id=row.id, status=row.status, githubIssueUrl=row.github_issue_url)


@router.get("/feedback/{feedback_id}/attachment")
def feedback_attachment(
    feedback_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """Serve a feedback screenshot. Scoped to the OWNER: a signed-in user can only
    fetch their own feedback's attachment. (Previously any authenticated user could
    read any feedback screenshot by guessing/learning its id — a cross-tenant read.)"""
    row = session.get(FeedbackRow, feedback_id)
    if row is None or not row.attachment_key or row.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="no attachment for this feedback")
    try:
        stream, size = get_storage().open_stream(row.attachment_key)
    except StorageError:
        raise HTTPException(
            status_code=404, detail="no attachment for this feedback"
        ) from None
    name = row.attachment_name or Path(row.attachment_key).name
    media = _IMAGE_MEDIA.get(Path(name).suffix.lower(), "application/octet-stream")
    return StreamingResponse(
        stream,
        media_type=media,
        headers={
            # inline so a browser-opened link renders the screenshot directly.
            "Content-Disposition": f'inline; filename="{name}"',
            "Content-Length": str(size),
        },
    )
