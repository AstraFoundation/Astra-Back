"""Rate limiting (slowapi) + upload validation.

The limiter uses Redis-backed storage in production so limits hold across scaled
API replicas; tests / inline mode fall back to in-memory. Validation of uploads
(extension allowlist + size cap) returns the structured {detail:{code}} shape.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings

_settings = get_settings()


def _storage_uri() -> str:
    # In-memory when there's no shared broker (tests / single box w/ inline jobs).
    return "memory://" if _settings.inline_jobs else _settings.redis_url


limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=_storage_uri(),
    enabled=_settings.rate_limit_enabled,
    # headers_enabled requires a `response` param on every decorated endpoint;
    # we don't inject limit headers, the 429 fires before the handler runs.
    headers_enabled=False,
    # Fail OPEN on a storage (Redis) error: a transient Redis blip must never
    # turn every login / signup / upload / telemetry POST into a 500. slowapi
    # then logs the error and lets the request through instead of propagating it.
    swallow_errors=True,
)


def validate_upload(file: UploadFile) -> None:
    """Reject unsupported extensions before staging bytes anywhere."""
    settings = get_settings()
    name = file.filename or ""
    ext = Path(name).suffix.lower()
    if ext not in settings.allowed_upload_ext_set:
        allowed = ", ".join(sorted(settings.allowed_upload_ext_set))
        raise HTTPException(status_code=400, detail={
            "code": "unsupported_format",
            "message": f"Unsupported file type '{ext or name}'. Allowed: {allowed}.",
        })


def enforce_size(num_bytes: int) -> None:
    settings = get_settings()
    limit = settings.max_upload_mb * 1024 * 1024
    if num_bytes > limit:
        raise HTTPException(status_code=413, detail={
            "code": "file_too_large",
            "message": f"File exceeds the {settings.max_upload_mb} MB upload limit.",
        })


def validate_feedback_image(file: UploadFile) -> None:
    """Reject non-image attachments before staging bytes. Uses the dedicated
    feedback image allowlist (NOT the model-format allowlist)."""
    settings = get_settings()
    name = file.filename or ""
    ext = Path(name).suffix.lower()
    if ext not in settings.feedback_image_ext_set:
        allowed = ", ".join(sorted(settings.feedback_image_ext_set))
        raise HTTPException(status_code=400, detail={
            "code": "unsupported_image",
            "message": f"Unsupported image type '{ext or name}'. Allowed: {allowed}.",
        })


def enforce_image_size(num_bytes: int) -> None:
    settings = get_settings()
    limit = settings.feedback_image_max_mb * 1024 * 1024
    if num_bytes > limit:
        raise HTTPException(status_code=413, detail={
            "code": "image_too_large",
            "message": f"Image exceeds the {settings.feedback_image_max_mb} MB limit.",
        })
