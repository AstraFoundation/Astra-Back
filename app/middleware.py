"""Structured request logging + X-Request-ID propagation (API and worker share
the logging config)."""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import Settings, get_settings

_access = logging.getLogger("astra.access")

# A caller-supplied X-Request-ID is only accepted if it's short and charset-safe;
# otherwise we mint our own. Prevents an oversized/forged id from muddying logs
# or being reflected verbatim in headers/error bodies.
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# The SDK telemetry batch is small (≤500 compact items); reject an oversized body
# by Content-Length BEFORE FastAPI/Pydantic materializes it into memory (OOM guard).
_TELEMETRY_PREFIX = "/api/v1/telemetry/"


def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler()
    if settings.log_json:
        from pythonjsonlogger import jsonlogger

        handler.setFormatter(jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
        ))
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
        ))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())
    # uvicorn duplicates access logs — let our middleware own them.
    logging.getLogger("uvicorn.access").handlers = []


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        inbound = request.headers.get("x-request-id") or ""
        rid = inbound if _REQUEST_ID_RE.match(inbound) else uuid.uuid4().hex[:12]
        request.state.request_id = rid

        # OOM guard: reject an oversized telemetry batch by Content-Length before
        # the body is read/parsed. The per-item 500 cap only fires post-parse, so
        # without this a single huge body is materialized into RAM first.
        if request.url.path.startswith(_TELEMETRY_PREFIX) and request.method == "POST":
            cl = request.headers.get("content-length")
            if cl and cl.isdigit():
                cap = get_settings().telemetry_body_max_mb * 1024 * 1024
                if int(cl) > cap:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": {"code": "payload_too_large",
                                            "message": "Telemetry batch body too large."}},
                        headers={"X-Request-ID": rid},
                    )

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration = (time.perf_counter() - start) * 1000
            _access.exception(
                "request_error",
                extra={"request_id": rid, "method": request.method,
                       "path": request.url.path, "duration_ms": round(duration, 1)},
            )
            raise
        duration = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = rid
        # Skip the health probes' chatter at INFO.
        if request.url.path not in ("/healthz", "/readyz"):
            _access.info(
                "request",
                extra={"request_id": rid, "method": request.method,
                       "path": request.url.path, "status": response.status_code,
                       "duration_ms": round(duration, 1)},
            )
        return response
