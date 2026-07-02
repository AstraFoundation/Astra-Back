"""Shared HTTP plumbing: bearer auth, retry with backoff, error mapping."""

from __future__ import annotations

import os
import random
import time
from typing import Any

import httpx

_RETRYABLE_STATUS = {429, 502, 503, 504}

# The hosted Astra origin every deployment lives behind. Baked in so SDK code
# never has to carry a base URL; override with the ASTRA_BASE_URL env var or an
# explicit base_url argument (e.g. for self-host / testing).
DEFAULT_BASE_URL = "https://astra.kwon5700.kr"


def resolve_base_url(base_url: str | None) -> str:
    """The base URL to use: explicit arg → ASTRA_BASE_URL env → hosted default."""
    return (base_url or os.environ.get("ASTRA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


class AstraApiError(Exception):
    """Non-2xx response from the Astra backend."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"[{status}] {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


def error_from_response(resp: httpx.Response) -> AstraApiError:
    detail: dict[str, Any] = {}
    try:
        body = resp.json()
        detail = body.get("detail", {}) if isinstance(body, dict) else {}
        if not isinstance(detail, dict):
            detail = {"message": str(detail)}
    except ValueError:
        pass
    return AstraApiError(
        resp.status_code,
        detail.get("code", "error"),
        detail.get("message", resp.text[:500]),
    )


class HttpSession:
    """httpx.Client wrapper with bearer auth + bounded retry/backoff.

    Retries transient failures (connect errors, 429/5xx) with exponential
    backoff capped at `max_backoff`; gives up after `max_attempts` and raises
    the last error. 4xx (except 429) never retries."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        max_attempts: int = 3,
        max_backoff: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.Client(timeout=timeout)
        self._max_attempts = max(1, max_attempts)
        self._max_backoff = max_backoff

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        merged = dict(self._headers)
        if headers:
            merged.update(headers)
        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                resp = self._client.request(method, url, json=json, headers=merged)
            except httpx.HTTPError as exc:
                last_exc = exc
            else:
                if resp.status_code < 400 or resp.status_code == 304:
                    return resp
                if resp.status_code not in _RETRYABLE_STATUS:
                    raise error_from_response(resp)
                last_exc = error_from_response(resp)
            if attempt < self._max_attempts - 1:
                backoff = min(self._max_backoff, (2 ** attempt) + random.random())
                time.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    def close(self) -> None:
        self._client.close()
