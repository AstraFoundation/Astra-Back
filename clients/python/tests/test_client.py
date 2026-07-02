"""HttpSession (shared transport) against httpx.MockTransport."""

from __future__ import annotations

import httpx
import pytest

from astra_sdk._http import AstraApiError, HttpSession


def _patch_transport(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    original_init = httpx.Client.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = transport
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", patched)


def test_http_session_retries_503(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"ok": True})

    _patch_transport(monkeypatch, handler)
    monkeypatch.setattr("astra_sdk._http.time.sleep", lambda _s: None)
    s = HttpSession("http://t", "k", max_attempts=3)
    assert s.request("GET", "/x").json() == {"ok": True}
    assert calls["n"] == 3
    s.close()


def test_http_session_no_retry_on_401(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, json={"detail": {"code": "invalid_api_key",
                                                    "message": "bad"}})

    _patch_transport(monkeypatch, handler)
    s = HttpSession("http://t", "k", max_attempts=3)
    with pytest.raises(AstraApiError) as exc:
        s.request("GET", "/x")
    assert exc.value.code == "invalid_api_key"
    assert calls["n"] == 1
    s.close()
