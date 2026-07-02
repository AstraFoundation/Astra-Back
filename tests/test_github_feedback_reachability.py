"""The startup reachability probe for the feedback→GitHub bridge.

``create_issue_for_feedback`` swallows failures by design, so this probe is the one
place a wrong repo / bad token surfaces — at boot. These tests pin its four outcomes
(disabled, reachable, missing repo, network error) without touching the network."""

from __future__ import annotations

import httpx

from app.config import get_settings
from app.services import github_feedback


def _enable(monkeypatch) -> None:
    """Configure a token + repo on the cached settings instance (same pattern as
    tests/test_google_oauth.py). get_settings() is lru-cached, so the probe reads
    exactly the instance we patch here."""
    s = get_settings()
    monkeypatch.setattr(s, "feedback_github_token", "test-token")
    monkeypatch.setattr(s, "feedback_github_repo", "AstraFoundation/Astra-Issue")


def _fake_get(status: int):
    def _get(url, headers=None, timeout=None):
        return httpx.Response(status, request=httpx.Request("GET", url))
    return _get


def test_disabled_returns_none_without_network(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "feedback_github_token", None)
    monkeypatch.setattr(s, "feedback_github_repo", None)

    def _boom(*a, **k):  # a disabled feature must not hit the network at all
        raise AssertionError("reachability probe called the network while disabled")

    monkeypatch.setattr(github_feedback.httpx, "get", _boom)
    assert github_feedback.check_feedback_github_reachable() is None


def test_reachable_repo_returns_none(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(github_feedback.httpx, "get", _fake_get(200))
    assert github_feedback.check_feedback_github_reachable() is None


def test_missing_repo_returns_actionable_warning(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(github_feedback.httpx, "get", _fake_get(404))
    msg = github_feedback.check_feedback_github_reachable()
    assert msg is not None
    assert "404" in msg
    assert "AstraFoundation/Astra-Issue" in msg
    assert "ASTRA_FEEDBACK_GITHUB_REPO" in msg


def test_network_error_returns_warning_not_raise(monkeypatch):
    _enable(monkeypatch)

    def _raise(*a, **k):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(github_feedback.httpx, "get", _raise)
    msg = github_feedback.check_feedback_github_reachable()
    assert msg is not None
    assert "unconfirmed" in msg.lower()
