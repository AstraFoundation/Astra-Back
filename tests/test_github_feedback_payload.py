"""The GitHub issue title is force-prefixed with the kind ([Feature]/[Bug]/…) and
uses the submitter's explicit title when given, else the message's first line."""

from __future__ import annotations

from app.dbmodels import FeedbackRow
from app.services.github_feedback import _issue_payload


def _row(**kw) -> FeedbackRow:
    base = dict(
        id="fb_test",
        kind="feature",
        message="First line\nSecond line",
        created_at="2026-07-03T00:00:00.000Z",
    )
    base.update(kw)
    return FeedbackRow(**base)


def test_explicit_title_gets_kind_prefix():
    assert _issue_payload(_row(kind="feature", title="Dark mode toggle"))["title"] == (
        "[Feature] Dark mode toggle"
    )


def test_prefix_is_capitalized_per_kind():
    assert _issue_payload(_row(kind="bug", title="It crashes"))["title"] == "[Bug] It crashes"
    assert _issue_payload(_row(kind="question", title="Why?"))["title"] == "[Question] Why?"
    assert _issue_payload(_row(kind="other", title="Thoughts"))["title"] == "[Other] Thoughts"


def test_missing_title_falls_back_to_first_message_line():
    p = _issue_payload(_row(kind="bug", title=None, message="It crashes\nmore detail"))
    assert p["title"] == "[Bug] It crashes"


def test_blank_title_falls_back_to_first_message_line():
    p = _issue_payload(_row(kind="feature", title="   ", message="Line one\ntwo"))
    assert p["title"] == "[Feature] Line one"


def test_long_title_is_truncated():
    p = _issue_payload(_row(kind="feature", title="x" * 500))
    # "[Feature] " prefix + at most 120 chars of the headline.
    assert p["title"] == "[Feature] " + "x" * 120
