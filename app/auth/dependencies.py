"""FastAPI auth dependencies — resolve the current user from the session cookie."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlmodel import Session

from app.auth.security import COOKIE_NAME, decode_access_token
from app.db import get_session, open_session
from app.dbmodels import UserRow

_UNAUTH = HTTPException(
    status_code=401,
    detail={"code": "not_authenticated", "message": "Sign in to continue."},
)


def _user_from_token(session: Session, token: str | None) -> UserRow | None:
    decoded = decode_access_token(token or "")
    if not decoded:
        return None
    user_id, tv = decoded
    user = session.get(UserRow, user_id)
    if user is None:
        return None
    # Reject a token whose revocation epoch predates the user's current one
    # (password change / "log out everywhere" bumps token_version).
    if user.token_version != tv:
        return None
    return user


def get_current_user(
    request: Request,
    session: Session = Depends(get_session),
) -> UserRow:
    """Require a valid session cookie; raise a structured 401 otherwise."""
    user = _user_from_token(session, request.cookies.get(COOKIE_NAME))
    if user is None:
        raise _UNAUTH
    return user


def get_current_user_id(current_user: CurrentUser) -> str:
    return current_user.id


def current_user_id_for_stream(request: Request) -> str:
    """Auth resolver for the long-lived SSE generator, which must not hold a
    request-scoped session. Opens its own short session just to verify."""
    with open_session() as s:
        user = _user_from_token(s, request.cookies.get(COOKIE_NAME))
        if user is None:
            raise _UNAUTH
        return user.id


CurrentUser = Annotated[UserRow, Depends(get_current_user)]
