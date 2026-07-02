"""Authentication endpoints: signup / login / logout / me / profile / password.

All validation failures return {detail: {code, message}} — the same structured
shape the SPA's apiErrorCode() parses for inline form errors.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from datetime import datetime, timezone

from email_validator import EmailNotValidError, validate_email
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from sqlmodel import Session, select

from app.auth import google
from app.auth.dependencies import CurrentUser
from app.auth.security import (
    clear_session_cookie,
    create_access_token,
    hash_password,
    set_session_cookie,
    verify_password,
)
from app.config import get_settings, iso
from app.db import get_session
from app.dbmodels import UserRow
from app.schemas.auth import (
    ChangePasswordRequest,
    LoginRequest,
    MeResponse,
    SignupRequest,
    UpdateProfileRequest,
)
from app.schemas.common import OkResponse
from app.services.limits import limiter

router = APIRouter(prefix="/auth", tags=["auth"])
log = logging.getLogger("astra.auth")

MIN_PASSWORD_LEN = 8
MAX_PASSWORD_LEN = 200
OAUTH_STATE_COOKIE = "astra_oauth_state"


def _err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _normalize_email(raw: str) -> str:
    try:
        # check_deliverability=False: don't do DNS at signup time.
        info = validate_email(raw.strip(), check_deliverability=False)
    except EmailNotValidError:
        raise _err(400, "invalid_email", "Enter a valid email address.") from None
    return info.normalized.lower()


def _check_password(pw: str) -> None:
    if len(pw) < MIN_PASSWORD_LEN:
        raise _err(400, "weak_password",
                   f"Password must be at least {MIN_PASSWORD_LEN} characters.")
    if len(pw) > MAX_PASSWORD_LEN:
        raise _err(400, "weak_password", "Password is too long.")


def _check_name(name: str) -> str:
    name = name.strip()
    if not 1 <= len(name) <= 80:
        raise _err(400, "invalid_name", "Name must be 1–80 characters.")
    return name


def _me(user: UserRow) -> MeResponse:
    return MeResponse(
        id=user.id, email=user.email, name=user.name,
        role=user.role, createdAt=user.created_at, authProvider=user.auth_provider,
    )


@router.post("/signup", response_model=MeResponse)
@limiter.limit(get_settings().rate_limit_auth)
def signup(
    request: Request,
    body: SignupRequest,
    response: Response,
    session: Session = Depends(get_session),
) -> MeResponse:
    if not get_settings().signup_enabled:
        raise _err(403, "signup_disabled", "Sign-ups are currently disabled.")
    email = _normalize_email(body.email)
    name = _check_name(body.name)
    _check_password(body.password)

    exists = session.exec(select(UserRow).where(UserRow.email == email)).first()
    if exists is not None:
        raise _err(409, "email_taken", "An account with this email already exists.")

    user = UserRow(
        id=f"u_{uuid.uuid4().hex[:12]}",
        email=email,
        password_hash=hash_password(body.password),
        name=name,
        created_at=iso(datetime.now(timezone.utc)),
    )
    session.add(user)
    session.commit()
    set_session_cookie(response, create_access_token(user.id, user.token_version))
    return _me(user)


@router.post("/login", response_model=MeResponse)
@limiter.limit(get_settings().rate_limit_auth)
def login(
    request: Request,
    body: LoginRequest,
    response: Response,
    session: Session = Depends(get_session),
) -> MeResponse:
    email = body.email.strip().lower()
    user = session.exec(select(UserRow).where(UserRow.email == email)).first()
    # Same message for unknown email and wrong password — no user enumeration.
    if user is None or not verify_password(body.password, user.password_hash):
        raise _err(401, "invalid_credentials", "Incorrect email or password.")
    set_session_cookie(response, create_access_token(user.id, user.token_version))
    return _me(user)


@router.post("/logout", response_model=OkResponse)
def logout(response: Response) -> OkResponse:
    clear_session_cookie(response)
    return OkResponse()


@router.get("/me", response_model=MeResponse)
def me(current_user: CurrentUser) -> MeResponse:
    return _me(current_user)


@router.patch("/me", response_model=MeResponse)
def update_profile(
    body: UpdateProfileRequest,
    current_user: CurrentUser,
    session: Session = Depends(get_session),
) -> MeResponse:
    current_user.name = _check_name(body.name)
    session.add(current_user)
    session.commit()
    session.refresh(current_user)
    return _me(current_user)


@router.post("/me/password", response_model=OkResponse)
@limiter.limit(get_settings().rate_limit_auth)
def change_password(
    request: Request,
    body: ChangePasswordRequest,
    current_user: CurrentUser,
    response: Response,
    session: Session = Depends(get_session),
) -> OkResponse:
    if not verify_password(body.currentPassword, current_user.password_hash):
        raise _err(400, "invalid_credentials", "Current password is incorrect.")
    _check_password(body.newPassword)
    current_user.password_hash = hash_password(body.newPassword)
    # Bump the revocation epoch so EVERY outstanding session token is invalidated
    # (an attacker holding a stolen token is evicted), then re-issue a fresh cookie
    # for THIS session so the user who just changed their password stays signed in.
    current_user.token_version += 1
    session.add(current_user)
    session.commit()
    set_session_cookie(response, create_access_token(current_user.id, current_user.token_version))
    return OkResponse()


# ── Google OAuth ──────────────────────────────────────────────────────────────

@router.get("/providers")
def providers() -> dict:
    """Public — tells the SPA which sign-in methods to offer."""
    return {"password": True, "google": get_settings().google_enabled}


def _set_state_cookie(response: Response, state: str) -> None:
    settings = get_settings()
    response.set_cookie(
        key=OAUTH_STATE_COOKIE, value=state, httponly=True,
        secure=settings.cookie_secure, samesite=settings.cookie_samesite,  # type: ignore[arg-type]
        path="/", max_age=600, domain=settings.cookie_domain,
    )


def _login_redirect(error: str | None = None) -> RedirectResponse:
    # Relative path → resolves to the same origin Caddy serves the SPA from.
    url = "/login" + (f"?error={error}" if error else "")
    return RedirectResponse(url, status_code=302)


@router.get("/google/login")
def google_login() -> RedirectResponse:
    if not get_settings().google_enabled:
        return _login_redirect("google_disabled")
    state = secrets.token_urlsafe(24)
    resp = RedirectResponse(google.build_auth_url(state), status_code=302)
    _set_state_cookie(resp, state)
    return resp


def _provision_google_user(session: Session, claims: dict) -> UserRow:
    email = (claims.get("email") or "").strip().lower()
    sub = claims.get("sub")
    name = (claims.get("name") or email.split("@")[0] or "User").strip()[:80]

    # 1) Returning Google user (matched by stable sub).
    if sub:
        existing = session.exec(select(UserRow).where(UserRow.google_sub == sub)).first()
        if existing is not None:
            return existing
    # 2) Existing account with the same (verified) email → link Google to it.
    by_email = session.exec(select(UserRow).where(UserRow.email == email)).first()
    if by_email is not None:
        if not by_email.google_sub and sub:
            by_email.google_sub = sub
            session.add(by_email)
            session.commit()
        return by_email
    # 3) Brand-new Google account (no password).
    user = UserRow(
        id=f"u_{uuid.uuid4().hex[:12]}",
        email=email,
        password_hash=None,
        name=name,
        created_at=iso(datetime.now(timezone.utc)),
        auth_provider="google",
        google_sub=sub,
    )
    session.add(user)
    session.commit()
    return user


@router.get("/google/callback")
def google_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    session: Session = Depends(get_session),
) -> RedirectResponse:
    if not get_settings().google_enabled:
        return _login_redirect("google_disabled")
    if error or not code:
        log.warning("google callback error param: %s", error)
        return _login_redirect("google")

    cookie_state = request.cookies.get(OAUTH_STATE_COOKIE)
    if not state or not cookie_state or not secrets.compare_digest(state, cookie_state):
        log.warning("google callback state mismatch")
        return _login_redirect("state")

    try:
        tokens = google.exchange_code(code)
        claims = google.verify_id_token(tokens["id_token"])
    except google.GoogleAuthError as exc:
        log.warning("google oauth exchange/verify failed: %s", exc)
        return _login_redirect("google")

    if not claims.get("email") or not claims.get("email_verified"):
        log.warning("google account has no verified email")
        return _login_redirect("email_unverified")

    user = _provision_google_user(session, claims)

    resp = RedirectResponse(get_settings().post_login_path, status_code=302)
    set_session_cookie(resp, create_access_token(user.id, user.token_version))
    resp.delete_cookie(
        OAUTH_STATE_COOKIE, path="/",
        samesite=get_settings().cookie_samesite,  # type: ignore[arg-type]
        secure=get_settings().cookie_secure, httponly=True,
        domain=get_settings().cookie_domain,
    )
    return resp
