"""GET /api/sdk/* — snippets is an OBJECT keyed by language (zod Record).

API keys / webhooks endpoints were removed: this is a local tool with no auth
or delivery infrastructure, and serving empty husks for them added nothing.
The SDK Hub now centers on the REAL compressed artifact
(`/api/models/{id}/artifact/info` for metadata + `/api/models/{id}/artifact`
for the download).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.db import get_session
from app.dbmodels import RecipeRow, SdkSnippetRow
from app.schemas.sdk import Recipe, SdkSnippet

router = APIRouter(prefix="/sdk", tags=["sdk"])


@lru_cache(maxsize=1)
def _sdk_version() -> str:
    """Real astra-ai-sdk client version. Source of truth, in order: the installed
    `astra_sdk` package, then the vendored client's pyproject, then a constant.
    Keeps the SDK Hub version chip honest instead of a hardcoded string."""
    try:
        import astra_sdk  # type: ignore[import-not-found]

        version = getattr(astra_sdk, "__version__", None)
        if version:
            return str(version)
    except Exception:  # noqa: BLE001 — package may not be on the backend path
        pass
    try:
        pyproject = (
            Path(__file__).resolve().parents[2] / "clients" / "python" / "pyproject.toml"
        )
        match = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', pyproject.read_text())
        if match:
            return match.group(1)
    except Exception:  # noqa: BLE001 — repo layout differs in some deployments
        pass
    return "0.5.0"


@router.get("/version")
def sdk_version() -> dict[str, str]:
    """Single source for the SDK Hub version chip — the real client version."""
    return {"version": _sdk_version()}


@router.get("/snippets")
def snippets(session: Session = Depends(get_session)) -> dict[str, SdkSnippet]:
    rows = session.exec(select(SdkSnippetRow)).all()
    return {
        r.language: SdkSnippet(language=r.language, filename=r.filename, code=r.code)  # type: ignore[arg-type]
        for r in rows
        if not r.language.startswith("_")  # "_meta" is the docs version marker
    }


@router.get("/recipes")
def recipes(session: Session = Depends(get_session)) -> list[Recipe]:
    rows = session.exec(select(RecipeRow)).all()
    return [
        Recipe(
            id=r.id, title=r.title, description=r.description,
            language=r.language, steps=json.loads(r.steps_json),  # type: ignore[arg-type]
        )
        for r in rows
    ]
