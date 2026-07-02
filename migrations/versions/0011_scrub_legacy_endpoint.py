"""scrub legacy deployment endpoint URLs (removed /infer path + peops host)

Revision ID: 0011_scrub_legacy_endpoint
Revises: 0010_prod_hardening
Create Date: 2026-07-02

`deployments.endpoint` is an absolute URL frozen at creation. Rows created before
the on-device pivot (SDK 0.3.0) hold the removed hosted path `/api/v1/infer/<id>`,
and rows from before the PEOps→Astra rebrand hold a `peops.` host. The UI no
longer displays this URL (it shows the deployment id), but scrub the stored value
so it's consistent: /infer → /artifacts (the real pull path) and peops → astra.

Idempotent, Postgres-only (SQLite/test DBs build fresh from SQLModel metadata).
"""
from __future__ import annotations

from alembic import op

revision = "0011_scrub_legacy_endpoint"
down_revision = "0010_prod_hardening"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute(
        "UPDATE deployments SET endpoint = "
        "replace(replace(endpoint, '/api/v1/infer/', '/api/v1/artifacts/'), "
        "'peops.kwon5700.kr', 'astra.kwon5700.kr') "
        "WHERE endpoint LIKE '%/api/v1/infer/%' OR endpoint LIKE '%peops.%'"
    )


def downgrade() -> None:
    # One-way data cleanup — the old hosted /infer path is a dead route; nothing
    # to restore it to.
    pass
