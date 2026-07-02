"""feedback title: an optional user-written headline on a feedback submission

Revision ID: 0012_feedback_title
Revises: 0011_scrub_legacy_endpoint
Create Date: 2026-07-03

Adds the `title` column to the `feedback` table so a submission can carry an
explicit headline. The opened GitHub issue titles as "[Kind] {title}" (force-
prefixed with the kind); when the title is blank it falls back to the message's
first line, preserving the previous behaviour.

Idempotent and Postgres-only: a freshly-created SQLite/test DB already has this
column because the schema is built from live SQLModel metadata via create_all,
so this migration is a no-op there while an already-migrated Postgres DB gets the
column added with IF NOT EXISTS.
"""
from __future__ import annotations

from alembic import op

revision = "0012_feedback_title"
down_revision = "0011_scrub_legacy_endpoint"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # SQLite path uses create_all, which already has this column.

    op.execute("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS title VARCHAR")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE feedback DROP COLUMN IF EXISTS title")
