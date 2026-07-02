"""prod hardening: artifact identity, session revocation, snapshot index

Revision ID: 0010_prod_hardening
Revises: 0009_processed_batches
Create Date: 2026-07-02

Three additive, idempotent changes (Postgres-only; SQLite/test DBs build the
schema fresh from SQLModel metadata via create_all):

  * models.artifact_sha256 / artifact_size_bytes — persisted artifact identity so
    the SDK /artifacts endpoints serve the ETag/size from metadata instead of
    reading the whole object + re-hashing it on every call (OOM guard, #5).
  * users.token_version — session-revocation epoch so logout / password change /
    "log out everywhere" can evict an outstanding 7-day session JWT (#12).
  * ix_snap_model_ts on telemetry_snapshots(model_id, ts) — the hardware/resource
    dashboards filter model_id + order by ts; without it they scan the model's
    whole snapshot history (#18).
"""
from __future__ import annotations

from alembic import op

revision = "0010_prod_hardening"
down_revision = "0009_processed_batches"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # SQLite path uses create_all, which already has these.

    op.execute("ALTER TABLE models ADD COLUMN IF NOT EXISTS artifact_sha256 VARCHAR")
    op.execute("ALTER TABLE models ADD COLUMN IF NOT EXISTS artifact_size_bytes INTEGER")
    op.execute(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS token_version INTEGER NOT NULL DEFAULT 0"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_snap_model_ts "
        "ON telemetry_snapshots (model_id, ts)"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("DROP INDEX IF EXISTS ix_snap_model_ts")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS token_version")
    op.execute("ALTER TABLE models DROP COLUMN IF EXISTS artifact_size_bytes")
    op.execute("ALTER TABLE models DROP COLUMN IF EXISTS artifact_sha256")
