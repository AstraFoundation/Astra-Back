"""closed-loop telemetry idempotency: processed_batches ledger

Revision ID: 0009_processed_batches
Revises: 0008_feedback_attachment
Create Date: 2026-07-01

The on-device SDK buffers telemetry durably on disk while offline and re-sends
each spool segment (with a stable client-generated batchId) on reconnect / after
a process restart. This ledger records ingested batchIds so a duplicate re-send
is acked without double-inserting — dashboard KPIs never inflate.

Idempotent and Postgres-only (SQLite/test DBs build the schema fresh from
SQLModel metadata via create_all).
"""
from __future__ import annotations

from alembic import op

revision = "0009_processed_batches"
down_revision = "0008_feedback_attachment"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return  # SQLite path uses create_all, which already has this table.

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_batches (
            batch_id           VARCHAR PRIMARY KEY,
            deployment_id      VARCHAR NOT NULL DEFAULT '',
            received_at        VARCHAR NOT NULL DEFAULT '',
            accepted_events    INTEGER NOT NULL DEFAULT 0,
            accepted_snapshots INTEGER NOT NULL DEFAULT 0,
            accepted_windows   INTEGER NOT NULL DEFAULT 0,
            dropped            INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_processed_batches_deployment_id "
        "ON processed_batches (deployment_id)"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("DROP TABLE IF EXISTS processed_batches")
