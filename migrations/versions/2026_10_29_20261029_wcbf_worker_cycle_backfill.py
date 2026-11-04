"""worker_cycle_records: backfill legacy NULL updated_at and enforce NOT NULL.

Revision ID: 20261029_wcbf
Revises: 20261024_mdig
Create Date: 2026-10-29

Item 21: the initp migration created lifecycle_phase/updated_at as nullable,
while the model declares them NOT NULL. 599/609 production rows carried NULL
updated_at (legacy cycles written before the column existed). This revision
backfills updated_at from created_at (the honest creation time), then rebuilds
both columns as NOT NULL via batch_alter_table (SQLite-safe).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20261029_wcbf"
down_revision: Union[str, None] = "20261024_mdig"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _rebuild_columns(nullable: bool) -> None:
    """SQLite cannot ALTER COLUMN in place; batch_alter_table recreates the
    table with the requested nullability. server_default on lifecycle_phase is
    preserved so future inserts without a phase keep working."""
    with op.batch_alter_table("worker_cycle_records") as batch:
        batch.alter_column(
            "lifecycle_phase",
            existing_type=sa.String(30),
            nullable=nullable,
            server_default="COMPLETED",
        )
        batch.alter_column(
            "updated_at",
            existing_type=sa.DateTime(),
            nullable=nullable,
        )


def upgrade() -> None:
    # Backfill first: legacy rows predate the column and their honest
    # last-update time IS their creation time.
    op.execute(
        "UPDATE worker_cycle_records SET updated_at = created_at WHERE updated_at IS NULL"
    )
    _rebuild_columns(nullable=False)


def downgrade() -> None:
    # Restore ONLY the legacy nullability. The backfilled updated_at values
    # are real row data (legacy rows' creation times, modern rows' real
    # update times) and MUST survive the downgrade — nulling them here would
    # destroy data the next upgrade cannot recover.
    _rebuild_columns(nullable=True)
