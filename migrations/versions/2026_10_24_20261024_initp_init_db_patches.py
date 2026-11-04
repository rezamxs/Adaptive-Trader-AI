"""Transfer init_db ad-hoc ALTER patches into the Alembic chain.

Covers the additive columns init_db.py used to patch at boot:
- worker_cycle_records.lifecycle_phase  VARCHAR(30) DEFAULT 'COMPLETED'
- worker_cycle_records.updated_at       DATETIME
- ml_models.validation_metrics          JSON NOT NULL DEFAULT '{}'

Revision ID: 20261024_initp
Revises: 20261015_sha256
Create Date: 2026-10-24
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20261024_initp"
down_revision: Union[str, None] = "20261015_sha256"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the boot-patch columns only where the tables predate them."""
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())

    if "worker_cycle_records" in tables:
        columns = {column["name"] for column in inspector.get_columns("worker_cycle_records")}
        if "lifecycle_phase" not in columns:
            op.add_column(
                "worker_cycle_records",
                sa.Column("lifecycle_phase", sa.String(30), server_default="COMPLETED", nullable=True),
            )
        if "updated_at" not in columns:
            op.add_column("worker_cycle_records", sa.Column("updated_at", sa.DateTime()))

    if "ml_models" in tables:
        columns = {column["name"] for column in inspector.get_columns("ml_models")}
        if "validation_metrics" not in columns:
            op.add_column(
                "ml_models",
                sa.Column(
                    "validation_metrics",
                    sa.JSON(),
                    nullable=False,
                    server_default=sa.text("'{}'"),
                ),
            )


def downgrade() -> None:
    """Remove the transferred columns only when they exist."""
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())

    if "ml_models" in tables:
        columns = {column["name"] for column in inspector.get_columns("ml_models")}
        if "validation_metrics" in columns:
            op.drop_column("ml_models", "validation_metrics")

    if "worker_cycle_records" in tables:
        columns = {column["name"] for column in inspector.get_columns("worker_cycle_records")}
        if "updated_at" in columns:
            op.drop_column("worker_cycle_records", "updated_at")
        if "lifecycle_phase" in columns:
            op.drop_column("worker_cycle_records", "lifecycle_phase")
