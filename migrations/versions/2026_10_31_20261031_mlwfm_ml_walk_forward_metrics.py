"""Add ml_walk_forward_metrics evidence column to ml_models (M8).

Per-fold-retrained ML walk-forward evidence (classification-metric
generalization across purged/embargoed out-of-sample folds), stored
alongside the existing strategy walk-forward payload.

Revision ID: 20261031_mlwfm
Revises: 20261029_wcbf
Create Date: 2026-10-31
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20261031_mlwfm"
down_revision: Union[str, None] = "20261029_wcbf"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "ml_models" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("ml_models")}
    if "ml_walk_forward_metrics" not in columns:
        op.add_column(
            "ml_models",
            sa.Column("ml_walk_forward_metrics", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "ml_models" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("ml_models")}
    if "ml_walk_forward_metrics" in columns:
        op.drop_column("ml_models", "ml_walk_forward_metrics")
