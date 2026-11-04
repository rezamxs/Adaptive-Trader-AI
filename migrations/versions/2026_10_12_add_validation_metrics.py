"""Add validation metrics to existing ML model records.

Revision ID: 20261012_valid
Revises: 01f57c481926
Create Date: 2026-10-12
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20261012_valid"
down_revision: Union[str, None] = "01f57c481926"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the column only when upgrading an older database."""
    bind = op.get_bind()
    inspector = inspect(bind)
    if "ml_models" not in inspector.get_table_names():
        return
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
    """Remove the added column only when it exists."""
    bind = op.get_bind()
    inspector = inspect(bind)
    if "ml_models" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("ml_models")}
    if "validation_metrics" in columns:
        op.drop_column("ml_models", "validation_metrics")
