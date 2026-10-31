"""Add metrics_digest attestation column to ml_models (audit fix H1).

Revision ID: 20261024_mdig
Revises: 20261024_initp
Create Date: 2026-10-24
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20261024_mdig"
down_revision: Union[str, None] = "20261024_initp"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "ml_models" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("ml_models")}
    if "metrics_digest" not in columns:
        op.add_column("ml_models", sa.Column("metrics_digest", sa.String(64), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "ml_models" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("ml_models")}
    if "metrics_digest" in columns:
        op.drop_column("ml_models", "metrics_digest")
