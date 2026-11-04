"""Add explicit `direction` to trading_rules (M4 fail-closed direction).

The rule name is no longer a direction source. This migration adds the
nullable `direction` column ('BUY' | 'SELL' | NULL) WITHOUT any backfill:
guessing a direction from the rule name — even once, even in a migration —
is exactly the M4 defect being removed. Existing rows keep direction=NULL,
which means every legacy rule is fail-closed: it can trigger but never casts
a direction vote until an operator assigns the direction consciously via
`POST /api/v1/rules/{rule_id}/direction` (a Phase P human decision that must
stay reviewable per the roadmap).

Inspector-guarded like 20261031_mlwfm: the initial schema revision builds the
tables from Base.metadata, so on a fresh chain the model-level column already
exists and the ADD COLUMN must be skipped.

Revision ID: 20261101_rdir
Revises: 20261031_auidx
Create Date: 2026-11-01
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "20261101_rdir"
down_revision: Union[str, None] = "20261031_auidx"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "trading_rules" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("trading_rules")}
    if "direction" in columns:
        return
    op.add_column(
        "trading_rules",
        sa.Column("direction", sa.String(length=4), nullable=True),
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "trading_rules" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("trading_rules")}
    if "direction" in columns:
        op.drop_column("trading_rules", "direction")
