"""Add suggested_direction to trading_rules (step 3.3 batch human review).

The extractor may now SUGGEST a direction from the document TEXT; the field
is informational and never executable — the generator reads `direction` only,
so a suggestion can never vote (M4 stays intact). Humans adopt, override or
ignore the suggestion in the batch review.

Inspector-guarded like 20261101_rdir: the initial schema revision builds the
tables from Base.metadata, so on a fresh chain the model-level column already
exists and the ADD COLUMN must be skipped.

Revision ID: 20261102_sdir
Revises: 20261101_rdir
Create Date: 2026-11-02
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "20261102_sdir"
down_revision: Union[str, None] = "20261101_rdir"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "trading_rules" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("trading_rules")}
    if "suggested_direction" in columns:
        return
    op.add_column(
        "trading_rules",
        sa.Column("suggested_direction", sa.String(length=4), nullable=True),
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "trading_rules" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("trading_rules")}
    if "suggested_direction" in columns:
        op.drop_column("trading_rules", "suggested_direction")
