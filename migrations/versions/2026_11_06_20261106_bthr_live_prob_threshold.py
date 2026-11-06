"""ml_models.live_prob_threshold — record the live-inference probability threshold.

Buglist item 33: live inference filtered at a hidden 0.55 function default
while the promotion evidence payloads were evaluated at the fixed 0.5, and
NEITHER value was stored with the model. The threshold is now an explicit,
recorded model attribute (this column + the attested artifact payload field);
the evidence threshold stays 0.5 by design and the two are documented side by
side.

Inspector-guarded like the sibling revisions: a fresh chain builds the column
from Base.metadata, so a blind ADD COLUMN would double-apply.

Revision ID: 20261106_bthr
Revises: 20261106_fkif
Create Date: 2026-11-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "20261106_bthr"
down_revision: Union[str, None] = "20261106_fkif"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "ml_models" not in inspector.get_table_names():
        return
    columns = {c["name"] for c in inspector.get_columns("ml_models")}
    if "live_prob_threshold" in columns:
        return
    op.add_column(
        "ml_models",
        sa.Column("live_prob_threshold", sa.Float(), nullable=False,
                  server_default="0.55"),
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "ml_models" not in inspector.get_table_names():
        return
    columns = {c["name"] for c in inspector.get_columns("ml_models")}
    if "live_prob_threshold" not in columns:
        return
    op.drop_column("ml_models", "live_prob_threshold")
