"""ML walk-forward run binding (roadmap 2.4) + artifact byte digest.

Two additive changes in one revision:
1. New table ``ml_walk_forward_runs`` — the persisted per-fold-retrained ML
   walk-forward evaluation. Evidence is persisted at GENERATION time with its
   digest; models attach only evidence matching a persisted row, and the
   promotion gate requires that binding (closes the M8 self-attestation
   residual).
2. ``ml_models.artifact_sha256`` — SHA-256 over the artifact file bytes,
   verified BEFORE every joblib.load (joblib.load executes pickle code, so
   byte integrity must be proven pre-load). Legacy rows stay NULL and fail
   closed on load.

Inspector-guarded like the sibling revisions: on a fresh chain the initial
revision builds tables from Base.metadata (which already contains both
changes), so a blind CREATE TABLE / ADD COLUMN would double-apply.

Revision ID: 20261104_mlwfr
Revises: 20261102_sdir
Create Date: 2026-11-04
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "20261104_mlwfr"
down_revision: Union[str, None] = "20261102_sdir"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = "ml_walk_forward_runs"
_COLUMNS = ("id", "run_id", "payload", "digest", "dataset_fingerprint", "fold_count", "model_id", "created_at")


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    table_names = inspector.get_table_names()
    if _TABLE not in table_names:
        op.create_table(
            _TABLE,
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("run_id", sa.String(length=64), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("digest", sa.String(length=64), nullable=False),
            sa.Column("dataset_fingerprint", sa.String(length=128), nullable=True),
            sa.Column("fold_count", sa.Integer(), nullable=False),
            sa.Column("model_id", sa.String(length=64), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_ml_walk_forward_runs_run_id", _TABLE, ["run_id"], unique=True)
        op.create_index("ix_ml_walk_forward_runs_model_id", _TABLE, ["model_id"], unique=False)
    if "ml_models" in table_names:
        columns = {column["name"] for column in inspector.get_columns("ml_models")}
        if "artifact_sha256" not in columns:
            op.add_column(
                "ml_models",
                sa.Column("artifact_sha256", sa.String(length=64), nullable=True),
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    table_names = inspector.get_table_names()
    if "ml_models" in table_names:
        columns = {column["name"] for column in inspector.get_columns("ml_models")}
        if "artifact_sha256" in columns:
            op.drop_column("ml_models", "artifact_sha256")
    if _TABLE in table_names:
        op.drop_table(_TABLE)
