"""Add created_at index to audit_logs for retention pruning and time queries.

Retention pruning (worker cycle, item 40) deletes audit_logs older than the
configured window with `created_at < cutoff`; without an index every pruning
pass and any future time-range query scans the whole table — a real cost on
an append-only table that grows while retention is disabled.

Revision ID: 20261031_auidx
Revises: 20261031_dshq
Create Date: 2026-10-31
"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy import inspect

revision: str = "20261031_auidx"
down_revision: Union[str, None] = "20261031_dshq"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "audit_logs" not in inspector.get_table_names():
        return
    indexes = {idx["name"] for idx in inspector.get_indexes("audit_logs")}
    if "ix_audit_logs_created_at" not in indexes:
        op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "audit_logs" not in inspector.get_table_names():
        return
    indexes = {idx["name"] for idx in inspector.get_indexes("audit_logs")}
    if "ix_audit_logs_created_at" in indexes:
        op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")
