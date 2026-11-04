"""Add content_sha256 to documents table for cryptographic deduplication.

Revision ID: 20261015_sha256
Revises: 20261012_valid
Create Date: 2026-10-15
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20261015_sha256"
down_revision: Union[str, None] = "20261012_valid"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add content_sha256 column and index safely to documents table if missing."""
    bind = op.get_bind()
    inspector = inspect(bind)
    if "documents" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("documents")}
    if "content_sha256" not in columns:
        op.add_column(
            "documents",
            sa.Column(
                "content_sha256",
                sa.String(64),
                nullable=True,
            ),
        )

    # Guarantee ix_documents_content_sha256 exists across all dialects
    indexes = {idx["name"] for idx in inspector.get_indexes("documents")}
    if "ix_documents_content_sha256" not in indexes:
        op.create_index(
            "ix_documents_content_sha256",
            "documents",
            ["content_sha256"],
            unique=False,
        )


def downgrade() -> None:
    """Remove content_sha256 index and column safely if present."""
    bind = op.get_bind()
    inspector = inspect(bind)
    if "documents" not in inspector.get_table_names():
        return

    indexes = {idx["name"] for idx in inspector.get_indexes("documents")}
    if "ix_documents_content_sha256" in indexes:
        op.drop_index("ix_documents_content_sha256", table_name="documents")

    columns = {column["name"] for column in inspector.get_columns("documents")}
    if "content_sha256" in columns:
        op.drop_column("documents", "content_sha256")
