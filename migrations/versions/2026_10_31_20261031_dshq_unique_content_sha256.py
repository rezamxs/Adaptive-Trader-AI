"""Enforce content-level dedup: UNIQUE index on documents.content_sha256.

KnowledgeManager's concurrent-ingestion recovery catches IntegrityError and
re-reads the winner row — that protection only works when the database
itself refuses a second row with the same non-null content hash. A plain
non-unique index (20261015_sha256) could not provide that guarantee.

The migration is non-destructive: existing duplicate twins keep ALL their
rows — only the hash on the higher-id duplicate(s) is cleared to NULL so the
unique index can be created. NULL hashes (legacy rows) remain allowed
(unique indexes treat NULLs as distinct), and the manager's legacy-provenance
fallback already knows how to attribute NULL-hash rows on the next re-ingest.

Revision ID: 20261031_dshq
Revises: 20261031_mlwfm
Create Date: 2026-10-31
"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy import inspect, text

revision: str = "20261031_dshq"
down_revision: Union[str, None] = "20261031_mlwfm"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "documents" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("documents")}
    if "content_sha256" not in columns:
        return

    # 1. Neutralize duplicate non-null hashes WITHOUT deleting rows: the
    # lowest-id row keeps the hash (it is the deterministic winner the
    # manager reads first), later twins are set to NULL.
    bind.execute(text(
        "UPDATE documents SET content_sha256 = NULL "
        "WHERE content_sha256 IS NOT NULL AND id NOT IN ("
        "  SELECT MIN(id) FROM documents"
        "  WHERE content_sha256 IS NOT NULL"
        "  GROUP BY content_sha256)"
    ))

    # 2. Create the unique index (idempotent at the name level).
    indexes = {idx["name"] for idx in inspector.get_indexes("documents")}
    if "ux_documents_content_sha256" not in indexes:
        op.create_index(
            "ux_documents_content_sha256", "documents", ["content_sha256"], unique=True
        )
    # The old non-unique index is now redundant.
    if "ix_documents_content_sha256" in indexes:
        op.drop_index("ix_documents_content_sha256", table_name="documents")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "documents" not in inspector.get_table_names():
        return
    indexes = {idx["name"] for idx in inspector.get_indexes("documents")}
    if "ux_documents_content_sha256" in indexes:
        op.drop_index("ux_documents_content_sha256", table_name="documents")
    if "ix_documents_content_sha256" not in indexes:
        op.create_index(
            "ix_documents_content_sha256", "documents", ["content_sha256"], unique=False
        )
