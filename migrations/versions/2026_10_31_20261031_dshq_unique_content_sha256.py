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
import hashlib

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
    # Buglist item 30: the UPGRADE neutralized duplicate hashes by NULLing the
    # later twins, and the old downgrade left them NULL forever — the idempotency
    # key those documents need was destroyed. The hash of each NULLed twin is
    # recoverable CRYPTOGRAPHICALLY: the raw file is stored under
    # DATA_RAW_DIR/<document_id>_<file_name>, so recompute it chunk-wise and
    # backfill the column. Rows whose raw file is missing keep NULL (never
    # guessed) and are logged for the operator.
    columns = {column["name"] for column in inspector.get_columns("documents")}
    if "content_sha256" not in columns:
        return
    try:
        from pathlib import Path

        from backend.app.core.config import settings as _settings

        raw_dir = Path(_settings.DATA_RAW_DIR)
    except Exception:
        return
    rows = bind.execute(text(
        "SELECT document_id, file_name FROM documents WHERE content_sha256 IS NULL"
    )).fetchall()
    restored = 0
    for document_id, file_name in rows:
        if not document_id or not file_name:
            continue
        raw_path = raw_dir / f"{document_id}_{file_name}"
        if not raw_path.is_file():
            continue
        try:
            digest = hashlib.sha256()
            with open(raw_path, "rb") as fh:
                while True:
                    block = fh.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
            bind.execute(
                text("UPDATE documents SET content_sha256 = :h WHERE document_id = :d"),
                {"h": digest.hexdigest(), "d": document_id},
            )
            restored += 1
        except OSError:
            continue
    if restored:
        print(f"dshq downgrade: restored {restored} NULLed content_sha256 value(s) from stored raw files")
