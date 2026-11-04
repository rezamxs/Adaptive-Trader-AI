"""Dedup-integrity migration (20261031_dshq): UNIQUE index on documents.

content_sha256 dedup in KnowledgeManager relies on IntegrityError for
concurrent identical uploads — which only exists when the DB enforces
uniqueness. These tests pin: duplicate twins are neutralized without row
loss, the unique index blocks new duplicates at the DB level, and the old
redundant non-unique index is replaced.
"""
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def alembic_cfg(tmp_path, monkeypatch):
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    return cfg


def _write_doc(conn, doc_id: str, file_name: str, sha: str | None):
    conn.execute(text(
        "INSERT INTO documents (document_id, file_name, file_type, file_size_bytes, "
        "source_description, processing_status, pages_count, content_sha256, created_at) "
        "VALUES (:d, :f, 'txt', 10, 'test', 'completed', 1, :s, '2026-10-30 00:00:00')"),
        {"d": doc_id, "f": file_name, "s": sha},
    )


def test_duplicate_hashes_neutralized_without_row_loss(tmp_path, alembic_cfg, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'dup.db').as_posix()}"
    monkeypatch.setenv("ALEMBIC_DATABASE_URL", url)

    # Build the chain to the revision BEFORE the unique index, then plant a
    # duplicate-hash pair exactly like the pre-fix race left behind.
    command.upgrade(alembic_cfg, "20261031_mlwfm")
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX IF EXISTS ux_documents_content_sha256"))
        _write_doc(conn, "doc_dup_keep", "strategy.pdf", "a" * 64)     # lowest id: keeps hash
        _write_doc(conn, "doc_dup_twin", "strategy.pdf", "a" * 64)     # twin: must lose hash
        _write_doc(conn, "doc_solo", "other.pdf", "b" * 64)
        _write_doc(conn, "doc_legacy", "legacy.pdf", None)             # NULL stays allowed
    engine.dispose()

    command.upgrade(alembic_cfg, "head")

    engine = create_engine(url)
    try:
        inspector = inspect(engine)
        indexes = {i["name"]: i for i in inspector.get_indexes("documents")}
        # SQLite's inspector reports `unique` as 1/0, not bool — compare truthiness.
        assert bool(indexes.get("ux_documents_content_sha256", {}).get("unique")), (
            "the unique dedup index must exist after the migration")
        assert "ix_documents_content_sha256" not in indexes, (
            "the redundant non-unique index must be replaced")
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT document_id, content_sha256 FROM documents ORDER BY id")).fetchall()
        by_id = dict(rows)
        assert len(rows) == 4, "migration must not delete any row"
        assert by_id["doc_dup_keep"] == "a" * 64, "lowest-id twin keeps the hash"
        assert by_id["doc_dup_twin"] is None, "higher-id twin loses the hash, not the row"
        assert by_id["doc_solo"] == "b" * 64
        assert by_id["doc_legacy"] is None, "legacy NULL hash rows stay allowed"
    finally:
        engine.dispose()


def test_unique_index_blocks_duplicate_hash_inserts(tmp_path, alembic_cfg, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'uniq.db').as_posix()}"
    monkeypatch.setenv("ALEMBIC_DATABASE_URL", url)
    command.upgrade(alembic_cfg, "head")

    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            _write_doc(conn, "doc_first", "first.pdf", "c" * 64)
        with pytest.raises(IntegrityError):
            with engine.begin() as conn:
                _write_doc(conn, "doc_second", "second.pdf", "c" * 64)
        with engine.begin() as conn:
            # A second NULL hash row remains legal (unique treats NULLs as distinct).
            _write_doc(conn, "doc_null_1", "n1.pdf", None)
            _write_doc(conn, "doc_null_2", "n2.pdf", None)
            count = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
        assert count == 3
    finally:
        engine.dispose()


def test_index_migration_is_reversible(tmp_path, alembic_cfg, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'rev.db').as_posix()}"
    monkeypatch.setenv("ALEMBIC_DATABASE_URL", url)
    command.upgrade(alembic_cfg, "head")
    command.downgrade(alembic_cfg, "20261031_mlwfm")
    engine = create_engine(url)
    try:
        indexes = {i["name"] for i in inspect(engine).get_indexes("documents")}
        assert "ux_documents_content_sha256" not in indexes
        assert "ix_documents_content_sha256" in indexes, "downgrade restores the non-unique index"
    finally:
        engine.dispose()
    command.upgrade(alembic_cfg, "head")
    engine = create_engine(url)
    try:
        indexes = {i["name"]: i for i in inspect(engine).get_indexes("documents")}
        assert bool(indexes.get("ux_documents_content_sha256", {}).get("unique"))
    finally:
        engine.dispose()
