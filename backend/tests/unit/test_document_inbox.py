"""Steps 3.1 + 3.2: document inbox auto-ingestion and page-type reporting.

Files dropped into data/document_inbox/ are ingested automatically through the
transactional KnowledgeManager pipeline (idempotent via content_sha256);
processed files move to processed/, failures to failed/, and image-only/
vector-only/blank pages are reported per document instead of vanishing.
"""
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.app.models  # noqa: F401 - register all ORM tables on Base
from backend.app.core.config import settings
from backend.app.db.base import Base
from backend.app.models.document import Document
from backend.app.models.knowledge_chunk import KnowledgeChunk
from backend.app.models.rule import TradingRule
from backend.app.services.knowledge.inbox import process_inbox, get_page_type_summary


@pytest.fixture
def db_session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DOCUMENT_INBOX_DIR", tmp_path)
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session, tmp_path
    session.close()
    engine.dispose()


def test_supported_file_is_ingested_and_moved_to_processed(db_session):
    session, inbox = db_session
    (inbox / "strategy_notes.txt").write_text(
        "Rule 1: Buy when RSI < 30 on XAUUSD M15\nwith stop loss below the swing low.\n",
        encoding="utf-8",
    )

    summary = process_inbox(session, inbox)

    assert summary["ingested"] == 1 and summary["failed"] == 0
    assert not (inbox / "strategy_notes.txt").exists()
    assert (inbox / "processed" / "strategy_notes.txt").exists()
    doc = session.query(Document).one()
    assert doc.processing_status == "completed"
    assert doc.source_description == "document inbox"
    assert session.query(TradingRule).count() >= 1, "extractor must produce pending rules"
    entry = summary["documents"][0]
    assert entry["document_id"] == doc.document_id
    assert "page_type_summary" in entry


def test_rescan_is_idempotent_no_duplicate_documents(db_session):
    session, inbox = db_session
    payload = "Rule: Buy on RSI < 25 for XAUUSD.\n"
    (inbox / "dup.txt").write_text(payload, encoding="utf-8")
    first = process_inbox(session, inbox)
    assert first["ingested"] == 1

    # The same content dropped again must resolve to the SAME document_id.
    (inbox / "dup_copy.txt").write_text(payload, encoding="utf-8")
    second = process_inbox(session, inbox)
    assert second["ingested"] == 1
    ids = {d["document_id"] for d in first["documents"]} | {d["document_id"] for d in second["documents"]}
    assert len(ids) == 1, "content_sha256 dedup must return the same document"
    assert session.query(Document).count() == 1


def test_unsupported_extension_is_counted_and_left_in_place(db_session):
    session, inbox = db_session
    (inbox / "notes.exe").write_bytes(b"MZ...")
    summary = process_inbox(session, inbox)
    assert summary["skipped_unsupported"] == 1
    assert summary["ingested"] == 0
    assert (inbox / "notes.exe").exists()


def test_failed_ingestion_moves_to_failed_folder(db_session, monkeypatch):
    session, inbox = db_session
    (inbox / "broken.pdf").write_bytes(b"not really a pdf")
    # Any ingestion failure (parser or persistence) must land the file in
    # failed/ with the error reported — never silently dropped.
    from backend.app.services.knowledge import inbox as inbox_module

    def _boom(self, file_name, file_path, source_description=None):
        raise ValueError("simulated parser crash")

    monkeypatch.setattr(inbox_module.KnowledgeManager, "ingest_document_path", _boom)
    summary = process_inbox(session, inbox)
    assert summary["failed"] == 1
    assert not (inbox / "broken.pdf").exists()
    assert (inbox / "failed" / "broken.pdf").exists()
    assert "simulated parser crash" in summary["documents"][0]["error"]


def test_missing_inbox_folder_is_reported_not_raised(db_session, tmp_path):
    session, _ = db_session
    summary = process_inbox(session, tmp_path / "does_not_exist")
    assert summary["ingested"] == 0
    assert "note" in summary


def test_page_type_summary_groups_no_text_pages(db_session):
    session, _ = db_session
    for page, ptype in [(1, "TEXT_PAGE"), (2, "IMAGE_ONLY_PAGE"), (3, "IMAGE_ONLY_PAGE"),
                        (4, "BLANK_PAGE"), (5, "VECTOR_ONLY_PAGE")]:
        session.add(KnowledgeChunk(
            chunk_id=f"chunk_{page}", document_id="doc_x", source_file="s.pdf",
            location_type="page", location_reference=f"Page {page}",
            page_number=page, title="t", content="text" if ptype == "TEXT_PAGE" else "",
            key_concepts=[], parameters={"page_type": ptype}, exceptions_and_limits=[],
        ))
    session.commit()
    summary = get_page_type_summary(session, "doc_x")
    assert summary["IMAGE_ONLY_PAGE"] == [2, 3], "scanned pages must be reported explicitly"
    assert summary["BLANK_PAGE"] == [4]
    assert summary["VECTOR_ONLY_PAGE"] == [5]
    assert summary["TEXT_PAGE"] == [1]
