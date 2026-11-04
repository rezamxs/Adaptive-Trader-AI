"""Items 35 + 32: blank documents are a terminal state (no infinite re-parse)
and the retriever caches per-chunk normalization/token counts at index build.
"""
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.core.config import settings
from backend.app.db.base import Base
import backend.app.models  # noqa: F401 - register all tables
from backend.app.models.document import Document
from backend.app.services.knowledge.manager import KnowledgeManager
from backend.app.services.knowledge.retriever import KnowledgeRetriever, normalize_persian_text
from backend.app.services.knowledge import retriever as retriever_module


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Item 35: blank source = terminal completed state, no infinite recovery.
# ---------------------------------------------------------------------------

def test_blank_document_returns_existing_without_reparse(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_RAW_DIR", tmp_path)
    manager = KnowledgeManager(db_session)

    blank = tmp_path / "blank.txt"
    blank.write_text("   \n\t", encoding="utf-8")
    content_hash = manager._hash_file_chunked(blank)

    db_session.add(Document(
        document_id="doc_blank_legacy",
        file_name="blank.txt",
        file_type="txt",
        file_size_bytes=blank.stat().st_size,
        processing_status="completed",
        pages_count=1,
        raw_text="",
        content_sha256=content_hash,
        error_message=None,
    ))
    db_session.commit()

    calls = {"n": 0}
    real_parse = DocumentParserProxy.parse_file_path

    def counting_parse(file_name, path):
        calls["n"] += 1
        return real_parse(file_name, path)

    monkeypatch.setattr(
        "backend.app.services.knowledge.manager.DocumentParser.parse_file_path",
        counting_parse,
    )
    doc, rules = manager.ingest_document_path("blank.txt", blank)

    assert doc.document_id == "doc_blank_legacy", "must return the existing record"
    assert calls["n"] == 0, (
        f"blank source must be terminal: parsed {calls['n']} time(s) during ingest"
    )
    assert (doc.error_message or "").startswith("NO_EXTRACTABLE_CONTENT"), (
        "the terminal state must be explicitly recorded")


def test_nonblank_incomplete_document_still_recovers(db_session, tmp_path, monkeypatch):
    """Documents with real text but zero chunks MUST still recover (the blank
    shortcut must not swallow genuine incompleteness)."""
    monkeypatch.setattr(settings, "DATA_RAW_DIR", tmp_path)
    manager = KnowledgeManager(db_session)

    real = tmp_path / "real.txt"
    real.write_text("قانون اول: ریسک هر معامله حداکثر یک درصد باشد.", encoding="utf-8")
    content_hash = manager._hash_file_chunked(real)

    db_session.add(Document(
        document_id="doc_real_incomplete",
        file_name="real.txt",
        file_type="txt",
        file_size_bytes=real.stat().st_size,
        processing_status="completed",
        pages_count=1,
        raw_text="قانون اول: ریسک هر معامله حداکثر یک درصد باشد.",
        content_sha256=content_hash,
        error_message=None,
    ))
    db_session.commit()

    doc, rules = manager.ingest_document_path("real.txt", real)

    assert doc.document_id == "doc_real_incomplete"
    assert len(rules) >= 1, "recovery must re-extract rules for genuinely incomplete docs"


# ---------------------------------------------------------------------------
# Item 32: normalization/token-count caching in the retriever.
# ---------------------------------------------------------------------------

class DocumentParserProxy:
    """Import indirection so the test patches the name used by manager.py."""

    from backend.app.services.knowledge.parsers import DocumentParser as _DP
    parse_file_path = staticmethod(_DP.parse_file_path)


def test_retriever_normalization_is_cached_not_per_query(db_session, monkeypatch):
    """normalize_persian_text must run once per chunk at INDEX BUILD; five
    queries over the same corpus must not re-normalize the whole corpus."""
    counts = {"n": 0}
    real_normalize = normalize_persian_text

    def counting_normalize(text):
        counts["n"] += 1
        return real_normalize(text)

    monkeypatch.setattr(retriever_module, "normalize_persian_text", counting_normalize)

    from backend.app.models.knowledge_chunk import KnowledgeChunk
    for i in range(30):
        db_session.add(KnowledgeChunk(
            chunk_id=f"chunk_cache_{i:03d}",
            document_id="doc_cache",
            source_file="cache_test.txt",
            location_type="page",
            location_reference=f"Page {i + 1}",
            page_number=i + 1,
            title=f"عنوان تست {i}",
            content=f"محتوای تستی شماره {i} با اهرم و حد ضرر برای جست‌وجو.",
            key_concepts=["اهرم"],
            parameters={},
            exceptions_and_limits=[],
        ))
    db_session.commit()

    # 30 chunks x (5 explicit normalize fields + 4 tokenize calls) at build.
    expected_build_calls = 30 * 9
    retriever = KnowledgeRetriever(db_session)
    # Item 8: the index builds LAZILY on first search — trigger it here, then
    # pin that the build (not the per-query path) did the normalization.
    retriever.search("اهرم", top_k=3, min_score=0.0)
    build_calls = counts["n"]
    assert build_calls >= expected_build_calls, (
        f"index build should normalize {expected_build_calls} fields, got {build_calls}")

    retriever.search("اهرم", top_k=3, min_score=0.0)
    after_q1 = counts["n"]
    for _ in range(5):
        results = retriever.search("اهرم", top_k=3, min_score=0.0)
        assert isinstance(results, list)

    per_query = (counts["n"] - after_q1) / 5
    assert per_query <= 5, (
        f"per-query normalization must be O(query) not O(corpus): "
        f"{per_query} calls per query over a 30-chunk corpus"
    )


def test_retriever_results_stable_after_caching(db_session):
    """Correctness guard: cached scoring returns the same ranking as a fresh
    retriever build over the same corpus."""
    from backend.app.models.knowledge_chunk import KnowledgeChunk
    for i in range(10):
        db_session.add(KnowledgeChunk(
            chunk_id=f"chunk_stable_{i:03d}",
            document_id="doc_stable",
            source_file="stable.txt",
            location_type="page",
            location_reference=f"Page {i + 1}",
            page_number=i + 1,
            title=f"اهرم در جلسه {i}",
            content=f"جلسه {i}: توضیح اهرم و مدیریت ریسک با مثال عملی شماره {i}.",
            key_concepts=["اهرم", "ریسک"],
            parameters={},
            exceptions_and_limits=[],
        ))
    db_session.commit()

    r1 = KnowledgeRetriever(db_session).search("اهرم", top_k=5, min_score=0.0)
    r2 = KnowledgeRetriever(db_session).search("اهرم", top_k=5, min_score=0.0)
    assert [x["chunk_id"] for x in r1] == [x["chunk_id"] for x in r2]
    assert len(r1) > 0
