"""Hybrid RAG + knowledge classification regression tests (mission 2026-11-07).

Semantic tests are gated on the LOCAL fastembed model cache: CI runners must
not download ~700MB per run. When the cache is absent the tests skip
themselves — the hybrid path degrades to BM25-only in production, which is
covered by the fallback tests that do NOT need the model.
"""
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.app.models  # noqa: F401
from backend.app.core.config import settings
from backend.app.db.base import Base


def _model_cached() -> bool:
    cache = Path(tempfile.gettempdir()) / "fastembed_cache"
    return any("paraphrase-multilingual" in d.name for d in cache.glob("models--*")) if cache.exists() else False


SEMANTIC_READY = _model_cached()


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session, tmp_dir_factory()
    session.close()
    engine.dispose()


def tmp_dir_factory():
    import tempfile

    return Path(tempfile.mkdtemp())


@pytest.fixture
def rag_env(db_session, monkeypatch):
    session, tmp = db_session
    monkeypatch.setattr(settings, "DATA_RAW_DIR", tmp / "raw")
    monkeypatch.setattr(settings, "DATA_PROCESSED_DIR", tmp / "processed")
    monkeypatch.setattr(settings, "VECTOR_STORE_DIR", tmp / "vector_store")
    yield session, tmp


# ------------------------------------------------------------- classification

def test_glossary_row_is_educational_not_a_rule():
    """THE 102-rule flood root cause: a glossary table row mentioning ورود
    inside a MEANING must be EDUCATIONAL, never a rule candidate."""
    from backend.app.services.knowledge.classifier import KnowledgeClass, classify_chunk

    klass, reasons = classify_chunk(
        "Row 10: 9 | Leverage | اهرم معاملاتی | ورود با حجمی بزرگ‌تر از سرمایه با نسبت 1:500")
    assert klass is KnowledgeClass.EDUCATIONAL_TERM, reasons


def test_explicit_actionable_rule_is_classified_explicit():
    from backend.app.services.knowledge.classifier import KnowledgeClass, classify_chunk

    klass, reasons = classify_chunk(
        "اگر RSI زیر 30 بود وارد معامله خرید شو و حد ضرر را زیر سوئینگ قرار بده")
    assert klass is KnowledgeClass.EXPLICIT_RULE, reasons


def test_subjective_rule_intent_is_ambiguous():
    from backend.app.services.knowledge.classifier import KnowledgeClass, classify_chunk

    klass, _ = classify_chunk("استراتژی: ورود با شکست مناسب و روند قوی و کندل باکیفیت")
    assert klass is KnowledgeClass.AMBIGUOUS_RULE


def test_glossary_ingest_creates_zero_registry_rules(rag_env):
    """End-to-end: the glossary docx must produce 102 knowledge chunks and
    ZERO rule candidates (was 102 fabricated rules before classification)."""
    import shutil

    from backend.app.services.knowledge.manager import KnowledgeManager
    from backend.app.models.rule import TradingRule

    session, tmp = rag_env
    src = Path(r"G:\mash\فایل ها\اصطلاح پرکاربرد در ترید.docx")
    if not src.is_file():
        pytest.skip("educational source file not present in this environment")
    copy = tmp / src.name
    shutil.copyfile(src, copy)
    doc, rules = KnowledgeManager(session).ingest_document_path(copy.name, copy)
    db_rules = session.query(TradingRule).filter(
        TradingRule.source_document == doc.document_id).all()
    assert doc.pages_count == 102
    assert len(db_rules) == 0, "glossary rows must never become registry rules"
    assert doc.processing_status == "completed"


def test_real_rule_still_extracts():
    from backend.app.services.knowledge.extractor import RuleExtractor

    rules = RuleExtractor.extract_from_pages(
        "sanity", [{"page": 1,
                    "text": "Rule 1: Buy when RSI < 30 on XAUUSD M15 with stop loss below the swing low."}])
    assert len(rules) == 1


# ------------------------------------------------------------- semantic / hybrid

def test_hybrid_falls_back_to_bm25_without_model(rag_env, monkeypatch):
    """Graceful degradation: with fastembed unavailable, hybrid returns the
    BM25 results unchanged and clearly labels the fusion mode."""
    session, _ = rag_env
    from unittest.mock import MagicMock

    from backend.app.services.knowledge.semantic import HybridRetriever

    broken = MagicMock()
    broken.available = False
    broken.unavailable_reason = "simulated import failure"
    monkeypatch.setattr(
        "backend.app.services.knowledge.semantic.SemanticRetriever",
        lambda db: broken)
    hybrid = HybridRetriever(session)
    assert hybrid.semantic.available is False
    out = hybrid.search("قانون ورود", top_k=3)
    assert out["fusion"] == "bm25_only"
    assert out["retrievers"] == ["bm25"]


def test_dedup_and_rebuild_semantics(rag_env):
    """No duplicate indexing: re-indexing the same chunks skips all of them.
    Rebuild truly empties then re-fills (qdrant-local resurrection workaround)
    while SQL chunks stay intact."""
    if not SEMANTIC_READY:
        pytest.skip("multilingual embedding model not cached in this environment")
    session, _ = rag_env
    from backend.app.models.knowledge_chunk import KnowledgeChunk
    from backend.app.services.knowledge.semantic import SemanticRetriever

    def _chunk(i):
        return KnowledgeChunk(
            chunk_id=f"rag_dedup_{i}", document_id="doc_rag", source_file="t.pdf",
            location_type="page", location_reference=f"Page {i+1}", title=f"t{i}",
            content=f"حد ضرر معامله {i} بر اساس ATR 14 محاسبه می‌شود",
            key_concepts=[], parameters={}, exceptions_and_limits=[])

    session.add_all([_chunk(i) for i in range(5)])
    session.commit()
    sem = SemanticRetriever(session)
    assert sem.available
    first = sem.index_chunks(session.query(KnowledgeChunk).all())
    assert first["indexed"] == 5
    second = sem.index_chunks(session.query(KnowledgeChunk).all())
    assert second == {"indexed": 0, "skipped": 5}, "no duplicate indexing"
    rebuilt = sem.rebuild_index()
    assert rebuilt["indexed"] == 5 and sem.count() == 5
    assert session.query(KnowledgeChunk).count() == 5, "SQL chunks survive the rebuild"
    hits = sem.search("حد ضرر چگونه محاسبه می‌شود", top_k=2)
    assert hits and hits[0]["location_reference"]


def test_semantic_cites_file_and_page(rag_env):
    if not SEMANTIC_READY:
        pytest.skip("multilingual embedding model not cached in this environment")
    session, _ = rag_env
    from backend.app.models.knowledge_chunk import KnowledgeChunk
    from backend.app.services.knowledge.semantic import SemanticRetriever

    session.add(KnowledgeChunk(
        chunk_id="rag_cite_1", document_id="doc_rag2", source_file="part.pdf",
        location_type="page", location_reference="Page 41", page_number=41,
        title="مفهوم حجم", content="حجم معاملات بالا نشانه تأیید حرکت قیمت است",
        key_concepts=[], parameters={}, exceptions_and_limits=[]))
    session.commit()
    sem = SemanticRetriever(session)
    assert sem.available
    stats = sem.index_chunks(session.query(KnowledgeChunk).all())
    assert stats["indexed"] == 1
    hits = sem.search("تأیید حرکت قیمت با حجم", top_k=1)
    assert hits, "cross-lingual/fa query must hit"
    assert hits[0]["source_file"] == "part.pdf"
    assert hits[0]["page_number"] == 41
