"""Unit and integration tests for grounded Knowledge Retrieval across all 6 learning sources."""
import pytest
from backend.app.db.session import SessionLocal
from backend.app.services.knowledge.manager import KnowledgeManager
from backend.app.services.knowledge.retriever import KnowledgeRetriever


@pytest.fixture
def db_session():
    db = SessionLocal()
    yield db
    db.close()


def test_retrieval_specific_detail_from_page(db_session):
    """Test 1: Specific detail retrieval from a specific page (KX Myths Page 4: 61 wins, 39 losses, -3090 USD)."""
    km = KnowledgeManager(db_session)
    res = km.query_knowledge_grounded("شصت و یک بار حق با تو بود پیپ ۳۰۹۰ دلار")

    assert res["evidence_found"] is True
    assert res["status"] == "GROUNDED_EVIDENCE_FOUND"
    top = res["results"][0]
    assert top["source_file"] == "KX_EIGHT_MYTHS.pdf"
    assert "Page 4" in top["location_reference"]
    assert "۳٬۰۹۰" in top["content"] or "۳۰۹۰" in top["content"] or "۴۸" in top["content"]


def test_retrieval_concept_across_multiple_sources(db_session):
    """Test 2: A common concept ('حد ضرر') retrieved across multiple distinct sources."""
    km = KnowledgeManager(db_session)
    results = km.retrieve_knowledge("حد ضرر", top_k=20)

    assert len(results) > 0
    distinct_sources = set(r["source_file"] for r in results)
    # Must hit at least 3 distinct educational files
    assert len(distinct_sources) >= 3
    assert any("KX_EIGHT_MYTHS" in s for s in distinct_sources)
    assert any("اصطلاح پرکاربرد" in s for s in distinct_sources)
    assert any("جزوه_بازطراحی_پروفایل" in s for s in distinct_sources)


def test_retrieval_exact_numbers_and_conditions(db_session):
    """Test 3: Exact numbers and conditions from Scalper guide (Leverage 1:500, Deposit 5000 USD)."""
    km = KnowledgeManager(db_session)
    res = km.query_knowledge_grounded("اهرم 1:500 موجودی حداقل ۵۰۰۰ دلار")

    assert res["evidence_found"] is True
    top = res["results"][0]
    assert "راهنمای_جامع" in top["source_file"]
    assert "Page 4" in top["location_reference"]
    assert "1:500" in top["content"]
    assert "۵,۰۰۰" in top["content"] or "۵۰۰۰" in top["content"]


def test_retrieval_exceptions_and_limits(db_session):
    """Test 4: Exceptions and common analysis pitfalls in Order Flow sessions."""
    km = KnowledgeManager(db_session)
    res = km.query_knowledge_grounded("خطاهای رایج والیوم پروفایل POC دلتا")

    assert res["evidence_found"] is True
    top = res["results"][0]
    assert "جزوه_بازطراحی_پروفایل" in top["source_file"]
    assert "Session" in top["location_reference"]
    assert len(top["content"]) > 100


def test_retrieval_zero_hallucination_on_absent_topic(db_session):
    """Test 5: System must refuse to hallucinate when asked about absent concepts (Ichimoku / Ethereum)."""
    km = KnowledgeManager(db_session)
    res = km.query_knowledge_grounded("استراتژی معاملاتی ابر کومو ایچیموکو در ارز دیجیتال اتریوم تنکانسن کیجونسن")

    assert res["evidence_found"] is False
    assert res["status"] == "NOT_FOUND"
    assert len(res["results"]) == 0
    assert "هیچ سند یا شواهدی" in res["message"]


def test_r1_slippage_glossary_definition_priority(db_session):
    """Phase R1 Test 1: Slippage definition query prioritizes glossary definition (Row 48) over long lectures."""
    km = KnowledgeManager(db_session)
    res = km.query_knowledge_grounded("تعریف Slippage چیست؟", top_k=5)

    assert res["evidence_found"] is True
    assert res["status"] == "GROUNDED_EVIDENCE_FOUND"
    top = res["results"][0]
    # Priority 1 must be the concise glossary entry
    assert "اصطلاح پرکاربرد" in top["source_file"]
    assert "Slippage" in top["location_reference"]
    assert "اجرای معامله با قیمت بدتر" in top["content"] or "اجرای معامله با قیمت بدتر" in top["relevant_passage"]


def test_r1_common_errors_orderflow_isolation(db_session):
    """Phase R1 Test 2: Isolates specific common errors sections from Order Flow sessions (11/12, 19, 22)."""
    km = KnowledgeManager(db_session)
    res = km.query_knowledge_grounded("خطاهای رایج Volume Profile، POC و Delta چیست؟", top_k=6)

    assert res["evidence_found"] is True
    assert len(res["results"]) >= 3

    locs = [r["location_reference"] for r in res["results"]]
    # Sessions covering Volume Profile, POC and Delta must appear in top matches
    assert any("Session 11" in loc or "Session 12" in loc for loc in locs)
    assert any("Session 19" in loc for loc in locs)
    assert any("Session 22" in loc for loc in locs)

    # Every Order Flow match must specifically extract the 'خطاهای رایج' section
    for r in res["results"]:
        if "Session" in r["location_reference"]:
            assert r["passage_heading"] == "خطاهای رایج"
            assert "خطاهای رایج" in r["relevant_passage"]


def test_r1_absorption_vs_exhaustion_distinction(db_session):
    """Phase R1 Test 3: Distinguishes Absorption from Exhaustion and ranks Session 22 first."""
    km = KnowledgeManager(db_session)
    res = km.query_knowledge_grounded("تفاوت Absorption و Exhaustion در دلتا چیست؟", top_k=5)

    assert res["evidence_found"] is True
    top = res["results"][0]
    assert "جزوه_بازطراحی_پروفایل" in top["source_file"]
    assert "Session 22" in top["location_reference"]
    assert "Absorption" in top["title"] and "Exhaustion" in top["title"]
    # Relevant passage must contain the conceptual contrast
    assert "Absorption" in top["relevant_passage"]
    assert "Exhaustion" in top["relevant_passage"]


def test_r1_scalper_exact_parameters_priority(db_session):
    """Phase R1 Test 4: Prioritizes Page 4 of Scalper Guide with exact leverage 1:500 and 5000 USD balance."""
    km = KnowledgeManager(db_session)
    res = km.query_knowledge_grounded("اهرم و حداقل موجودی پیشنهادی ربات اسکلپر چیست؟", top_k=5)

    assert res["evidence_found"] is True
    top = res["results"][0]
    assert "راهنمای_جامع" in top["source_file"]
    assert "Page 4" in top["location_reference"]
    assert "1:500" in top["relevant_passage"]
    assert "۵,۰۰۰" in top["relevant_passage"] or "5000" in top["relevant_passage"]


def test_r1_absent_topic_strict_zero_hallucination(db_session):
    """Phase R1 Test 5: Rejects absent topic query with zero hallucination."""
    km = KnowledgeManager(db_session)
    res = km.query_knowledge_grounded("استراتژی معاملاتی پرچم سه گانه در تایم فریم ثانیه ای")

    assert res["evidence_found"] is False
    assert res["status"] == "NOT_FOUND"
    assert len(res["results"]) == 0
    assert "هیچ سند یا شواهدی" in res["message"]
