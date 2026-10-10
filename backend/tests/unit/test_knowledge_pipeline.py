"""Unit and integration tests for Knowledge Ingestion, Extraction, and Approval workflows."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pymupdf
from backend.app.db.base import Base
from backend.app.schemas.rule_schema import ApprovalStatus, RuleCategory
from backend.app.services.knowledge.parsers import DocumentParser
from backend.app.services.knowledge.extractor import RuleExtractor
from backend.app.services.knowledge.manager import KnowledgeManager


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def create_sample_pdf_bytes() -> bytes:
    """Creates an in-memory 2-page PDF document for testing."""
    doc = pymupdf.open()
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text((50, 72), "Trading Playbook Volume 1\nRule: Buy EURUSD when RSI < 30 in strong trend.")
    
    # Page 2
    page2 = doc.new_page()
    page2.insert_text((50, 72), "Risk Management Rules\nRule: Max risk per trade is 1% and stop loss must be at swing low.")
    
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_pdf_parsing_and_page_tracking():
    """Verify PDF parser extracts correct text and tracks page numbers."""
    pdf_bytes = create_sample_pdf_bytes()
    full_text, pages_count, chunks = DocumentParser.parse_pdf(pdf_bytes)
    
    assert pages_count == 2
    assert len(chunks) == 2
    assert chunks[0]["page"] == 1
    assert "RSI < 30" in chunks[0]["text"]
    assert chunks[1]["page"] == 2
    assert "Max risk" in chunks[1]["text"]


def test_extractor_identifies_ambiguous_terms():
    """Verify subjective keywords are flagged in ambiguities, not converted to arbitrary conditions."""
    chunks = [{
        "page": 1,
        "text": "Rule: Entry on EURUSD when RSI < 30 and there is a strong trend with a quality candle."
    }]
    rules = RuleExtractor.extract_from_pages("doc_001", chunks)
    assert len(rules) >= 1
    rule = rules[0]
    
    assert len(rule.ambiguities) >= 2
    assert any("strong trend" in amb or "روند قوی" in amb for amb in rule.ambiguities)
    assert any("quality candle" in amb or "کندل باکیفیت" in amb for amb in rule.ambiguities)
    assert rule.confidence < 0.8  # penalizes confidence for subjective statements


def test_human_approval_workflow_safety_invariant(db_session):
    """CRITICAL SAFETY TEST: Verify unapproved rules NEVER enter active strategy."""
    km = KnowledgeManager(db_session)
    pdf_bytes = create_sample_pdf_bytes()
    
    doc, rules = km.ingest_document("playbook.pdf", pdf_bytes, "Forex Strategy Document")
    assert len(rules) >= 1
    
    # Immediately after ingestion, ALL rules must be PENDING
    for r in rules:
        assert r.approval_status == ApprovalStatus.PENDING.value
        
    # Active strategy rules must be EMPTY
    active_rules = km.get_active_strategy_rules()
    assert len(active_rules) == 0
    
    # Approve first rule
    first_rule_id = rules[0].rule_id
    km.approve_rule(first_rule_id, actor="SENIOR_TRADER")
    
    # Now active strategy rules must contain exactly 1 approved rule
    active_rules_now = km.get_active_strategy_rules()
    assert len(active_rules_now) == 1
    assert active_rules_now[0].rule_id == first_rule_id
    
    # Reject second rule if present
    if len(rules) > 1:
        second_rule_id = rules[1].rule_id
        km.reject_rule(second_rule_id, "Ambiguous stop loss definition", actor="RISK_OFFICER")
        
        # Verify rejected rule is NOT in active rules
        assert second_rule_id not in [r.rule_id for r in km.get_active_strategy_rules()]
        
        # Verify rejection reason is recorded
        db_rule = db_session.query(km.get_rules()[0].__class__).filter_by(rule_id=second_rule_id).first()
        assert db_rule.approval_status == ApprovalStatus.REJECTED.value
        assert db_rule.rejection_reason == "Ambiguous stop loss definition"


def test_txt_and_csv_parsing():
    """Verify parsing text and CSV formats."""
    txt_bytes = b"Rule 1: Buy on SMA 20 crossover.\nRule 2: Sell when price touches upper channel."
    txt_full, txt_pages, txt_chunks = DocumentParser.parse_file("strategy.txt", txt_bytes)
    assert txt_pages == 1
    assert "SMA 20" in txt_full

    csv_bytes = b"Symbol,Action,RSI,Timeframe\nEURUSD,BUY,25,M15\nGBPUSD,SELL,75,H1"
    csv_full, csv_pages, csv_chunks = DocumentParser.parse_file("signals.csv", csv_bytes)
    assert "EURUSD | BUY | 25 | M15" in csv_full
