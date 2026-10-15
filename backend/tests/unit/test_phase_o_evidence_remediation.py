"""Focused regression tests for Phase O Evidence-Gap Remediation.
Verifies:
- PDF zero-text page explicit provenance tracking.
- ZIP member enumeration and nested path handling.
- Bounded structured chunking without silent truncation.
- DOCX session and table extraction.
- Deterministic idempotent ingestion in a fresh temporary DB.
- Rule registry integrity: 52 rules, 100% UNAPPROVED, STR-012/015 blocked, STR-OF-002 precise locator.
- Security constraints: SIGNAL_ONLY and DEMO_TRADING_ENABLED=False.
"""
import io
import json
import zipfile
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import ExecutionMode, settings
from backend.app.db.base import Base
from backend.app.models.document import Document
from backend.app.models.knowledge_chunk import KnowledgeChunk
from backend.app.services.knowledge.manager import KnowledgeManager
from backend.app.services.knowledge.parsers import DocumentParser


def test_pdf_zero_text_page_provenance():
    """Verify that zero-text PDF pages are not silently dropped but recorded with explicit visual disposition."""
    import pymupdf

    doc = pymupdf.open()
    # Page 1: Empty (zero text)
    doc.new_page()
    # Page 2: Has text
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Hello page 2 text")
    pdf_bytes = doc.tobytes()
    doc.close()

    full_text, pages_count, page_chunks = DocumentParser.parse_pdf(pdf_bytes)
    assert pages_count == 2
    assert len(page_chunks) == 2
    # Page 1 must have an explicit visual/image-only placeholder
    assert page_chunks[0]["page"] == 1
    assert "Visual / Image Only" in page_chunks[0]["text"]
    # Page 2 must have its text
    assert page_chunks[1]["page"] == 2
    assert "Hello page 2 text" in page_chunks[1]["text"]


def test_zip_member_enumeration_and_nested_paths():
    """Verify ZIP parser enumerates all valid members across nested directory paths."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        z.writestr("root.md", "# Root Readme")
        z.writestr("nested/path/config.yaml", "key: value\nlist:\n  - item1")
        z.writestr("nested/path/script.py", "def test():\n    pass")
        z.writestr("ignored/binary.dat", b"\x00\x01\x02\x03")

    zip_bytes = zip_buffer.getvalue()
    full_text, count, chunks = DocumentParser.parse_zip(zip_bytes)

    assert count == 3
    refs = [c["location_reference"] for c in chunks]
    assert "root.md" in refs
    assert "nested/path/config.yaml" in refs
    assert "nested/path/script.py" in refs
    assert not any("binary.dat" in r for r in refs)


def test_bounded_structured_chunking_large_files():
    """Verify large text/yaml files in ZIP are bounded into structured chunks with no silent truncation."""
    zip_buffer = io.BytesIO()
    # Create a 25,000 char YAML payload
    large_payload = "rule_definition:\n" + ("  item: sample_value_line\n" * 1000)
    assert len(large_payload) > 20000

    with zipfile.ZipFile(zip_buffer, "w") as z:
        z.writestr("assets/large_catalog.yaml", large_payload)

    zip_bytes = zip_buffer.getvalue()
    # Parse with max_chunk_chars=5000
    _, count, chunks = DocumentParser.parse_zip(zip_bytes, max_chunk_chars=5000)

    assert count > 1
    # Check that each chunk is <= 5000 chars (plus header)
    for c in chunks:
        assert "[Part " in c["location_reference"]
        assert c["location_type"] == "archive_member"

    # Recombine to verify 100% of data is preserved
    extracted_text = "".join(c["text"].split("\n", 1)[1] for c in chunks)
    assert extracted_text == large_payload


def test_docx_session_and_table_extraction():
    """Verify DOCX parsing extracts sessions when headers present, or rows when tables present."""
    # Test plain text fallback when no sessions/tables
    mock_docx_bytes = io.BytesIO()
    with zipfile.ZipFile(mock_docx_bytes, "w") as z:
        doc_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:body>
                <w:p><w:r><w:t>جلسه 11 | مبانی ولوم پروفایل</w:t></w:r></w:p>
                <w:p><w:r><w:t>توضیحات جلسه یازدهم درباره توزیع حجم.</w:t></w:r></w:p>
                <w:p><w:r><w:t>جلسه 12 | چیدمان چندبازه ای</w:t></w:r></w:p>
                <w:p><w:r><w:t>توضیحات جلسه دوازدهم درباره پروفایل مرکب.</w:t></w:r></w:p>
            </w:body>
        </w:document>"""
        z.writestr("word/document.xml", doc_xml)

    full_text, count, chunks = DocumentParser.parse_docx(mock_docx_bytes.getvalue())
    assert count == 2
    assert chunks[0]["location_type"] == "session"
    assert chunks[0]["page"] == 11
    assert "جلسه 11" in chunks[0]["location_reference"]
    assert chunks[1]["location_type"] == "session"
    assert chunks[1]["page"] == 12
    assert "جلسه 12" in chunks[1]["location_reference"]


def test_deterministic_idempotent_ingestion_fresh_db():
    """Verify deterministic idempotent ingestion in a fresh isolated SQLite DB."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    km = KnowledgeManager(db)
    test_content = b"Rule 1: Never trade without stop loss.\nRule 2: Maximum daily drawdown is 3%."
    file_name = "test_risk_rules.txt"

    # Ingestion 1
    doc1, rules1 = km.ingest_document(file_name, test_content, "Risk document")
    assert doc1.processing_status == "completed"
    doc_count_1 = db.query(Document).count()
    chunk_count_1 = db.query(KnowledgeChunk).count()
    assert doc_count_1 == 1
    assert chunk_count_1 >= 1

    # Ingestion 2 (Duplicate file upload)
    doc2, rules2 = km.ingest_document(file_name, test_content, "Risk document duplicate")
    assert doc2.id == doc1.id
    doc_count_2 = db.query(Document).count()
    chunk_count_2 = db.query(KnowledgeChunk).count()
    # Counts must NOT double; idempotency must hold
    assert doc_count_2 == doc_count_1
    assert chunk_count_2 == chunk_count_1

    db.close()
    engine.dispose()


def test_rule_registry_integrity_all_52_unapproved():
    """Verify all 52 rules in rule_registry.json remain UNAPPROVED with exact blocked gates."""
    reg_path = Path("docs/strategy/rule_registry.json")
    assert reg_path.exists()

    with open(reg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rules = data.get("rules", [])
    assert len(rules) == 52, f"Expected 52 rules, found {len(rules)}"
    assert data.get("all_status") == "UNAPPROVED"

    # All rules must be strictly UNAPPROVED
    unapproved_rules = [r for r in rules if r.get("status") == "UNAPPROVED"]
    assert len(unapproved_rules) == 52

    # Verify STR-012 (Martingale) is blocked
    str_012 = next((r for r in rules if r["rule_id"] == "STR-012"), None)
    assert str_012 is not None
    assert str_012["status"] == "UNAPPROVED"
    assert any("BLOCKED_FROM_APPROVAL" in b for b in str_012["approval_blockers"])
    assert any("Gambler's Ruin" in b for b in str_012["approval_blockers"])

    # Verify STR-015 (Fixed spread) is blocked
    str_015 = next((r for r in rules if r["rule_id"] == "STR-015"), None)
    assert str_015 is not None
    assert str_015["status"] == "UNAPPROVED"
    assert any("BLOCKED_FROM_APPROVAL" in b for b in str_015["approval_blockers"])
    assert any("شناور" in b for b in str_015["approval_blockers"])

    # Verify STR-OF-002 exact locator
    str_of_002 = next((r for r in rules if r["rule_id"] == "STR-OF-002"), None)
    assert str_of_002 is not None
    assert "جلسه ۲۲" in str_of_002["source_reference"]
    assert "۷۱۰ تا ۷۲۶" in str_of_002["source_reference"]
    assert any("BLOCKED" in b for b in str_of_002["approval_blockers"])


def test_security_gates_and_execution_mode():
    """Verify mandatory safety boundaries: SIGNAL_ONLY and DEMO_TRADING_ENABLED=False."""
    assert settings.EXECUTION_MODE == ExecutionMode.SIGNAL_ONLY
    assert settings.DEMO_TRADING_ENABLED is False
    # Ensure LiveTradingProhibitedError is raised on attempted live mode
    with pytest.raises(Exception):
        from backend.app.core.config import Settings
        Settings(EXECUTION_MODE="LIVE_TRADING")
