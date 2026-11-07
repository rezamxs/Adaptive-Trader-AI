"""Unit tests for Phase O Three Defects Remediation:
1. Agent 1: Cryptographic content_sha256 deduplication, collision resilience, and legacy document support.
2. Agent 2: DOCX strict session heading regex, false positive elimination, sequential order, and STR-OF-002 paragraph provenance.
3. Agent 3: PDF multi-state page categorization (BLANK, VECTOR_ONLY, IMAGE_ONLY, TEXT_VECTOR), image dimension capture,
   anti-hallucination grounding (visual placeholders cannot produce GROUNDED_EVIDENCE_FOUND), and OCR feature flag.
"""
import io
import re
import hashlib
import json
from pathlib import Path
import importlib
import pytest
import pymupdf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.db.base import Base
from backend.app.models.document import Document
from backend.app.models.knowledge_chunk import KnowledgeChunk
from backend.app.models.rule import TradingRule
from backend.app.core.config import settings
from backend.app.services.knowledge.parsers import DocumentParser
from backend.app.services.knowledge.manager import KnowledgeManager
from backend.app.services.knowledge.retriever import KnowledgeRetriever, is_semantic_chunk


@pytest.fixture
def isolated_db():
    """Provides an isolated in-memory SQLite database for test reproducibility."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    yield session
    session.close()


# ==============================================================================
# Agent 1 Tests: Knowledge Integrity & Cryptographic Deduplication
# ==============================================================================

def test_agent1_sha256_identical_content_deduplication(isolated_db):
    """Test that identical content with identical or different filename is safely deduplicated."""
    km = KnowledgeManager(isolated_db)
    content = b"Unique educational trading content about support and resistance."

    doc1, _ = km.ingest_document("lesson1.txt", content, source_description="First upload")
    assert doc1.content_sha256 is not None
    assert len(doc1.content_sha256) == 64

    # Upload identical content with same filename
    doc2, _ = km.ingest_document("lesson1.txt", content, source_description="Duplicate upload")
    assert doc2.document_id == doc1.document_id

    # Upload identical content with different filename
    doc3, _ = km.ingest_document("renamed_lesson.txt", content, source_description="Renamed upload")
    assert doc3.document_id == doc1.document_id

    # Verify only 1 document exists in DB
    total_docs = isolated_db.query(Document).count()
    assert total_docs == 1


def test_agent1_collision_resilience_different_content_same_size(isolated_db):
    """Test that two files with the exact same size and name but different content are NOT merged."""
    km = KnowledgeManager(isolated_db)
    content_a = b"AAAA_1234"  # 9 bytes
    content_b = b"BBBB_5678"  # 9 bytes

    doc_a, _ = km.ingest_document("strategy.txt", content_a)
    doc_b, _ = km.ingest_document("strategy.txt", content_b)

    assert doc_a.document_id != doc_b.document_id
    assert doc_a.content_sha256 != doc_b.content_sha256
    assert isolated_db.query(Document).count() == 2


def test_agent1_legacy_document_backfill_and_read(isolated_db):
    """Test that legacy records with content_sha256=None are read safely and backfilled upon re-ingest."""
    legacy_content = b"Legacy course material without content_sha256."
    legacy_doc = Document(
        document_id="legacy_doc_001",
        file_name="legacy_course.txt",
        file_type="txt",
        file_size_bytes=len(legacy_content),
        source_description="Legacy upload from v1",
        processing_status="completed",
        pages_count=1,
        raw_text=legacy_content.decode("utf-8"),
        content_sha256=None,  # Pre-migration state
    )
    isolated_db.add(legacy_doc)
    isolated_db.commit()

    # Verify querying legacy document works without error
    retrieved = isolated_db.query(Document).filter(Document.document_id == "legacy_doc_001").first()
    assert retrieved is not None
    assert retrieved.content_sha256 is None

    # Re-ingesting the legacy file should backfill its content_sha256 without creating a duplicate
    km = KnowledgeManager(isolated_db)
    reingested_doc, _ = km.ingest_document("legacy_course.txt", legacy_content)
    assert reingested_doc.document_id == "legacy_doc_001"
    assert reingested_doc.content_sha256 is not None
    assert isolated_db.query(Document).count() == 1


def test_agent1_legacy_collision_different_content_same_name_and_size_no_data_loss(isolated_db):
    """Regression Test: A legacy doc (completed, content_sha256=None) must NOT merge with
    incoming content that has the same filename and size but different content.
    Proves:
      1. New content is NOT dropped/deleted (new Document record created).
      2. Old legacy document's raw_text remains untouched.
      3. New hash is NOT erroneously assigned to the legacy document.
    """
    old_content = b"OLD_DATA!"  # 9 bytes
    new_content = b"NEW_DATA!"  # 9 bytes, same size, different bytes!

    legacy_doc = Document(
        document_id="legacy_doc_safe",
        file_name="risk_rules.txt",
        file_type="txt",
        file_size_bytes=len(old_content),
        source_description="Original legacy strategy",
        processing_status="completed",
        pages_count=1,
        raw_text=old_content.decode("utf-8"),
        content_sha256=None,
    )
    isolated_db.add(legacy_doc)
    isolated_db.commit()

    km = KnowledgeManager(isolated_db)
    # Ingest new content with identical filename and identical byte size
    new_doc, _ = km.ingest_document("risk_rules.txt", new_content)

    # 1. New document must be created separately (not dropped)
    assert new_doc.document_id != "legacy_doc_safe"
    assert isolated_db.query(Document).count() == 2

    # 2. Old document's text and metadata must remain completely untouched
    refreshed_legacy = isolated_db.query(Document).filter(Document.document_id == "legacy_doc_safe").first()
    assert refreshed_legacy.raw_text == "OLD_DATA!"

    # 3. Old document's hash must NOT be falsely set to the new content's hash
    assert refreshed_legacy.content_sha256 is None

    # 4. New document has its own correct hash and text
    assert new_doc.content_sha256 == hashlib.sha256(new_content).hexdigest()
    assert "NEW_DATA!" in new_doc.raw_text


def test_agent1_migration_adds_column_and_index():
    """Verify that Alembic migration 20261015_sha256 creates both content_sha256 column
    and ix_documents_content_sha256 index on an unmigrated database, and drops them cleanly on downgrade.
    """
    from sqlalchemy import Table, Column, Integer, String, MetaData, inspect
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    mod = importlib.import_module("migrations.versions.2026_10_15_add_content_sha256_to_documents")
    upgrade = mod.upgrade
    downgrade = mod.downgrade

    test_engine = create_engine("sqlite:///:memory:")
    meta = MetaData()
    # Create legacy documents table without content_sha256
    Table(
        "documents",
        meta,
        Column("id", Integer, primary_key=True),
        Column("document_id", String(64)),
        Column("file_name", String(255)),
    )
    meta.create_all(test_engine)

    # Initial check: column and index must not exist
    insp = inspect(test_engine)
    cols_before = {c["name"] for c in insp.get_columns("documents")}
    assert "content_sha256" not in cols_before

    # Run upgrade() using Alembic Operations context
    with test_engine.connect() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            upgrade()
        conn.commit()

    # Verify column and index are created
    insp_after = inspect(test_engine)
    cols_after = {c["name"] for c in insp_after.get_columns("documents")}
    assert "content_sha256" in cols_after

    indexes = {idx["name"] for idx in insp_after.get_indexes("documents")}
    assert "ix_documents_content_sha256" in indexes

    # Run downgrade() and verify removal
    with test_engine.connect() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            downgrade()
        conn.commit()

    insp_down = inspect(test_engine)
    cols_down = {c["name"] for c in insp_down.get_columns("documents")}
    assert "content_sha256" not in cols_down
    indexes_down = {idx["name"] for idx in insp_down.get_indexes("documents")}
    assert "ix_documents_content_sha256" not in indexes_down


def test_agent1_concurrent_race_condition_recovery(isolated_db):
    """Rigorous Race Condition Error Handling Test:
    Proves that when two concurrent workers check the database simultaneously:
      1. Initial check finds NO existing document (cannot exit early).
      2. Worker B proceeds to commit(), which raises IntegrityError (simulating unique collision).
      3. Worker B enters except IntegrityError, executes rollback(), and fetches the winning doc.
      4. Test CANNOT pass without entering the except block and executing rollback.
    NOTE: This test validates application-level error handling, NOT database-level concurrency prevention.
    """
    from sqlalchemy.exc import IntegrityError

    content = b"Simulated concurrent trading rule material 12345."
    expected_hash = hashlib.sha256(content).hexdigest()
    km = KnowledgeManager(isolated_db)

    # 1. Prove initial check finds nothing in the database
    initial_check = isolated_db.query(Document).filter(
        Document.content_sha256 == expected_hash,
        Document.processing_status == "completed"
    ).first()
    assert initial_check is None, "Database must be empty initially to prevent early return"

    events = {
        "commit_called": False,
        "rollback_called": False,
        "except_block_entered": False,
    }

    original_commit = isolated_db.commit
    original_rollback = isolated_db.rollback

    def spy_rollback():
        events["rollback_called"] = True
        original_rollback()

    def race_simulating_commit():
        events["commit_called"] = True
        # Simulate concurrent Worker A committing the winning record right before Worker B
        winner_doc = Document(
            document_id="winner_doc_first_commit",
            file_name="concurrent.txt",
            file_type="txt",
            file_size_bytes=len(content),
            processing_status="completed",
            raw_text="Simulated concurrent trading rule material 12345.",
            content_sha256=expected_hash,
        )
        original_rollback()
        isolated_db.add(winner_doc)
        original_commit()
        # Worker B's commit now fails with IntegrityError
        raise IntegrityError("UNIQUE constraint failed: documents.content_sha256", params={}, orig=Exception())

    isolated_db.commit = race_simulating_commit
    isolated_db.rollback = spy_rollback

    # Worker B calls ingest_document: cannot exit at line 52; MUST proceed to commit, fail, rollback, and recover
    returned_doc, _ = km.ingest_document("concurrent.txt", content)

    # 2. Assertions proving every single step of the race condition recovery path was executed
    assert events["commit_called"] is True, "commit() was never called; function returned early!"
    assert events["rollback_called"] is True, "rollback() was never called; except block not entered!"
    assert returned_doc.document_id == "winner_doc_first_commit", "Failed to recover winning document!"
    assert returned_doc.content_sha256 == expected_hash


# ==============================================================================
# Agent 2 Tests: DOCX Parser & Order Flow Session Heading Provenance
# ==============================================================================

def test_agent2_session_regex_false_positive_rejection():
    """Test that body text mentioning session numbers or 'جلسه' are rejected as session headings.

    Asserts against the PRODUCTION pattern (parsers.SESSION_HEADING_PATTERN),
    not a copy-pasted lookalike — a copy in the test cannot detect drift
    between the parser and its contract.
    """
    from backend.app.services.knowledge.parsers import SESSION_HEADING_PATTERN
    session_pattern = SESSION_HEADING_PATTERN

    # False positive body sentences that MUST NOT match as session titles
    false_positives = [
        "در هر جلسه حداقل ۱۵ دقیقه تمرین کنید تا مهارت کافی به دست آورید.",
        "جلسه بعدی ما درباره مدیریت سرمایه با اهرم 1:20 خواهد بود.",
        "مفهوم جلسه ۱۲ در بازار بسیار حائز اهمیت است اما این پاراگراف صرفاً توضیح است.",
        "در این جلسه با ۲۰ پیپ سود خارج شدیم و حد ضرر را حفظ کردیم.",
        "طول این جلسه ۲۵ دقیقه بود و مباحث عمیقی مطرح شد.",
    ]

    for sentence in false_positives:
        assert not (session_pattern.match(sentence) and len(sentence) <= 120), (
            f"False positive matched as heading: {sentence}"
        )

    # True positive headings that MUST match
    true_headings = [
        "جلسه ۱۱ | ساختار ولوم پروفایل",
        "جلسه 12: دلتای تجمعی و سفارشات محدود",
        "جلسه ۲۲ — اردر فلو پیشرفته و جذب",
        "جلسه 25 . چک‌لیست نهایی ورود به معامله",
        "  جلسه  18  پروفایل روزانه  ",
    ]

    for heading in true_headings:
        m = session_pattern.match(heading)
        assert m is not None, f"Valid heading failed to match: {heading}"
        assert len(heading) <= 120


def test_agent2_str_of_002_excerpt_count_and_fingerprint():
    """Pin the 2-excerpt / 3-sentence-fragment contract and detect silent text edits."""
    repo_root = Path(__file__).resolve().parents[3]
    registry_path = repo_root / "docs" / "strategy" / "rule_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rule = next(r for r in registry["rules"] if r.get("rule_id") == "STR-OF-002")
    exact_text = rule["exact_source_text"]

    # A hash mismatch forces an explicit review when exact_source_text changes.
    assert rule["exact_source_text_sha256"] == hashlib.sha256(
        exact_text.encode("utf-8")
    ).hexdigest()

    excerpts = [part.strip() for part in exact_text.split("...")]
    assert len(excerpts) == 2, f"Expected exactly 2 excerpts, got {len(excerpts)}"
    fragments = [
        fragment.strip()
        for excerpt in excerpts
        for fragment in re.split(r"(?<=[.!?])\s+", excerpt)
        if fragment.strip()
    ]
    assert len(fragments) == 3, (
        "The first excerpt is composite and must be checked as two sentence fragments; "
        "the second excerpt contributes one fragment."
    )


def test_agent2_composite_excerpt_is_checked_as_independent_fragments_in_docx():
    """Synthetic DOCX proves the first excerpt is not one continuous source span."""
    import zipfile
    from xml.sax.saxutils import escape

    repo_root = Path(__file__).resolve().parents[3]
    registry = json.loads((repo_root / "docs" / "strategy" / "rule_registry.json").read_text(encoding="utf-8"))
    rule = next(r for r in registry["rules"] if r.get("rule_id") == "STR-OF-002")
    excerpts = [part.strip() for part in rule["exact_source_text"].split("...")]
    fragments = [
        fragment.strip()
        for excerpt in excerpts
        for fragment in re.split(r"(?<=[.!?])\s+", excerpt)
        if fragment.strip()
    ]
    assert len(excerpts) == 2
    assert len(fragments) == 3

    # Model the actual source layout: fragment 1, an intervening Exhaustion
    # sentence, fragment 2, then fragment 3 later in the cited paragraph range.
    paragraphs = [f"unrelated source paragraph {i}" for i in range(1, 727)]
    paragraphs[712] = fragments[0]  # paragraph 713
    paragraphs[713] = "Exhaustion is discussed between the two absorption sentences."
    paragraphs[714] = fragments[1]  # paragraph 715
    paragraphs[723] = fragments[2]  # paragraph 724
    body = "".join(
        f'<w:p><w:r><w:t xml:space="preserve">{escape(paragraph)}</w:t></w:r></w:p>'
        for paragraph in paragraphs
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body>{body}<w:sectPr/></w:body></w:document>'
    ).encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", xml)

    full_text, _, _ = DocumentParser.parse_docx(buffer.getvalue())
    source_paragraphs = full_text.splitlines()
    assert len(source_paragraphs) == 726
    cited_text = " ".join(source_paragraphs[709:726])
    normalize = lambda value: " ".join(value.split())
    assert excerpts[0] not in full_text, (
        "The composite first excerpt must not be treated as a contiguous source substring."
    )
    for fragment in fragments:
        assert normalize(fragment) in normalize(cited_text), (
            f"Atomic source fragment missing from paragraphs 710-726: {fragment!r}"
        )


def test_agent2_real_docx_orderflow_sessions_and_str_of_002_provenance():
    """Test real Order Flow DOCX parsing: exactly 15 sequential sessions (11-25) + Intro,
    and verify Session 22 paragraph provenance matches STR-OF-002 citation (paragraphs 710-726).
    """
    repo_root = Path(__file__).resolve().parents[3]
    registry_path = repo_root / "docs" / "strategy" / "rule_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rule = next(r for r in registry["rules"] if r.get("rule_id") == "STR-OF-002")
    docx_path = repo_root / "\u0641\u0627\u06cc\u0644 \u0647\u0627" / rule["source_document"]
    try:
        file_bytes = docx_path.read_bytes()
    except FileNotFoundError:
        pytest.skip(f"Original local-only strategy source not available: {docx_path}")

    full_text, chunks_count, page_chunks = DocumentParser.parse_docx(file_bytes)

    assert chunks_count == 16, f"Expected 16 chunks (1 Intro + 15 sessions), got {chunks_count}"
    assert page_chunks[0]["location_reference"] == "Intro"
    assert page_chunks[0]["page"] == 0

    session_chunks = [c for c in page_chunks if c.get("location_type") == "session"]
    assert len(session_chunks) == 15

    # Check exact sequential ordering of sessions 11 to 25
    session_numbers = [c["page"] for c in session_chunks]
    assert session_numbers == list(range(11, 26))

    # Verify Session 22 paragraph boundaries
    session_22 = next(c for c in session_chunks if c["page"] == 22)
    params = session_22["parameters"]
    assert params["session_number"] == 22
    assert params["paragraph_start_1based"] == 710
    assert params["paragraph_end_1based"] == 766

    # Confirm STR-OF-002 reference (paragraphs 710-726) falls within Session 22 bounds.
    assert params["paragraph_start_1based"] <= 710
    assert params["paragraph_end_1based"] >= 726

    # The first registry excerpt contains two sentences with an intervening
    # Exhaustion sentence in the source. Verify the 3 atomic fragments separately.
    excerpts = [part.strip() for part in rule["exact_source_text"].split("...")]
    fragments = [
        fragment.strip()
        for excerpt in excerpts
        for fragment in re.split(r"(?<=[.!?])\s+", excerpt)
        if fragment.strip()
    ]
    source_paragraphs = full_text.splitlines()
    assert len(source_paragraphs) >= 726, "Parser paragraph ordinal contract changed"
    cited_text = " ".join(source_paragraphs[709:726])
    normalize = lambda value: " ".join(value.split())
    for fragment in fragments:
        assert normalize(fragment) in normalize(cited_text), (
            "Atomic STR-OF-002 source fragment missing from paragraphs 710-726: "
            f"{fragment!r}"
        )


# ==============================================================================
# Agent 3 Tests: PDF Multi-State Classification, Dimensions & Anti-Hallucination
# ==============================================================================

def test_agent3_pdf_four_state_categorization_and_dimensions():
    """Test that parse_pdf categorizes BLANK, VECTOR_ONLY, IMAGE_ONLY, and TEXT_VECTOR pages,
    and extracts image dimensions without heavy OCR.
    """
    doc = pymupdf.open()

    # Page 1: Blank page
    doc.new_page(width=595, height=842)

    # Page 2: Vector-only page (drawings, no text, no raster images)
    p2 = doc.new_page(width=595, height=842)
    p2.draw_rect(pymupdf.Rect(50, 50, 200, 200), color=(1, 0, 0), fill=(0, 1, 0))

    # Page 3: Image-only page (has raster image, no text)
    p3 = doc.new_page(width=595, height=842)
    # Generate a small 120x80 synthetic pixmap
    pix = pymupdf.Pixmap(pymupdf.csRGB, 120, 80, bytearray(120 * 80 * 3), False)
    img_bytes = pix.tobytes("png")
    p3.insert_image(pymupdf.Rect(100, 100, 220, 180), stream=img_bytes)

    # Page 4: Text page (has extractable text)
    p4 = doc.new_page(width=595, height=842)
    p4.insert_text(pymupdf.Point(50, 100), "Price action confirmation and entry rules")

    pdf_bytes = doc.tobytes()
    doc.close()

    _, pages_count, page_chunks = DocumentParser.parse_pdf(pdf_bytes)
    assert pages_count == 4
    assert len(page_chunks) == 4

    # Page 1: BLANK_PAGE
    assert page_chunks[0]["page_type"] == "BLANK_PAGE"
    assert page_chunks[0]["is_semantic_content"] is False
    assert "Blank Page" in page_chunks[0]["text"]

    # Page 2: VECTOR_ONLY_PAGE
    assert page_chunks[1]["page_type"] == "VECTOR_ONLY_PAGE"
    assert page_chunks[1]["is_semantic_content"] is False
    assert "Vector Graphics Only" in page_chunks[1]["text"]

    # Page 3: IMAGE_ONLY_PAGE
    assert page_chunks[2]["page_type"] == "IMAGE_ONLY_PAGE"
    assert page_chunks[2]["is_semantic_content"] is False
    assert page_chunks[2]["image_count"] == 1
    assert len(page_chunks[2]["image_dimensions"]) == 1
    assert page_chunks[2]["image_dimensions"][0]["width"] == 120
    assert page_chunks[2]["image_dimensions"][0]["height"] == 80
    assert "120x80" in page_chunks[2]["text"]

    # Page 4: TEXT_VECTOR_PAGE
    assert page_chunks[3]["page_type"] == "TEXT_VECTOR_PAGE"
    assert page_chunks[3]["is_semantic_content"] is True
    assert "Price action confirmation" in page_chunks[3]["text"]


def test_agent3_visual_placeholder_cannot_produce_grounded_evidence(isolated_db):
    """Test that a non-semantic visual placeholder chunk can NEVER by itself produce GROUNDED_EVIDENCE_FOUND."""
    # Insert a visual placeholder chunk and a genuine semantic chunk
    placeholder_chunk = KnowledgeChunk(
        chunk_id="chunk_visual_001",
        document_id="doc_pa_001",
        source_file="_پرایس اکشن.pdf",
        location_type="page",
        location_reference="Page 1",
        page_number=1,
        title="Page 1",
        content="[Page 1: Visual / Image Only - No Extractable Text (1 images [2480x3425])]",
        key_concepts=[],
        parameters={"page_type": "IMAGE_ONLY_PAGE", "is_semantic_content": False},
        exceptions_and_limits=[],
    )
    isolated_db.add(placeholder_chunk)
    isolated_db.commit()

    retriever = KnowledgeRetriever(isolated_db)

    # Query specifically with terms from the placeholder
    res = retriever.query_with_grounding("Visual Image Only")

    # Anti-hallucination assertion: Must NOT return GROUNDED_EVIDENCE_FOUND
    assert res["status"] == "NOT_FOUND"
    assert res["evidence_found"] is False
    assert len(res["results"]) == 0
    assert "هیچ سند یا شواهدی" in res["message"]


def test_agent3_ocr_feature_flag_disabled_by_default():
    """Verify that OCR_ENABLED is False by default to prevent heavy OCR dependencies."""
    assert hasattr(settings, "OCR_ENABLED")
    assert settings.OCR_ENABLED is False


# ==============================================================================
# Incomplete Document Ingestion & Chunk Recovery Regression Tests
# ==============================================================================

def test_incomplete_ingestion_failure_reproduction_and_recovery(isolated_db):
    """Regression Test: Failure during chunk persistence must mark document 'failed'.
    Subsequent retry with same content must recover missing chunks and mark 'completed',
    without creating duplicate Document, KnowledgeChunk, or TradingRule rows.
    """
    km = KnowledgeManager(isolated_db)
    # 2026-11-07 classification contract: vague rule-intent ("when price
    # rejects support level" — no quantified condition) is AMBIGUOUS and
    # stays OUT of the registry, so this recovery test uses an explicitly
    # documented, quantified rule sentence.
    content = b"Rule 1: Buy when RSI < 30 on XAUUSD M15 with stop loss below the swing low."

    # 1. Simulate failure during chunk persistence
    original_commit = isolated_db.commit
    call_count = [0]

    def failing_commit():
        call_count[0] += 1
        # Commit 1: Document record creation (status='processing')
        if call_count[0] == 1:
            return original_commit()
        # Commit 2: KnowledgeChunk persistence - simulate DB crash / constraint error
        elif call_count[0] == 2:
            isolated_db.rollback()
            raise RuntimeError("Simulated failure during chunk persistence")
        return original_commit()

    isolated_db.commit = failing_commit

    with pytest.raises(RuntimeError, match="Simulated failure during chunk persistence"):
        km.ingest_document("gold_strategy.txt", content)

    # Restore commit
    isolated_db.commit = original_commit

    # Verify state after failure: Document is marked 'failed' and has 0 chunks
    failed_doc = isolated_db.query(Document).first()
    assert failed_doc is not None
    assert failed_doc.processing_status == "failed"
    assert "Chunk/rule persistence failed" in (failed_doc.error_message or "")
    chunks_after_failure = isolated_db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == failed_doc.document_id).all()
    assert len(chunks_after_failure) == 0, "No chunks should be present after rollback"

    # 2. Retry ingestion: Must detect incomplete state, repair chunks, and mark 'completed'
    recovered_doc, recovered_rules = km.ingest_document("gold_strategy.txt", content)
    assert recovered_doc.document_id == failed_doc.document_id, "Must reuse document_id without creating duplicate Document row"
    assert recovered_doc.processing_status == "completed"
    assert recovered_doc.error_message is None

    chunks_after_retry = isolated_db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == recovered_doc.document_id).all()
    assert len(chunks_after_retry) == 1, "Chunk must be successfully created and persisted on retry"
    assert "Rule 1: Buy when RSI < 30" in chunks_after_retry[0].content

    # Total counts in DB: exactly 1 document, 1 chunk
    assert isolated_db.query(Document).count() == 1
    assert isolated_db.query(KnowledgeChunk).count() == 1

    # 3. Duplicate ingestion after completion: must be idempotent and create NO duplicates
    dup_doc, dup_rules = km.ingest_document("gold_strategy.txt", content)
    assert dup_doc.document_id == recovered_doc.document_id
    assert isolated_db.query(Document).count() == 1
    assert isolated_db.query(KnowledgeChunk).count() == 1


def test_incomplete_ingestion_preexisting_completed_doc_without_chunks_is_repaired(isolated_db):
    """Regression Test: If a document has processing_status='completed' but chunks count == 0
    (legacy corrupted state), re-ingestion must NOT silently return 0 chunks.
    It must detect the missing chunks, re-populate them, and return a genuinely complete document.
    """
    content = b"Volume profile point of control acceptance criteria."
    content_hash = hashlib.sha256(content).hexdigest()

    # Pre-populate corrupted document in DB (completed, but 0 chunks)
    corrupted_doc = Document(
        document_id="doc_corrupted_001",
        file_name="vp_criteria.txt",
        file_type="txt",
        file_size_bytes=len(content),
        source_description="Corrupted completed document",
        processing_status="completed",
        pages_count=1,
        raw_text=content.decode("utf-8"),
        content_sha256=content_hash,
    )
    isolated_db.add(corrupted_doc)
    isolated_db.commit()

    # Verify initial corrupted state
    assert isolated_db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == "doc_corrupted_001").count() == 0

    km = KnowledgeManager(isolated_db)
    repaired_doc, _ = km.ingest_document("vp_criteria.txt", content)

    assert repaired_doc.document_id == "doc_corrupted_001"
    assert repaired_doc.processing_status == "completed"

    chunks = isolated_db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == "doc_corrupted_001").all()
    assert len(chunks) == 1, "Must repair missing chunks even if status was already 'completed'"
    assert "Volume profile point of control" in chunks[0].content
    assert isolated_db.query(Document).count() == 1


def test_incomplete_ingestion_repeated_retries_after_multiple_failures(isolated_db):
    """Test repeated retries across multiple failures:
    Even if ingestion fails twice consecutively, the database must not accumulate duplicate
    Document rows, and the third successful attempt must leave exactly 1 completed document.
    """
    km = KnowledgeManager(isolated_db)
    content = b"Resilience test content across multiple consecutive crashes."

    original_commit = isolated_db.commit

    # Failure 1
    call_count = [0]
    def fail_first():
        call_count[0] += 1
        if call_count[0] == 2:
            isolated_db.rollback()
            raise RuntimeError("Crash 1")
        return original_commit()

    isolated_db.commit = fail_first
    with pytest.raises(RuntimeError, match="Crash 1"):
        km.ingest_document("retry_test.txt", content)

    isolated_db.commit = original_commit
    assert isolated_db.query(Document).count() == 1
    assert isolated_db.query(Document).first().processing_status == "failed"
    assert isolated_db.query(KnowledgeChunk).count() == 0

    # Failure 2
    call_count = [0]
    def fail_second():
        call_count[0] += 1
        if call_count[0] == 2:
            isolated_db.rollback()
            raise RuntimeError("Crash 2")
        return original_commit()

    isolated_db.commit = fail_second
    with pytest.raises(RuntimeError, match="Crash 2"):
        km.ingest_document("retry_test.txt", content)

    isolated_db.commit = original_commit
    assert isolated_db.query(Document).count() == 1, "Must NOT create second Document row on repeated failure"
    assert isolated_db.query(KnowledgeChunk).count() == 0

    # Attempt 3: Success
    final_doc, _ = km.ingest_document("retry_test.txt", content)
    assert final_doc.processing_status == "completed"
    assert isolated_db.query(Document).count() == 1
    assert isolated_db.query(KnowledgeChunk).count() == 1


def test_incomplete_ingestion_rule_stage_failure_and_recovery(isolated_db):
    """Regression Test: failure at the RULE-persistence commit (stage 3, after
    chunks were already committed) must still leave the document 'failed' with
    NO partial rules, and the retry must clean the earlier chunk batch and
    rebuild exactly once — no duplicate chunks, no duplicate rules, and never
    a 'completed' state without both outputs persisted."""
    km = KnowledgeManager(isolated_db)
    # 2026-11-07 classification contract: vague rule-intent ("when price
    # rejects support level" — no quantified condition) is AMBIGUOUS and
    # stays OUT of the registry, so this recovery test uses an explicitly
    # documented, quantified rule sentence.
    content = b"Rule 1: Buy when RSI < 30 on XAUUSD M15 with stop loss below the swing low."

    original_commit = isolated_db.commit
    call_count = [0]

    def failing_rules_commit():
        call_count[0] += 1
        # 1: document creation, 2: chunk persistence, 3: rule persistence.
        if call_count[0] in (1, 2):
            return original_commit()
        if call_count[0] == 3:
            isolated_db.rollback()
            raise RuntimeError("Simulated failure during rule persistence")
        return original_commit()

    isolated_db.commit = failing_rules_commit
    with pytest.raises(RuntimeError, match="Simulated failure during rule persistence"):
        km.ingest_document("gold_strategy.txt", content)

    isolated_db.commit = original_commit

    failed_doc = isolated_db.query(Document).first()
    assert failed_doc is not None
    assert failed_doc.processing_status == "failed"
    assert "Chunk/rule persistence failed" in (failed_doc.error_message or "")
    # Rules were rolled back with the failed stage-3 commit.
    assert isolated_db.query(TradingRule).filter(
        TradingRule.source_document == failed_doc.document_id).count() == 0

    # Retry: recovery deletes partial chunks, repopulates, and completes.
    recovered_doc, recovered_rules = km.ingest_document("gold_strategy.txt", content)
    assert recovered_doc.document_id == failed_doc.document_id
    assert recovered_doc.processing_status == "completed"

    assert isolated_db.query(Document).count() == 1
    assert isolated_db.query(KnowledgeChunk).filter(
        KnowledgeChunk.document_id == recovered_doc.document_id).count() == 1
    rules_after = isolated_db.query(TradingRule).filter(
        TradingRule.source_document == recovered_doc.document_id).all()
    assert len(rules_after) == len(recovered_rules)

    # A second identical ingestion stays idempotent.
    dup_doc, _ = km.ingest_document("gold_strategy.txt", content)
    assert dup_doc.document_id == recovered_doc.document_id
    assert isolated_db.query(KnowledgeChunk).filter(
        KnowledgeChunk.document_id == recovered_doc.document_id).count() == 1
    assert isolated_db.query(TradingRule).filter(
        TradingRule.source_document == recovered_doc.document_id).count() == len(rules_after)


def test_incomplete_ingestion_status_commit_failure_and_recovery(isolated_db):
    """Failure Injection at persistence boundary 4 (the 'completed' status
    commit): chunks and rules are already persisted, so the failure must NOT
    strand the document completed-without-verification — it must end in a
    clearly recoverable non-completed state whose retry repairs exactly once,
    with no duplicate rows and no fabricated completion."""
    km = KnowledgeManager(isolated_db)
    # 2026-11-07 classification contract: vague rule-intent ("when price
    # rejects support level" — no quantified condition) is AMBIGUOUS and
    # stays OUT of the registry, so this recovery test uses an explicitly
    # documented, quantified rule sentence.
    content = b"Rule 1: Buy when RSI < 30 on XAUUSD M15 with stop loss below the swing low."

    original_commit = isolated_db.commit
    call_count = [0]

    def failing_status_commit():
        call_count[0] += 1
        # 1: document creation, 2: chunks, 3: rules, 4: completed status.
        if call_count[0] in (1, 2, 3):
            return original_commit()
        if call_count[0] == 4:
            isolated_db.rollback()
            raise RuntimeError("Simulated failure during completion commit")
        return original_commit()

    isolated_db.commit = failing_status_commit
    with pytest.raises(RuntimeError, match="Simulated failure during completion commit"):
        km.ingest_document("gold_strategy.txt", content)

    isolated_db.commit = original_commit

    failed_doc = isolated_db.query(Document).first()
    assert failed_doc is not None
    assert failed_doc.processing_status == "failed", (
        "a failed completion commit must leave a RECOVERABLE state, not completed")
    # Chunks and rules survive their own committed boundaries — the retry
    # recovery path must clean and rebuild them exactly once.
    doc_id = failed_doc.document_id
    assert isolated_db.query(KnowledgeChunk).filter(
        KnowledgeChunk.document_id == doc_id).count() == 1

    recovered_doc, recovered_rules = km.ingest_document("gold_strategy.txt", content)
    assert recovered_doc.document_id == doc_id
    assert recovered_doc.processing_status == "completed"
    assert isolated_db.query(Document).count() == 1
    assert isolated_db.query(KnowledgeChunk).filter(
        KnowledgeChunk.document_id == doc_id).count() == 1, "no duplicate chunks after retry"
    assert isolated_db.query(TradingRule).filter(
        TradingRule.source_document == doc_id).count() == len(recovered_rules)


def test_incomplete_ingestion_audit_commit_failure_keeps_document_complete(isolated_db):
    """Failure Injection at persistence boundary 5 (the audit-log commit,
    after the document was already committed 'completed'): the ingestion
    raises, but the document itself is genuinely complete — the retry must
    return the EXISTING completed document without recreating any chunk,
    rule, or document row (post-completion audit loss must not corrupt
    dedup)."""
    km = KnowledgeManager(isolated_db)
    # 2026-11-07 classification contract: vague rule-intent ("when price
    # rejects support level" — no quantified condition) is AMBIGUOUS and
    # stays OUT of the registry, so this recovery test uses an explicitly
    # documented, quantified rule sentence.
    content = b"Rule 1: Buy when RSI < 30 on XAUUSD M15 with stop loss below the swing low."

    original_commit = isolated_db.commit
    call_count = [0]

    def failing_audit_commit():
        call_count[0] += 1
        # 1: document, 2: chunks, 3: rules, 4: completed status, 5: audit.
        if call_count[0] in (1, 2, 3, 4):
            return original_commit()
        if call_count[0] == 5:
            isolated_db.rollback()
            raise RuntimeError("Simulated failure during audit commit")
        return original_commit()

    isolated_db.commit = failing_audit_commit
    with pytest.raises(RuntimeError, match="Simulated failure during audit commit"):
        km.ingest_document("gold_strategy.txt", content)

    isolated_db.commit = original_commit

    doc = isolated_db.query(Document).first()
    assert doc is not None
    assert doc.processing_status == "completed", (
        "the completed state was committed at boundary 4 and must survive")
    doc_id = doc.document_id
    assert isolated_db.query(KnowledgeChunk).filter(
        KnowledgeChunk.document_id == doc_id).count() == 1

    # Retry: the completed document is recognized and returned as-is.
    retry_doc, retry_rules = km.ingest_document("gold_strategy.txt", content)
    assert retry_doc.document_id == doc_id
    assert retry_doc.processing_status == "completed"
    assert isolated_db.query(Document).count() == 1
    assert isolated_db.query(KnowledgeChunk).filter(
        KnowledgeChunk.document_id == doc_id).count() == 1
    assert isolated_db.query(TradingRule).filter(
        TradingRule.source_document == doc_id).count() == len(retry_rules)


def test_recovery_preserves_human_approved_rules_and_their_identity(isolated_db):
    """Agent-B finding (2026-10-31): recovery used to delete ALL rules of the
    document and rebuild them with fresh random ids — silently de-approving
    rules the user had already approved. Regression: an APPROVED rule must
    survive a failed boundary-4 commit + retry with its exact rule_id and
    status, and the fresh extraction must not clone it as a PENDING
    duplicate."""
    km = KnowledgeManager(isolated_db)
    # 2026-11-07 classification contract: vague rule-intent ("when price
    # rejects support level" — no quantified condition) is AMBIGUOUS and
    # stays OUT of the registry, so this recovery test uses an explicitly
    # documented, quantified rule sentence.
    content = b"Rule 1: Buy when RSI < 30 on XAUUSD M15 with stop loss below the swing low."

    # Attempt 1 fails at boundary 4 (completion commit): the document ends up
    # 'failed' while its chunks (boundary 2) and PENDING rules (boundary 3)
    # are already committed.
    original_commit = isolated_db.commit
    call_count = [0]

    def failing_status_commit():
        call_count[0] += 1
        if call_count[0] in (1, 2, 3):
            return original_commit()
        if call_count[0] == 4:
            isolated_db.rollback()
            raise RuntimeError("Simulated failure during completion commit")
        return original_commit()

    isolated_db.commit = failing_status_commit
    with pytest.raises(RuntimeError, match="Simulated failure during completion commit"):
        km.ingest_document("gold_strategy.txt", content)
    isolated_db.commit = original_commit

    failed_doc = isolated_db.query(Document).first()
    assert failed_doc.processing_status == "failed"
    pending_rules = isolated_db.query(TradingRule).filter(
        TradingRule.source_document == failed_doc.document_id).all()
    assert pending_rules, "rules committed at boundary 3 must exist on the failed doc"

    # The human approves one of the extracted rules while the document is
    # still in the failed/recoverable state.
    approved = km.approve_rule(pending_rules[0].rule_id, actor="OWNER")
    assert approved.approval_status == "approved"
    approved_id = approved.rule_id

    # Retry: recovery must preserve the approved rule with its identity and
    # must not create a PENDING clone of it.
    recovered_doc, recovered_rules = km.ingest_document("gold_strategy.txt", content)
    assert recovered_doc.document_id == failed_doc.document_id
    assert recovered_doc.processing_status == "completed"

    surviving = isolated_db.query(TradingRule).filter(
        TradingRule.source_document == failed_doc.document_id).all()
    approved_survivors = [r for r in surviving if r.approval_status == "approved"]
    assert [(r.rule_id, r.approval_status) for r in approved_survivors] == [
        (approved_id, "approved")
    ], "the human-approved rule must keep its id and status through recovery"

    clones = [r for r in surviving
              if r.approval_status == "pending"
              and (r.name, r.category) == (approved.name, approved.category)]
    assert clones == [], "recovery must not create a PENDING clone of an approved rule"
    assert isolated_db.query(Document).count() == 1
