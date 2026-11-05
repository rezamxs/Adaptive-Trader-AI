"""Buglist 2026-11-05 round: regression tests for items 2, 6, 7, 20.

Item 2  — retention must NEVER prune demo orders still awaiting a final
          status (PENDING / EXECUTION_UNKNOWN): they are the only record
          available to reconcile the broker outcome later.
Item 6  — the XLSX parser must refuse a workbook whose OOXML members expand
          beyond the per-member decompression cap (same bomb guard as the
          DOCX/ZIP parsers).
Item 7  — the inbox must NOT count a parse-failed document as ingested: the
          manager returns a failed-status Document instead of raising, and
          the file must move to failed/ so a retry stays possible.
Item 20 — the document LIST endpoint must not carry raw_text.
"""
import io
import zipfile
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.app.models  # noqa: F401 - register all ORM tables on Base
from backend.app.core.config import settings
from backend.app.db.base import Base
from backend.app.services.knowledge.parsers import DocumentParser
from backend.app.services.knowledge.inbox import process_inbox
from backend.app.services.worker.market_worker import prune_operational_records
from backend.app.models.order import DemoOrder


@pytest.fixture
def db_session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DOCUMENT_INBOX_DIR", tmp_path)
    monkeypatch.setattr(settings, "DATA_RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(settings, "DATA_PROCESSED_DIR", tmp_path / "processed_data")
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session, tmp_path
    session.close()
    engine.dispose()


# ---------------------------------------------------------------- item 2 (retention)

def _old_order(session, status: str, days: int) -> str:
    import uuid

    order = DemoOrder(
        order_id=f"ord_{uuid.uuid4().hex[:10]}",
        signal_id=f"sig_{uuid.uuid4().hex[:10]}",
        idempotency_key=f"idem_{uuid.uuid4().hex[:16]}",
        symbol="XAUUSD",
        order_type="BUY",
        volume=0.01,
        requested_price=2000.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        status=status,
    )
    session.add(order)
    session.commit()
    # Retention compares created_at against utc_now_naive() - window.
    from backend.app.models.order import DemoOrder as M

    session.query(M).filter(M.order_id == order.order_id).update(
        {"created_at": order.created_at - timedelta(days=days)}
    )
    session.commit()
    return order.order_id


def test_retention_prunes_final_orders_but_keeps_unresolved_ones(db_session):
    """Buglist item 2: an old order that never reached a final status must
    survive the 30-day window so broker reconciliation stays possible."""
    session, _ = db_session
    filled = _old_order(session, "FILLED", days=45)
    pending = _old_order(session, "PENDING", days=45)
    unknown = _old_order(session, "EXECUTION_UNKNOWN", days=45)

    deleted = prune_operational_records(db=session, demo_order_days=30)

    assert deleted["demo_orders"] == 1, "only the final-status order is aged out"
    remaining_ids = {o.order_id for o in session.query(DemoOrder).all()}
    assert filled not in remaining_ids
    assert pending in remaining_ids, "PENDING orders must never be pruned"
    assert unknown in remaining_ids, "EXECUTION_UNKNOWN orders must never be pruned"


def test_retention_prunes_unresolved_orders_once_they_reach_a_final_status(db_session):
    session, _ = db_session
    resolved_late = _old_order(session, "EXECUTION_UNKNOWN", days=45)
    from backend.app.models.order import DemoOrder as M

    session.query(M).filter(M.order_id == resolved_late).update({"status": "CLOSED"})
    session.commit()

    deleted = prune_operational_records(db=session, demo_order_days=30)
    assert deleted["demo_orders"] == 1
    assert session.query(DemoOrder).count() == 0


# ---------------------------------------------------------------- item 6 (xlsx bomb)

def _xlsx_with_bomb_member() -> bytes:
    """A tiny compressed archive whose xl/sharedStrings.xml member expands
    beyond the 64MB per-member decompression cap."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("xl/sharedStrings.xml", b"0" * (64 * 1024 * 1024 + 1))
    return buf.getvalue()


def test_xlsx_parser_refuses_decompression_bomb(db_session):
    """Buglist item 6: openpyxl decompresses OOXML members fully into memory;
    a crafted workbook must be rejected BEFORE the library touches it."""
    bomb = _xlsx_with_bomb_member()
    assert len(bomb) < 1024 * 1024, "the crafted archive itself stays tiny"
    with pytest.raises(ValueError, match="per-member limit"):
        DocumentParser.parse_excel(bomb)


def test_xlsx_parser_still_reads_legitimate_workbooks(db_session):
    import openpyxl

    buf = io.BytesIO()
    wb = openpyxl.Workbook()
    wb.active.title = "Rules"
    wb.active.append(["name", "condition"])
    wb.active.append(["rsi_buy", "RSI < 30"])
    notes = wb.create_sheet("Notes")
    notes.append(["note one", "note two"])
    wb.save(buf)

    full_text, sheets, chunks = DocumentParser.parse_excel(buf.getvalue())
    assert sheets == 2
    assert "rsi_buy" in full_text
    assert any("Rules" in str(c.get("sheet")) for c in chunks)


def test_xlsx_path_parser_bounds_members_too(db_session, tmp_path):
    bomb_path = tmp_path / "bomb.xlsx"
    bomb_path.write_bytes(_xlsx_with_bomb_member())
    with pytest.raises(ValueError, match="per-member limit"):
        DocumentParser.parse_excel_path(bomb_path)


# ---------------------------------------------------------------- item 7 (inbox honesty)

def test_inbox_counts_parse_failure_as_failed_not_ingested(db_session):
    """Buglist item 7: a file whose parse fails is reported by the manager as
    a failed-status Document (no exception). The inbox must count it as
    failed and move it to failed/ — never as ingested into processed/."""
    session, inbox = db_session
    # A .docx whose body is not a ZIP at all: the parser raises, the manager
    # converts that into a FAILED document record.
    (inbox / "corrupt.docx").write_bytes(b"this is not a zip container")

    summary = process_inbox(session, inbox)

    assert summary["failed"] == 1
    assert summary["ingested"] == 0
    entry = summary["documents"][0]
    assert entry["ok"] is False
    assert entry["processing_status"] == "failed"
    assert entry.get("error")
    assert not (inbox / "corrupt.docx").exists()
    assert (inbox / "failed" / "corrupt.docx").exists(), (
        "a failed parse must stay available for a later retry")


def test_inbox_still_ingests_supported_files(db_session):
    session, inbox = db_session
    (inbox / "ok_notes.txt").write_text(
        "Rule: Buy when RSI crosses below 30 on XAUUSD M15.\n", encoding="utf-8")
    summary = process_inbox(session, inbox)
    assert summary["ingested"] == 1 and summary["failed"] == 0
    assert (inbox / "processed" / "ok_notes.txt").exists()


# ---------------------------------------------------------------- item 20 (list without raw_text)

def test_document_list_excludes_raw_text(db_session):
    """Buglist item 20: the list response must not carry raw_text (up to 500
    full source texts per page), and the ORM must not even fetch the column."""
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.db.session import get_db
    from backend.app.models.document import Document

    session, _ = db_session
    big_text = "UNCHANGED_RAW_TEXT_MARKER " * 1000
    session.add(Document(
        document_id="doc_listtest",
        file_name="big.txt",
        file_type="txt",
        file_size_bytes=len(big_text),
        processing_status="completed",
        pages_count=1,
        raw_text=big_text,
        content_sha256="0" * 64,
    ))
    session.commit()

    def _override_db():
        yield session

    app.dependency_overrides[get_db] = _override_db
    try:
        client = TestClient(app)
        resp = client.get(
            "/api/v1/documents/",
            headers={"X-API-TOKEN": settings.API_TOKEN},
        )
        assert resp.status_code == 200, resp.text[:300]
        items = resp.json()
        assert items and items[0]["file_name"] == "big.txt"
        assert all("raw_text" not in item for item in items), (
            "the listing must not serialize the full source text")
        assert "UNCHANGED_RAW_TEXT_MARKER" not in resp.text
    finally:
        app.dependency_overrides.pop(get_db, None)
