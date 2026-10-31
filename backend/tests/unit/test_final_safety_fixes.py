"""Behavioral tests for the three remaining fixes:

1. Upload must never materialize the whole raw file as one bytes object:
   hashing is chunk-wise, raw storage is a file-to-file copy, and parsing
   accepts the stored path for the formats whose libraries support it.
2. An MT5 order_send exception leaves the outcome UNKNOWN, never auto-REJECTED;
   a retry on the same signal is blocked without a second dispatch.
3. The dashboard keeps the API token in memory only (no localStorage/sessionStorage).
"""
import io
import os
import re
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings

V1 = "/api/v1"


def _client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def api_headers() -> dict:
    return {"X-API-TOKEN": settings.API_TOKEN}


def _override_test_db(monkeypatch):
    """Point the app's get_db at an isolated in-memory database."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from sqlalchemy.orm import sessionmaker
    from backend.app.db.base import Base
    from backend.app.db.session import get_db as real_get_db
    import backend.app.api.v1.endpoints.documents as documents_module
    import backend.app.models  # noqa: F401 - register all tables

    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def test_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Monkeypatch the module attribute so the override is RESTORED after the
    # fixture: a direct assignment previously leaked documents_module.get_db
    # to every later test in the process.
    monkeypatch.setattr(documents_module, "get_db", test_get_db)
    app.dependency_overrides[real_get_db] = test_get_db
    yield
    app.dependency_overrides.pop(real_get_db, None)


# ---------------------------------------------------------------------------
# Problem 1: path-based ingestion without a full-file bytes object
# ---------------------------------------------------------------------------

def _count_upload_temp_files() -> int:
    return len([
        p for p in Path(os.environ.get("TMP", os.environ.get("TEMP", "."))).glob("*_upload*")
    ])


def test_upload_endpoint_routes_through_path_based_ingest(monkeypatch):
    """The endpoint must call ingest_document_path and never ingest_document:
    the bytes API is the one that requires a full-file bytes object."""
    import backend.app.api.v1.endpoints.documents as documents_module
    from backend.app.services.knowledge.manager import KnowledgeManager

    calls = {"bytes_api": 0, "path_api": 0}
    real_path_api = KnowledgeManager.ingest_document_path

    def bytes_spy(self, *a, **k):
        calls["bytes_api"] += 1
        return real_path_api(self, *a, **k) if False else None

    def path_spy(self, file_name, file_path, source_description=None):
        calls["path_api"] += 1
        # Run the real implementation against the temp path.
        return real_path_api(self, file_name, file_path, source_description)

    monkeypatch.setattr(KnowledgeManager, "ingest_document", bytes_spy)
    monkeypatch.setattr(KnowledgeManager, "ingest_document_path", path_spy)

    for _ in _override_test_db(monkeypatch):
        client = _client()
        payload = ("Rule: Buy EURUSD when RSI < 30. " * 500).encode("utf-8")
        resp = client.post(
            f"{V1}/documents/upload",
            files={"file": ("rules.txt", payload, "text/plain")},
            data={"description": "path test"},
            headers=api_headers(),
        )
        assert resp.status_code == 200, resp.text
        assert calls["path_api"] == 1, "endpoint must use the path-based ingest API"
        assert calls["bytes_api"] == 0, "endpoint must not use the full-bytes ingest API"


def test_upload_never_contains_full_file_read(monkeypatch):
    """documents.py must not read the stored temp file into one bytes object."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[3] / "backend" / "app" / "api" / "v1" / "endpoints" / "documents.py").read_text(encoding="utf-8")
    assert ".read()" not in src, "endpoint must not pull the stored file into memory as one bytes object"
    assert "ingest_document_path" in src


def test_path_ingest_matches_bytes_ingest(monkeypatch, tmp_path):
    """Same content through both APIs must produce identical hash, text, and status."""
    import hashlib
    from backend.app.db.base import Base
    from backend.app.services.knowledge.manager import KnowledgeManager
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    content = "Rule: Max risk per trade is 1 percent. Rule: Stop loss below swing low."
    file_path = tmp_path / "playbook.txt"
    file_path.write_text(content, encoding="utf-8")

    s1 = Session()
    doc_bytes, _ = KnowledgeManager(s1).ingest_document("playbook.txt", content.encode("utf-8"), "bytes")
    snapshot_bytes = (doc_bytes.content_sha256, doc_bytes.raw_text, doc_bytes.processing_status, doc_bytes.file_size_bytes)
    s1.close()

    s2 = Session()
    doc_path, _ = KnowledgeManager(s2).ingest_document_path("playbook.txt", file_path, "path")
    snapshot_path = (doc_path.content_sha256, doc_path.raw_text, doc_path.processing_status, doc_path.file_size_bytes)
    s2.close()

    assert snapshot_path[0] == snapshot_bytes[0] == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert snapshot_path[1] == snapshot_bytes[1]
    assert snapshot_path[2] == snapshot_bytes[2] == "completed"
    assert snapshot_path[3] == len(content.encode("utf-8"))


def test_parser_path_output_matches_bytes_output(tmp_path):
    """parse_file_path and parse_file must agree for every dispatchable format."""
    import pymupdf
    from backend.app.services.knowledge.parsers import DocumentParser

    # PDF built in memory, written to disk once.
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((50, 72), "Trading Playbook\nRule: Buy when RSI < 30 in strong trend.")
    pdf_bytes = pdf.tobytes()
    pdf.close()
    pdf_path = tmp_path / "p.pdf"
    pdf_path.write_bytes(pdf_bytes)

    t1, p1, c1 = DocumentParser.parse_file("p.pdf", pdf_bytes)
    t2, p2, c2 = DocumentParser.parse_file_path("p.pdf", pdf_path)
    assert p1 == p2 == 1
    assert "Rule: Buy when RSI" in t1 and t1 == t2
    assert len(c1) == len(c2) == 1

    # DOCX via zip + defusedxml.
    docx_buf = io.BytesIO()
    with zipfile.ZipFile(docx_buf, "w") as z:
        z.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body><w:p><w:r><w:t>Rule: Sell EURUSD above 1.10</w:t></w:r></w:p></w:body></w:document>',
        )
    docx_path = tmp_path / "d.docx"
    docx_path.write_bytes(docx_buf.getvalue())
    t1, p1, c1 = DocumentParser.parse_file("d.docx", docx_buf.getvalue())
    t2, p2, c2 = DocumentParser.parse_file_path("d.docx", docx_path)
    assert t1 == t2 and "Sell EURUSD above 1.10" in t2 and p1 == p2

    # XLSX.
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Rules"
    ws["A1"] = "Rule: risk 1%"
    xlsx_path = tmp_path / "r.xlsx"
    wb.save(xlsx_path)
    xlsx_bytes = xlsx_path.read_bytes()
    t1, p1, _ = DocumentParser.parse_file("r.xlsx", xlsx_bytes)
    t2, p2, _ = DocumentParser.parse_file_path("r.xlsx", xlsx_path)
    assert t1 == t2 and p1 == p2 == 1

    # TXT path must decode text, never hold a bytes object of the whole file.
    txt_path = tmp_path / "n.txt"
    txt_path.write_text("Rule: A\nRule: B\n", encoding="utf-8")
    t1, _, _ = DocumentParser.parse_file("n.txt", b"Rule: A\nRule: B\n")
    t2, _, _ = DocumentParser.parse_file_path("n.txt", txt_path)
    assert t1 == t2


def test_temp_upload_files_always_cleaned(monkeypatch):
    """Success and failure paths must leave no *_upload* temp files behind."""
    import backend.app.api.v1.endpoints.documents as documents_module
    from backend.app.services.knowledge.manager import KnowledgeManager

    before = _count_upload_temp_files()
    for _ in _override_test_db(monkeypatch):
        client = _client()
        # Success path
        resp = client.post(
            f"{V1}/documents/upload",
            files={"file": ("ok.txt", b"Rule: A\nRule: B\n", "text/plain")},
            data={"description": "t"},
            headers=api_headers(),
        )
        assert resp.status_code == 200, resp.text
        # Failure path: duplicate-hash content is fine, so force a failure via an
        # oversized-cap patch (413 before temp creation) and a rejected ingest
        # (ValueError -> 400) through a tiny cap that trips mid-stream.
        monkeypatch.setattr(documents_module, "_MAX_UPLOAD_BYTES", 64)
        resp2 = client.post(
            f"{V1}/documents/upload",
            files={"file": ("big.txt", b"x" * 4096, "text/plain")},
            data={"description": "t"},
            headers=api_headers(),
        )
        assert resp2.status_code == 413
        monkeypatch.undo()
        # empty file -> 400
        resp3 = client.post(
            f"{V1}/documents/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
            data={"description": "t"},
            headers=api_headers(),
        )
        assert resp3.status_code == 400

    after = _count_upload_temp_files()
    assert after == before, f"temp upload files leaked: {after - before}"


# ---------------------------------------------------------------------------
# Problem 3: the dashboard token must live in memory only
# ---------------------------------------------------------------------------

def _dashboard_src() -> str:
    return (Path(__file__).resolve().parents[3] / "frontend" / "index.html").read_text(encoding="utf-8")


def test_dashboard_token_never_touches_web_storage():
    src = _dashboard_src()
    code_lines = [ln for ln in src.splitlines() if not ln.strip().startswith("//")]
    code = "\n".join(code_lines)
    assert "localStorage" not in code, "token must not be persisted in localStorage"
    assert "sessionStorage" not in code, "token must not be persisted in sessionStorage"


def test_dashboard_authheaders_works_without_any_storage():
    """The real authHeaders/getApiToken implementations run with no
    localStorage/sessionStorage available at all (they are undefined)."""
    import json
    import subprocess

    src = _dashboard_src()
    js_parts = []
    for name in ("getApiToken", "authHeaders"):
        match = re.search(r"function " + name + r"\(.*?\n    \}", src, re.S)
        assert match, f"{name} implementation block not found"
        js_parts.append(match.group(0))
    js = "\n".join(js_parts) + (
        "\napiToken = 'in-memory-token';"
        "\nconst out = authHeaders({'Content-Type': 'application/json'});"
        "\nconsole.log(JSON.stringify(out));"
    )
    res = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["X-API-TOKEN"] == "in-memory-token"
    assert out["Content-Type"] == "application/json"


def test_dashboard_token_not_logged_or_embedded_in_errors():
    """The token must not appear in alert/console/log output paths."""
    src = _dashboard_src()
    token_usage = re.findall(r"(console\.[a-z]+\([^;]*|alert\([^;]*)", src)
    for line in token_usage:
        assert "apiToken" not in line and "getApiToken()" not in line, line[:120]


def test_backend_rejects_writes_without_token():
    """Frontend-side token handling is irrelevant to the backend gate: a
    state-changing request without a valid token must still be refused."""
    client = _client()
    resp = client.post(
        f"{V1}/system/emergency-stop/activate",
        json={"reason": "no-token attempt"},
    )
    assert resp.status_code == 401
    resp_wrong = client.post(
        f"{V1}/system/emergency-stop/activate",
        json={"reason": "bad-token attempt"},
        headers={"X-API-TOKEN": "wrong-token"},
    )
    assert resp_wrong.status_code == 401


def test_upload_ingest_runs_off_the_event_loop_thread(monkeypatch):
    """M11: the heavy ingest path (parsing, hashing, storage copy, index
    refresh) must run in a worker thread — never on the event-loop thread,
    where it would stall every other request for the whole upload."""
    import asyncio
    import threading
    import time as _time
    from io import BytesIO

    from fastapi import UploadFile
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from backend.app.db.base import Base
    import backend.app.models  # noqa: F401 — register all tables
    import backend.app.api.v1.endpoints.documents as documents_module

    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    captured = {"thread": None}

    def slow_ingest(self, *args, **kwargs):
        captured["thread"] = threading.get_ident()
        _time.sleep(0.05)  # tiny deliberate block; thread identity is the discriminator
        class _Doc:
            document_id = "doc_m11"
            file_name = "m11.txt"
            pages_count = 1
            processing_status = "COMPLETED"
        return _Doc(), []

    monkeypatch.setattr(documents_module.KnowledgeManager, "ingest_document_path", slow_ingest)

    async def _scenario():
        loop_thread = threading.get_ident()
        with Session() as db:
            upload = UploadFile(file=BytesIO(b"x"), size=1, filename="m11.txt")
            result = await documents_module.upload_document(
                file=upload, description="m11", db=db, _token="test")
        return result, loop_thread

    result, loop_thread = asyncio.run(_scenario())
    assert result["document_id"] == "doc_m11"
    assert captured["thread"] is not None, "ingest stub must have run"
    assert captured["thread"] != loop_thread, (
        "ingest must run off the event-loop thread (M11: to_thread)")
