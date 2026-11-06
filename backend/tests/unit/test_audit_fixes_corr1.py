"""TDD tests for audit fixes C1 (auth), H3 (defusedxml), H5 (no mock fallback), H7 (streamed upload).

C1: every state-changing endpoint must reject requests without a valid API token.
H3: XML parsing must use defusedxml with DOCTYPE/ENTITY rejected.
H5: /market must return 503 when MT5 is down, never silently serve mock data.
H7: upload must not require reading the whole file into memory upfront.
"""
import io
import re
import zipfile

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings

V1 = "/api/v1"


def _client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def api_headers() -> dict:
    return {"X-API-TOKEN": settings.API_TOKEN}


# ---------------------------------------------------------------------------
# C1: unauthenticated state-changing endpoints must be rejected with 401/403
# ---------------------------------------------------------------------------

def test_unauthenticated_state_changing_endpoints_rejected():
    client = _client()
    protected_calls = [
        ("post", f"{V1}/system/emergency-stop/activate", {"json": {"reason": "test"}}),
        ("post", f"{V1}/system/emergency-stop/deactivate", {"json": {"notes": "test"}}),
        ("post", f"{V1}/rules/rule_fake/approve", {"json": {"actor": "USER"}}),
        ("post", f"{V1}/rules/rule_fake/reject", {"json": {"reason": "test"}}),
        ("post", f"{V1}/signals/sig_fake/execute-demo", {}),
        ("post", f"{V1}/ml/models/model_fake/promote", {"json": {"notes": "test"}}),
        ("post", f"{V1}/ml/train", {"json": {"symbol": "XAUUSD"}}),
        ("post", f"{V1}/documents/upload", {"files": {"file": ("a.txt", b"x")}}),
        # Stage-coverage completion: these two were previously untested here.
        ("post", f"{V1}/backtest/run",
         {"json": {"symbol": "XAUUSD", "timeframe": "M15", "bars_count": 100}}),
        ("post", f"{V1}/signals/generate", {"json": {"symbol": "XAUUSD", "timeframe": "M15"}}),
    ]
    for method, url, kwargs in protected_calls:
        resp = getattr(client, method)(url, **kwargs)
        assert resp.status_code in (401, 403), f"{method.upper()} {url} returned {resp.status_code}, expected 401/403"


def test_wrong_token_rejected():
    client = _client()
    resp = client.post(
        f"{V1}/system/emergency-stop/activate",
        json={"reason": "test"},
        headers={"X-API-TOKEN": "definitely-wrong-token"},
    )
    assert resp.status_code in (401, 403)


def test_backtest_and_signal_generate_accept_valid_token(monkeypatch):
    """Positive auth path for the two endpoints the C1 list previously
    missed. MT5 is forced down so the run stops at a deterministic 503 BEFORE
    any heavy work; the assertion is that a VALID token passes authentication
    (never 401/403). Isolated in-memory DB via _override_test_db."""
    monkeypatch.setattr(
        "backend.app.services.market_data.mt5_provider.MT5MarketDataProvider.connect",
        lambda self: False,
    )
    for _ in _override_test_db(monkeypatch):
        client = _client()
        resp_bt = client.post(
            f"{V1}/backtest/run",
            json={"symbol": "XAUUSD", "timeframe": "M15", "bars_count": 100},
            headers=api_headers(),
        )
        assert resp_bt.status_code not in (401, 403), (
            f"valid token must authenticate: got {resp_bt.status_code}")
        assert resp_bt.status_code == 503  # deterministic stop at MT5-down

        resp_sig = client.post(
            f"{V1}/signals/generate",
            json={"symbol": "XAUUSD", "timeframe": "M15"},
            headers=api_headers(),
        )
        assert resp_sig.status_code not in (401, 403), (
            f"valid token must authenticate: got {resp_sig.status_code}")


def test_empty_token_rejected():
    client = _client()
    resp = client.post(
        f"{V1}/system/emergency-stop/activate",
        json={"reason": "test"},
        headers={"X-API-TOKEN": ""},
    )
    assert resp.status_code in (401, 403)


def test_sensitive_reads_require_token_and_public_reads_stay_open():
    """M15 policy: every data-bearing GET (documents with raw_text, account
    status, audit logs, orders, signals, models, model comparison, rules,
    backtest runs, worker telemetry) requires the token — 127.0.0.1 binding
    alone is not protection. Buglist item 17 tightens this further: the
    /market-data/* reads are token-gated TOO (each call acquires and probes
    the shared MT5 terminal, so an unauthenticated poller could pressure the
    worker's terminal connection for free). Only /health (liveness probe,
    probe-cache-backed) stays public, and the dashboard sends authHeaders on
    every GET fetch."""
    client = _client()
    sensitive_gets = [
        f"{V1}/documents/",
        f"{V1}/system/status",
        f"{V1}/system/worker/status",
        f"{V1}/system/worker/cycles",
        f"{V1}/system/audit-logs",
        f"{V1}/rules/",
        f"{V1}/rules/conflicts",
        f"{V1}/signals/",
        f"{V1}/signals/orders",
        f"{V1}/ml/models",
        f"{V1}/ml/models/some_model/compare",
        f"{V1}/backtest/runs",
    ]
    for url in sensitive_gets:
        resp = client.get(url)
        assert resp.status_code == 401, f"GET {url} must be token-gated, got {resp.status_code}"

    # Buglist item 17: market-data reads joined the token-gated surface.
    for url in (f"{V1}/market-data/symbols", f"{V1}/market-data/quote/XAUUSD",
                f"{V1}/market-data/candles/XAUUSD"):
        resp = client.get(url)
        assert resp.status_code == 401, f"GET {url} must be token-gated (item 17), got {resp.status_code}"

    # The only public surface left is the /health liveness probe (documented
    # policy; its governance probe is TTL-cached so polling stays cheap).
    assert client.get("/health").status_code == 200

    # Valid token passes authentication on every gated read (may still 5xx on
    # MT5/worker state, but never 401).
    for url in sensitive_gets:
        resp = client.get(url, headers=api_headers())
        assert resp.status_code != 401, f"valid token must authenticate GET {url}"

    # Dashboard compatibility: every GET fetch in the real dashboard script
    # sends authHeaders like the POSTs do. A fetch with NO options object is
    # headerless by construction; a fetch with options but no `method:` key
    # is a GET and must carry authHeaders(); an explicit method:"GET" must
    # carry authHeaders() too (future-proofing the scan).
    src = _dashboard_src()
    headerless = re.findall(r"fetch\(`[^`]+`\)(?!\s*,)", src)
    assert not headerless, f"dashboard fetch without any headers: {headerless}"
    for options in re.findall(r"fetch\(`[^`]+`,\s*\{([^}]*?)\}", src, re.S):
        if "method:" not in options or re.search(r'method:\s*"GET"', options):
            assert "authHeaders()" in options, (
                f"dashboard GET fetch without authHeaders: {options[:90]}")


# ---------------------------------------------------------------------------
# H3: XML parsing must go through defusedxml (billion-laughs safe)
# ---------------------------------------------------------------------------

def test_parsers_use_defusedxml():
    import backend.app.services.knowledge.parsers as parsers_module
    source = open(parsers_module.__file__, encoding="utf-8").read()
    assert "defusedxml" in source, "parsers must import defusedxml"
    assert "ET.fromstring" not in source, "stdlib ElementTree.fromstring must be removed"


def test_docx_entity_expansion_rejected():
    """A DOCX with a billion-laughs XML must be rejected, never expanded."""
    from backend.app.services.knowledge.parsers import DocumentParser

    bomb = (
        '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">'
        '<!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
        '<!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">'
        ']><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>&lol2;</w:t></w:r></w:p></w:body></w:document>"
    ).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", bomb)

    # Contract (audit 2026-10-30, fail-loud): parse_docx no longer swallows
    # parse failures into an empty success — an empty ("", 0, []) return was
    # indistinguishable from a legitimately blank file and became a TERMINAL
    # NO_EXTRACTABLE_CONTENT record (silent data loss). defusedxml's
    # EntitiesForbidden now propagates so the ingestion manager records the
    # document as FAILED with the real cause. The no-expansion property is
    # preserved a fortiori: nothing is ever parsed or returned at all.
    from defusedxml.common import EntitiesForbidden
    with pytest.raises(EntitiesForbidden):
        DocumentParser.parse_docx(buf.getvalue())


def test_docx_normal_content_still_parses():
    """Healthy DOCX files must keep parsing after the defusedxml swap."""
    from backend.app.services.knowledge.parsers import DocumentParser

    body = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Hello trading rule</w:t></w:r></w:p></w:body></w:document>"
    ).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", body)

    text, pages, chunks = DocumentParser.parse_docx(buf.getvalue())
    assert "Hello trading rule" in text


# ---------------------------------------------------------------------------
# H5: /market must fail closed with 503 when MT5 is unavailable
# ---------------------------------------------------------------------------

def test_market_quote_returns_503_without_mt5(monkeypatch):
    monkeypatch.setattr(
        "backend.app.services.market_data.mt5_provider.MT5MarketDataProvider.connect",
        lambda self: False,
    )
    client = _client()
    resp = client.get(f"{V1}/market-data/quote/XAUUSD", headers=api_headers())
    assert resp.status_code == 503, f"expected 503 when MT5 down, got {resp.status_code}"
    assert "unavailable" in resp.text.lower()


def test_market_candles_returns_503_without_mt5(monkeypatch):
    monkeypatch.setattr(
        "backend.app.services.market_data.mt5_provider.MT5MarketDataProvider.connect",
        lambda self: False,
    )
    client = _client()
    resp = client.get(f"{V1}/market-data/candles/XAUUSD", headers=api_headers())
    assert resp.status_code == 503


def test_market_provider_factory_never_returns_mock(monkeypatch):
    import backend.app.api.v1.endpoints.market as mmod
    from fastapi import HTTPException

    monkeypatch.setattr(
        "backend.app.services.market_data.mt5_provider.MT5MarketDataProvider.connect",
        lambda self: False,
    )
    # Narrowed: the factory must raise the specific 503 HTTPException, not
    # just "any exception" (which would hide real structural regressions).
    with pytest.raises(HTTPException) as exc_info:
        mmod.get_market_provider()
    assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# H7: upload streams to temp file; endpoint rejects oversized gracefully
# ---------------------------------------------------------------------------

def _override_test_db(monkeypatch):
    """Point the app's get_db at an isolated in-memory database."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from sqlalchemy.orm import sessionmaker
    from backend.app.db.base import Base
    from backend.app.db.session import get_db as real_get_db
    import backend.app.api.v1.endpoints.documents as documents_module
    # backend.app.models re-exports every model module, which registers all
    # tables on Base metadata before create_all; omitting this leaves
    # knowledge_chunks and friends missing and request-time queries 500.
    import backend.app.models  # noqa: F401

    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,
        # StaticPool pins one shared connection, so create_all's tables stay
        # visible to every later session; without it a fresh per-connection
        # empty in-memory DB makes any request-time query 500.
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

    # Monkeypatch so the module attribute is RESTORED after the fixture: a
    # direct assignment previously leaked documents_module.get_db to every
    # later test in the process.
    monkeypatch.setattr(documents_module, "get_db", test_get_db)
    app.dependency_overrides[real_get_db] = test_get_db
    yield engine
    app.dependency_overrides.pop(real_get_db, None)


def test_upload_stream_boundary_stops_overrun(monkeypatch):
    """Even with no trustworthy pre-declared size, the streamed chunk boundary
    enforces the cap; the upload must abort with 413 once the stream overruns."""
    import backend.app.api.v1.endpoints.documents as documents_module
    for _ in _override_test_db(monkeypatch):
        monkeypatch.setattr(documents_module, "_MAX_UPLOAD_BYTES", 2048)
        client = _client()
        resp = client.post(
            f"{V1}/documents/upload",
            files={"file": ("f.bin", b"x" * 8192, "application/octet-stream")},
            data={"description": "test"},
            headers=api_headers(),
        )
        assert resp.status_code == 413


def test_upload_rejects_oversized_body(monkeypatch):
    """Actual streamed corpus exceeding the cap is cut off with 413 (uses full 250MB cap via patch)."""
    import backend.app.api.v1.endpoints.documents as documents_module
    for _ in _override_test_db(monkeypatch):
        # Patch the cap low so the streamed-overrun branch is exercisable in CI.
        monkeypatch.setattr(documents_module, "_MAX_UPLOAD_BYTES", 1024)
        client = _client()
        resp = client.post(
            f"{V1}/documents/upload",
            files={"file": ("big.bin", b"x" * 4096, "application/octet-stream")},
            data={"description": "test"},
            headers=api_headers(),
        )
        assert resp.status_code == 413


def test_upload_rejects_empty_file(monkeypatch):
    for _ in _override_test_db(monkeypatch):
        client = _client()
        resp = client.post(
            f"{V1}/documents/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
            data={"description": "test"},
            headers=api_headers(),
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# H2: dashboard must not sink server-controlled data into innerHTML unescaped
# ---------------------------------------------------------------------------

def _dashboard_src() -> str:
    """Minor 3/18: the dashboard is index.html + the external /static/app.js
    bundle (inline script and onclick attributes were removed for the strict
    CSP) — source scans must see BOTH files."""
    from pathlib import Path
    frontend = Path(__file__).resolve().parents[3] / "frontend"
    return (frontend / "index.html").read_text(encoding="utf-8") + "\n" + (
        frontend / "app.js").read_text(encoding="utf-8")


def test_h2_dashboard_defines_and_uses_html_escaping():
    import re
    src = _dashboard_src()
    assert "function escapeHtml" in src, "dashboard must define an escapeHtml helper"

    # Every interpolation of server-controlled fields inside innerHTML template
    # literals must be wrapped in esc(...); a bare ${<expr>} for any known
    # server field is the XSS sink this fix removes.
    server_fields = [
        r"r\.rule_id", r"r\.name", r"r\.category", r"r\.approval_status",
        r"m\.model_id", r"m\.algorithm", r"m\.status",
        r"s\.signal_id", r"s\.symbol", r"s\.direction", r"s\.risk_status", r"s\.execution_mode",
        r"l\.event_type", r"l\.severity", r"l\.actor", r"l\.details",
        r"scName",
    ]
    for field in server_fields:
        for match in re.finditer(r"\$\{[^{}]*" + field + r"[^{}]*\}", src):
            inner = match.group(0)[2:-1].strip()
            assert inner.startswith("esc("), (
                f"unescaped sink for {field}: ...{src[max(0, match.start() - 40):match.end()]}"
            )


def test_h2_escapehtml_function_neutralizes_vectors():
    import json
    import re
    import subprocess

    src = _dashboard_src()
    match = re.search(r"function escapeHtml\(.*?\n    \}", src, re.S)
    assert match, "escapeHtml implementation block not found"
    js = (
        match.group(0)
        + "\nconst payloads = ["
        + ",".join(json.dumps(p) for p in [
            "<img src=x onerror=alert(1)>",
            "&<>'\"`",
            "</td><script>x</script>",
        ])
        + "];\nconsole.log(JSON.stringify(payloads.map(escapeHtml)));"
    )
    res = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    escaped = json.loads(res.stdout.strip().splitlines()[-1])
    for esc in escaped:
        assert "<" not in esc and ">" not in esc, esc
        assert '"' not in esc and "'" not in esc, esc


def test_h2_ambiguity_text_is_not_double_escaped():
    """Finding 4: each ambiguity member is escaped once, joined with raw <br>,
    and the joined result must NOT be passed through esc again."""
    import re
    src = _dashboard_src()
    match = re.search(r"const ambigText = .*?\n            : \"[^\"]+\";", src, re.S)
    assert match, "ambigText template not found"
    text = match.group(0)
    assert "r.ambiguities.map(esc).join('<br>')" in text, text
    assert not re.search(r"esc\(r\.ambiguities\.map\(a\s*=>\s*esc\(", text), (
        "double escaping: members already escaped, joined result must not be escaped again"
    )


# ---------------------------------------------------------------------------
# Finding 1: the dashboard must send the API token on every state-changing call
# ---------------------------------------------------------------------------

def test_dashboard_auth_helper_adds_token_header():
    """Run the real authHeaders implementation in node with NO storage of any
    kind available and verify the X-API-TOKEN header lands on the request."""
    import json
    import re
    import subprocess

    src = _dashboard_src()
    for name in ("getApiToken", "setApiToken", "authHeaders"):
        match = re.search(r"function " + name + r"\(.*?\n    \}", src, re.S)
        assert match, f"{name} implementation block not found"
    js = (
        re.search(r"function getApiToken\(.*?\n    \}", src, re.S).group(0) + "\n"
        + re.search(r"function authHeaders\(.*?\n    \}", src, re.S).group(0) + "\n"
        "apiToken = 'sekret123';\n"
        "const out = authHeaders({'Content-Type': 'application/json'});\n"
        "const bare = authHeaders();\n"
        "console.log(JSON.stringify({out, bare}));"
    )
    res = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout.strip().splitlines()[-1])
    assert data["out"]["X-API-TOKEN"] == "sekret123"
    assert data["out"]["Content-Type"] == "application/json"
    assert data["bare"]["X-API-TOKEN"] == "sekret123"


def test_dashboard_every_post_fetch_uses_authheaders():
    """Every fetch with method POST in the dashboard must pass authHeaders(...)
    so the backend C1 token check is satisfied by the real UI flow.

    Stage-2 note: activate/deactivate were consolidated into one dynamic
    emergency-stop fetch, so the POST count dropped from 9 to 8; the security
    property asserted here (authHeaders on EVERY POST) is unchanged."""
    import re
    src = _dashboard_src()
    post_calls = re.findall(r"fetch\(`[^`]+`,\s*\{[^}]*method:\s*\"POST\"[^}]*\}", src, re.S)
    assert len(post_calls) >= 8, f"expected >=8 POST fetch calls, found {len(post_calls)}"
    assert "/system/emergency-stop/" in src, "emergency-stop toggle endpoint must remain wired"
    for call in post_calls:
        assert "authHeaders(" in call, "POST fetch without auth headers: " + call[:120]


def test_dashboard_offers_token_entry():
    src = _dashboard_src()
    assert "function setApiToken" in src
    assert "setApiToken()" in src, "dashboard must expose a button wired to setApiToken()"
    assert "let apiToken" in src, "token must live in a page-memory variable"
    assert 'prompt(' in src, "token entry must be interactive (re-entry after reload)"


def test_dashboard_never_persists_token_in_web_storage():
    """Item 5: the token lives ONLY in page memory (XSS-resilient). No
    web-storage API access may return, and no doc may claim persistence."""
    import re as _re
    from pathlib import Path
    src = _dashboard_src()
    assert not _re.search(r"localStorage\s*[.\[]", src), (
        "dashboard must not access localStorage")
    assert not _re.search(r"sessionStorage\s*[.\[]", src), (
        "dashboard must not access sessionStorage")
    readme = (Path(__file__).resolve().parents[3] / "README.md").read_text(encoding="utf-8")
    assert "تا در `localStorage` بماند" not in readme, (
        "README must not claim the API token persists in web storage")
    assert "حافظهٔ همان صفحه" in readme, (
        "README must document the page-memory-only token flow")


# ---------------------------------------------------------------------------
# M22 / item 3: distinguishable failure statuses + baseline security headers
# ---------------------------------------------------------------------------

def test_security_headers_on_every_response(isolated_app_db):
    """Item 3 + Minor 3 phase 2: hardening headers ride on every response and
    the CSP is path-scoped — STRICT (script-src 'self', no 'unsafe-inline')
    for the dashboard and the whole API; the single documented exception is
    FastAPI's own Swagger UI, whose bundled inline bootstrap needs
    'unsafe-inline' + its jsdelivr assets.

    isolated_app_db: the /system/status leg queries the database."""
    client = _client()
    strict_csp_prefixes = ("/dashboard", f"{V1}/system/status")
    for url in strict_csp_prefixes:
        resp = client.get(url, headers=api_headers())
        assert resp.status_code == 200, f"GET {url} -> {resp.status_code}"
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("Referrer-Policy") == "no-referrer"
        csp = resp.headers.get("Content-Security-Policy", "")
        assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp
        assert "script-src 'self';" in csp, f"strict script policy required on {url}"
        assert "unsafe-inline" not in csp.split("style-src", 1)[0], (
            f"script-src must not allow unsafe-inline on {url}")
        assert "https://fonts.googleapis.com" in csp, "dashboard webfont style must stay loadable"
        assert "https://fonts.gstatic.com" in csp, "dashboard webfont files must stay loadable"
        assert "unsafe-eval" not in csp, "script evaluation must stay blocked"

    # The ONLY documented exception: FastAPI's own Swagger UI bootstrap.
    docs = client.get("/docs")
    assert docs.status_code == 200
    docs_csp = docs.headers.get("Content-Security-Policy", "")
    assert "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net;" in docs_csp


def test_dashboard_has_no_inline_script_and_no_inline_handlers():
    """Minor 3/18 structural pins: the dashboard HTML ships ZERO inline
    <script> blocks and ZERO onclick= attributes — all behavior lives in the
    external same-origin /static/app.js, wired through ONE delegated
    data-action listener (strict-CSP compatible)."""
    from pathlib import Path
    frontend = Path(__file__).resolve().parents[3] / "frontend"
    html = (frontend / "index.html").read_text(encoding="utf-8")
    app_js = (frontend / "app.js").read_text(encoding="utf-8")

    import re as _re
    assert "<script>" not in html, "inline script block must not return"
    assert 'src="/static/app.js"' in html, "app.js must stay wired"
    assert not _re.search(r"onclick\s*=", html), "inline onclick must not return"
    assert not _re.search(r"onclick\s*=\s*['\"]", app_js), "generated onclick must not return"
    assert "data-action" in html and "data-action" in app_js
    assert 'document.addEventListener("click"' in app_js, "delegated listener required"
    for action in ("set-token", "toggle-emergency", "switch-tab", "upload-document",
                   "train-model", "run-backtest", "generate-signal",
                   "approve-rule", "reject-rule", "promote-model"):
        assert f'"{action}"' in app_js, f"data-action {action} must stay delegated"


def test_execution_failure_status_mapping():
    """M22: executor failures must map to distinguishable statuses instead of
    one blanket 400 (fragments pinned by test_audit_fixes_h6.py)."""
    from backend.app.api.v1.endpoints.signals import _execution_failure_status as status_for
    assert status_for("Duplicate order prevented by idempotency key K.") == 409
    assert status_for("Previous dispatch outcome is UNKNOWN; broker reconciliation "
                      "required before retry (idempotency key K).") == 409
    assert status_for("Safety Gate Blocked: emergency stop is active.") == 403
    assert status_for("Execution rejected: Signal risk status is 'REJECTED_RISK'.") == 403
    assert status_for("Execution rejected by current risk checks: "
                      "REJECTED_DAILY_DRAWDOWN.") == 403
    assert status_for("Market quote is stale or unavailable at moment of order dispatch.") == 503
    assert status_for("Execution rejected: broker symbol specifications are unavailable.") == 503
    assert status_for("Execution rejected: Calculated volume is missing or non-positive.") == 400
    assert status_for("some unexpected wording") == 400


def test_execution_failure_mapping_tracks_executor_constants():
    """M22 hardening: the classifier fragments must be the executor's OWN
    constants (identity, not copies) — rewording an executor message cannot
    silently desync the HTTP failure mapping."""
    import backend.app.api.v1.endpoints.signals as signals_module
    import backend.app.services.execution.demo_executor as executor_module
    for name in ("EXEC_MSG_DUPLICATE", "EXEC_MSG_UNRESOLVED", "EXEC_MSG_SAFETY_GATE",
                 "EXEC_MSG_RISK_CHECKS", "EXEC_MSG_RISK_STATUS",
                 "EXEC_MSG_STALE_QUOTE", "EXEC_MSG_NO_SPECS"):
        assert getattr(signals_module, name) is getattr(executor_module, name), (
            f"{name} must be imported from demo_executor, not re-typed")


def test_promote_error_mapping_is_distinguishable(monkeypatch):
    """M22: unknown model id -> 404; governance refusal -> 409; never a
    blanket 400 that hides the failure class from the operator/dashboard."""
    from sqlalchemy.orm import Session as _Session
    from backend.app.models.ml_model import MLModelRecord
    from backend.app.core.metrics_integrity import metric_digest
    for engine in _override_test_db(monkeypatch):
        with _Session(bind=engine) as session:
            session.add(MLModelRecord(
                model_id="mdl_rejected", dataset_version_tag="dv-test",
                algorithm="baseline", target_name="target_tp_before_sl",
                status="rejected", artifact_path="unused.joblib",
                metrics_digest=metric_digest("mdl_rejected", {}, {}, {}),
            ))
            session.commit()
        client = _client()
        resp_missing = client.post(
            f"{V1}/ml/models/no_such_model/promote",
            json={"notes": "t", "actor": "TEST"}, headers=api_headers())
        resp_rejected = client.post(
            f"{V1}/ml/models/mdl_rejected/promote",
            json={"notes": "t", "actor": "TEST"}, headers=api_headers())
    assert resp_missing.status_code == 404
    assert resp_rejected.status_code == 409
    assert "REJECTED" in resp_rejected.json()["detail"]


def test_backtest_without_approved_rules_is_conflict(monkeypatch):
    """M22: missing approved rules is a precondition on SERVER state -> 409,
    not a malformed-request 400."""
    import pandas as pd
    from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider

    idx = pd.date_range("2026-10-12 10:00:00", periods=100, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {"open": 2000.0, "high": 2001.0, "low": 1999.0, "close": 2000.5,
         "volume": 10.0, "spread": 20}, index=idx)

    class _SI:
        point = 0.01
        spread = 20
        tick_size = 0.01
        tick_value = 1.0
        trade_contract_size = 100.0
        volume_min = 0.01
        volume_step = 0.01
        volume_max = 10.0

    monkeypatch.setattr(MT5MarketDataProvider, "connect", lambda self: True)
    monkeypatch.setattr(
        MT5MarketDataProvider, "get_historical_dataframe",
        lambda self, symbol, timeframe, count=100: frame)
    monkeypatch.setattr(MT5MarketDataProvider, "get_symbol_info", lambda self, s: _SI())
    monkeypatch.setattr(MT5MarketDataProvider, "disconnect", lambda self: None)

    for _engine in _override_test_db(monkeypatch):
        client = _client()
        resp = client.post(
            f"{V1}/backtest/run",
            json={"symbol": "XAUUSD", "timeframe": "M15", "bars_count": 100,
                  "spread_points": 20},
            headers=api_headers(),
        )
    assert resp.status_code == 409, f"got {resp.status_code}: {resp.text[:200]}"
    assert "No approved strategy rules" in resp.json()["detail"]
