"""Integration tests for FastAPI REST Endpoints."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings

client = TestClient(app)

# C1: state-changing endpoints require the API token.
AUTH = {"X-API-TOKEN": settings.API_TOKEN}


@pytest.fixture(autouse=True)
def _isolate_real_db(isolated_app_db):
    """Every test in this module runs against the isolated database (C1)."""
    yield


def test_health_endpoint():
    """Verify health check returns 200 and healthy status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["mode"] == "SIGNAL_ONLY"


def test_system_status_endpoint():
    """Verify system status reports execution mode and MT5 state (M15: gated read)."""
    res = client.get("/api/v1/system/status", headers=AUTH)
    assert res.status_code == 200
    data = res.json()
    assert data["execution_mode"] == "SIGNAL_ONLY"
    assert "mt5" in data
    assert "emergency_stop" in data


def test_emergency_stop_api_lifecycle():
    """Verify API activation and deactivation of circuit breaker."""
    # Activate
    res = client.post("/api/v1/system/emergency-stop/activate", headers=AUTH, json={
        "reason": "API test stop",
        "actor": "TEST_RUNNER"
    })
    assert res.status_code == 200
    assert res.json()["status"] == "ACTIVATED"

    # Check status reflects active (M15: gated read)
    status_res = client.get("/api/v1/system/status", headers=AUTH)
    assert status_res.json()["emergency_stop"]["is_active"] is True

    # Deactivate
    res_deact = client.post("/api/v1/system/emergency-stop/deactivate", headers=AUTH, json={
        "notes": "API test clear",
        "actor": "TEST_RUNNER"
    })
    assert res_deact.status_code == 200
    assert res_deact.json()["status"] == "DEACTIVATED"

    # Check status reflects inactive (M15: gated read)
    status_res2 = client.get("/api/v1/system/status", headers=AUTH)
    assert status_res2.json()["emergency_stop"]["is_active"] is False


def test_rules_api_list():
    """Verify listing trading rules via API (M15: gated read)."""
    res = client.get("/api/v1/rules/", headers=AUTH)
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_backtest_run_endpoint(monkeypatch):
    """Verify running backtest via API returns realistic metrics and stress scenarios using test provider."""
    from backend.app.services.market_data.mock_provider import MockMarketDataProvider
    from backend.app.services.market_data.base import TimeFrame

    class StubMT5Provider:
        def connect(self):
            return True

        def disconnect(self):
            pass

        def get_historical_dataframe(self, symbol, timeframe, count):
            tf = TimeFrame(timeframe) if isinstance(timeframe, str) else timeframe
            return MockMarketDataProvider().get_historical_dataframe(symbol, tf, count=count)

        def get_symbol_info(self, symbol):
            return MockMarketDataProvider().get_symbol_info(symbol)

    from backend.app.models.rule import TradingRule
    sample_rule = TradingRule(
        rule_id="rule_rsi_oversold",
        name="RSI Oversold Buy",
        category="entry",
        description="Buy when RSI < 70",
        conditions=[{"indicator": "RSI", "operator": "<", "threshold": 70.0}],
        parameters={"rsi_period": 14},
        symbol_scope=["EURUSD"],
        timeframe_scope=["M15"],
        approval_status="approved",
        version=1,
    )
    monkeypatch.setattr(
        "backend.app.api.v1.endpoints.backtest.KnowledgeManager.get_active_strategy_rules",
        lambda self: [sample_rule],
    )
    monkeypatch.setattr("backend.app.api.v1.endpoints.backtest.MT5MarketDataProvider", StubMT5Provider)

    res = client.post("/api/v1/backtest/run", headers=AUTH, json={
        "symbol": "EURUSD",
        "timeframe": "M15",
        "bars_count": 120,
        "spread_points": 15,
        "slippage_points": 3,
        "commission_per_lot": 7.0,
        "initial_balance": 10000.0
    })
    assert res.status_code == 200
    data = res.json()
    assert "run_id" in data
    assert "metrics" in data
    assert "net_profit" in data["metrics"]
    assert "stress_test" in data


def test_backtest_run_endpoint_returns_503_when_mt5_unavailable(monkeypatch):
    """Verify 503 Service Unavailable is cleanly returned when MT5 market provider fails to connect."""
    class DisconnectedMT5Provider:
        def connect(self):
            return False

        def disconnect(self):
            pass

    monkeypatch.setattr("backend.app.api.v1.endpoints.backtest.MT5MarketDataProvider", DisconnectedMT5Provider)

    res = client.post("/api/v1/backtest/run", headers=AUTH, json={
        "symbol": "EURUSD",
        "timeframe": "M15",
        "bars_count": 120,
    })
    assert res.status_code == 503
    assert "MT5 is unavailable" in res.json()["detail"]


def test_dashboard_html_served():
    """Verify dashboard is served at root /."""
    res = client.get("/")
    assert res.status_code == 200
    assert "Adaptive Trader AI" in res.text
    assert "dir=\"rtl\"" in res.text


def _sqlite_content_snapshot(db_path):
    """Read-only content fingerprint of a SQLite file.

    Returns {table: {"rows": <n>, "sha256": <hex>}}. Built from
    ``con.iterdump()`` — the dump is produced by sqlite3 itself, so NO SQL is
    interpolated in this process (identifiers cannot be parameterized in
    SQLite; per-table dynamic queries are therefore avoided by design). The
    per-table digest covers the CREATE statement plus every INSERT in stable
    dump order, so schema changes, insertions, deletions and in-place content
    changes are all detected — not just row-count drift.
    """
    import hashlib
    import re
    import sqlite3
    from pathlib import Path

    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        dump_lines = list(con.iterdump())
    finally:
        con.close()

    create_re = re.compile(r'^CREATE TABLE (?:IF NOT EXISTS )?"?([^"\s(]+)"?')
    insert_re = re.compile(r'^INSERT INTO "?([^"\s(]+)"?')

    # First pass: table registry from the CREATE statements (deterministic
    # dump order); internal sqlite_* tables are excluded.
    tables = []
    for line in dump_lines:
        m = create_re.match(line)
        if m and not m.group(1).startswith("sqlite_"):
            tables.append(m.group(1))

    snapshot = {table: {"rows": 0, "sha256": hashlib.sha256(table.encode("utf-8")).hexdigest()}
                for table in tables}
    body_re = re.compile(r"^INSERT INTO \"?([^\"\s(]+)\"?")
    per_table_lines = {table: [] for table in tables}
    for line in dump_lines:
        m = body_re.match(line)
        if not m:
            continue
        table = m.group(1)
        if table in per_table_lines:
            per_table_lines[table].append(line)
            snapshot[table]["rows"] += 1

    for table in tables:
        payload = "\n".join([f"TABLE {table}"] + per_table_lines[table])
        snapshot[table]["sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return snapshot


def _real_db_snapshot():
    """Content snapshot of the real adaptive_trader.db (C1 sentinel)."""
    from pathlib import Path

    return _sqlite_content_snapshot(
        Path(settings.DATABASE_URL.replace("sqlite:///", "", 1)))


def test_api_tests_cannot_change_real_database():
    """C1 regression: API tests must never write the real adaptive_trader.db.

    The emergency-stop lifecycle below exercises the real endpoint code path;
    with test isolation in place it runs against an isolated database, so a
    read-only content snapshot of the real database (every column of every row
    of every table, including emergency_stop_states, audit_logs and
    backtest_runs) must be identical before and after.
    """
    real_db = Path(settings.DATABASE_URL.replace("sqlite:///", "", 1))
    if not real_db.is_file():
        pytest.skip(
            "real adaptive_trader.db not present in this environment; the C1 "
            "sentinel only applies when the production database file exists"
        )
    before = _real_db_snapshot()
    res = client.post("/api/v1/system/emergency-stop/activate", headers=AUTH, json={
        "reason": "C1 isolation regression probe", "actor": "AUDIT_PROBE"
    })
    assert res.status_code == 200
    res_deact = client.post("/api/v1/system/emergency-stop/deactivate", headers=AUTH, json={
        "notes": "C1 isolation regression probe", "actor": "AUDIT_PROBE"
    })
    assert res_deact.status_code == 200
    after = _real_db_snapshot()
    assert after == before, f"API tests changed the real database: {before} != {after}"


def test_content_snapshot_detects_content_changes(tmp_path):
    """The C1 sentinel must catch same-row-count in-place edits, not just counts.

    Uses a throwaway SQLite file only; the real database is never written here.
    """
    import sqlite3

    fake = tmp_path / "fake_state.db"
    con = sqlite3.connect(fake)
    con.execute(
        "CREATE TABLE emergency_stop_states "
        "(id INTEGER PRIMARY KEY, is_active INTEGER, reason TEXT)")
    con.execute("INSERT INTO emergency_stop_states VALUES (1, 0, 'operator inactive')")
    con.commit()
    con.close()

    first = _sqlite_content_snapshot(fake)

    # Equal row count, different content: a row silently flipped in place.
    con = sqlite3.connect(fake)
    con.execute("UPDATE emergency_stop_states SET is_active = 1, reason = 'wiped by tests'")
    con.commit()
    con.close()
    flipped = _sqlite_content_snapshot(fake)

    assert first["emergency_stop_states"]["rows"] == flipped["emergency_stop_states"]["rows"] == 1
    assert first != flipped, "in-place content change was not detected"
    assert first["emergency_stop_states"]["sha256"] != flipped["emergency_stop_states"]["sha256"]

    # Identical content yields an identical snapshot (no false positives).
    assert _sqlite_content_snapshot(fake) == flipped

    # Row insertion is detected.
    con = sqlite3.connect(fake)
    con.execute("INSERT INTO emergency_stop_states VALUES (2, 0, 'another')")
    con.commit()
    con.close()
    grown = _sqlite_content_snapshot(fake)
    assert grown != flipped and grown["emergency_stop_states"]["rows"] == 2

    # Row deletion is detected.
    con = sqlite3.connect(fake)
    con.execute("DELETE FROM emergency_stop_states WHERE id = 2")
    con.commit()
    con.close()
    assert _sqlite_content_snapshot(fake) == flipped
