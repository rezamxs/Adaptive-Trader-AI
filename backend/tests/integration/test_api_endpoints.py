"""Integration tests for FastAPI REST Endpoints."""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify health check returns 200 and healthy status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["mode"] == "SIGNAL_ONLY"


def test_system_status_endpoint():
    """Verify system status reports execution mode and MT5 state."""
    res = client.get("/api/v1/system/status")
    assert res.status_code == 200
    data = res.json()
    assert data["execution_mode"] == "SIGNAL_ONLY"
    assert "mt5" in data
    assert "emergency_stop" in data


def test_emergency_stop_api_lifecycle():
    """Verify API activation and deactivation of circuit breaker."""
    # Activate
    res = client.post("/api/v1/system/emergency-stop/activate", json={
        "reason": "API test stop",
        "actor": "TEST_RUNNER"
    })
    assert res.status_code == 200
    assert res.json()["status"] == "ACTIVATED"

    # Check status reflects active
    status_res = client.get("/api/v1/system/status")
    assert status_res.json()["emergency_stop"]["is_active"] is True

    # Deactivate
    res_deact = client.post("/api/v1/system/emergency-stop/deactivate", json={
        "notes": "API test clear",
        "actor": "TEST_RUNNER"
    })
    assert res_deact.status_code == 200
    assert res_deact.json()["status"] == "DEACTIVATED"

    # Check status reflects inactive
    status_res2 = client.get("/api/v1/system/status")
    assert status_res2.json()["emergency_stop"]["is_active"] is False


def test_rules_api_list():
    """Verify listing trading rules via API."""
    res = client.get("/api/v1/rules/")
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

    res = client.post("/api/v1/backtest/run", json={
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

    res = client.post("/api/v1/backtest/run", json={
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
