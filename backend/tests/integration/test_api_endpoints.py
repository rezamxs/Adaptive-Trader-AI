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


def test_backtest_run_endpoint():
    """Verify running backtest via API returns realistic metrics and stress scenarios."""
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


def test_dashboard_html_served():
    """Verify dashboard is served at root /."""
    res = client.get("/")
    assert res.status_code == 200
    assert "Adaptive Trader AI" in res.text
    assert "dir=\"rtl\"" in res.text
