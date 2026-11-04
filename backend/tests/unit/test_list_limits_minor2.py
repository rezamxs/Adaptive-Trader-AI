"""Minor 2: every list endpoint carries a bounded `limit` (1..500) so SQLite's
LIMIT -1 (unbounded table dump) can never be requested, and the backtest
equity curve is capped at storage/return time.
"""
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app

V1 = "/api/v1"


@pytest.fixture
def client(isolated_app_db):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def api_headers() -> dict:
    return {"X-API-TOKEN": settings.API_TOKEN}


@pytest.mark.parametrize("path", [
    f"{V1}/backtest/runs",
    f"{V1}/rules",
    f"{V1}/documents",
    f"{V1}/ml/models",
])
def test_list_endpoints_reject_out_of_range_limits(client, path):
    """An over-cap limit must be a 422 request error, never a full-table dump."""
    resp = client.get(path, params={"limit": 501}, headers=api_headers())
    assert resp.status_code == 422, f"{path} limit=501 -> {resp.status_code}"
    resp = client.get(path, params={"limit": 0}, headers=api_headers())
    assert resp.status_code == 422, f"{path} limit=0 -> {resp.status_code}"


@pytest.mark.parametrize("path", [
    f"{V1}/backtest/runs",
    f"{V1}/rules",
    f"{V1}/documents",
    f"{V1}/ml/models",
])
def test_list_endpoints_accept_bounded_limits(client, path):
    resp = client.get(path, params={"limit": 10}, headers=api_headers())
    assert resp.status_code == 200, f"{path} limit=10 -> {resp.status_code}: {resp.text[:120]}"


def test_equity_curve_is_capped_with_provenance(monkeypatch):
    """Minor 2: the stored/returned equity curve is a bounded visualization —
    drawdown is computed on the FULL timeline first, then the curve is
    downsampled with the original length recorded."""
    from backend.app.services.backtest import metrics as metrics_module
    from backend.app.services.backtest.metrics import BacktestMetrics, downsample_equity_curve

    curve = [{"time": str(i), "equity": 10000.0 + i} for i in range(50)]
    sampled = downsample_equity_curve(curve, max_points=10)
    assert len(sampled) <= 10
    assert sampled[0] == curve[0] and sampled[-1] == curve[-1], (
        "downsample must preserve the endpoints")

    # > max_points input through calculate(): the cap engages and reports it.
    monkeypatch.setattr(metrics_module, "_MAX_EQUITY_CURVE_POINTS", 10)
    trades = [{"net_profit": 0.1, "exit_time": f"t{i}"} for i in range(15)]
    result = BacktestMetrics.calculate(trades, initial_balance=10000.0)
    assert result["equity_curve_points_original"] == 16  # START + 15 exits
    assert result["equity_curve_downsampled"] is True
    assert len(result["equity_curve"]) <= 10
    assert result["max_drawdown_percent"] == 0.0  # monotonic gains: full-timeline truth

    # Under the cap the curve passes through untouched.
    small = BacktestMetrics.calculate(
        [{"net_profit": 0.1, "exit_time": f"t{i}"} for i in range(3)],
        initial_balance=10000.0,
    )
    assert small["equity_curve_downsampled"] is False
    assert len(small["equity_curve"]) == 4
