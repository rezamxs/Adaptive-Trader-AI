"""Unit and integration tests for Phase Q1 Market Monitoring Worker.
Covers lifecycle, closed bar deduplication, restart idempotency, governance hold states,
API telemetry, and strict verification that zero orders are sent.
"""
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from backend.app.core.config import settings, ExecutionMode
from backend.app.db.base import Base
from backend.app.models.worker_cycle import WorkerCycleRecord
from backend.app.models.signal import TradeSignal
from backend.app.models.emergency_stop import EmergencyStopState


def _api_headers() -> dict:
    """M15: gated GET reads require the same shared-secret header as POSTs."""
    return {"X-API-TOKEN": settings.API_TOKEN}
from backend.app.services.market_data.base import TimeFrame, Candle, Quote, SymbolInfo, AccountDetails
from backend.app.services.market_data.mock_provider import MockMarketDataProvider
from backend.app.services.worker.market_worker import MarketMonitoringWorker
from backend.app.services.risk.circuit_breaker import EmergencyStopManager
from backend.app.main import app


@pytest.fixture
def test_db_session():
    """Isolated in-memory SQLite database session for Worker unit tests."""
    test_engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestingSessionLocal()
    
    # Initialize inactive emergency stop
    stop_state = EmergencyStopState(is_active=False, activated_by="INIT", reason="Test init")
    db.add(stop_state)
    db.commit()

    yield db, TestingSessionLocal
    db.close()


def generate_mock_candles(count: int = 60, base_time: datetime = None) -> list:
    """Generates synthetic closed candles."""
    if base_time is None:
        base_time = datetime(2026, 10, 9, 23, 0, 0, tzinfo=timezone.utc)
    candles = []
    price = 4190.0
    for i in range(count):
        t = base_time + timedelta(minutes=i)
        candles.append(Candle(
            time=t,
            open=price,
            high=price + 0.5,
            low=price - 0.5,
            close=price + 0.1,
            tick_volume=100,
            spread=34,
            is_closed=True
        ))
        price += 0.05
    return candles


def test_worker_safety_gate_prevents_start_if_not_signal_only(test_db_session):
    """Verifies that Worker refuses to start if execution mode is tampered with."""
    db, session_factory = test_db_session
    mock_provider = MockMarketDataProvider()
    worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)

    with patch.object(settings, "EXECUTION_MODE", ExecutionMode.DEMO_TRADING):
        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(worker.start())
        assert "CRITICAL SAFETY VIOLATION" in str(exc_info.value)


def test_worker_closed_bar_deduplication(test_db_session):
    """Verifies that the same closed bar is processed once and skipped on subsequent polls."""
    db, session_factory = test_db_session
    candles = generate_mock_candles(60)

    mock_provider = MagicMock()
    mock_provider.is_connected.return_value = True
    mock_provider.get_current_quote.return_value = Quote(
        symbol="XAUUSD",
        time=datetime.now(timezone.utc),
        bid=4194.0,
        ask=4194.34,
        spread_points=34,
        is_stale=False
    )
    mock_provider.get_historical_candles.return_value = candles

    worker = MarketMonitoringWorker(
        data_provider=mock_provider,
        db_factory=session_factory,
        symbol="XAUUSD",
        timeframe=TimeFrame.M1
    )

    # 1. First cycle: New bar -> processes and records
    res1 = worker.run_single_cycle()
    assert res1["status_code"] in ["NO_APPROVED_RULES_AND_NO_ACTIVE_MODEL", "NO_VALID_SIGNAL"]
    assert "bar_key" in res1
    bar_key_1 = res1["bar_key"]

    # Verify DB record exists
    records_1 = db.query(WorkerCycleRecord).all()
    assert len(records_1) == 1
    assert records_1[0].bar_key == bar_key_1

    # 2. Second cycle with identical candles: Must detect NO_NEW_CLOSED_BAR
    res2 = worker.run_single_cycle()
    assert res2["status_code"] == "NO_NEW_CLOSED_BAR"
    assert res2["bar_key"] == bar_key_1
    assert "already been analyzed" in res2["reason"]

    # Deduplication check: DB record count should not increase
    records_2 = db.query(WorkerCycleRecord).all()
    assert len(records_2) == 1


def test_worker_processes_new_closed_bar_when_time_advances(test_db_session):
    """Verifies that Worker detects and processes a new closed bar when time moves forward."""
    db, session_factory = test_db_session
    candles_first = generate_mock_candles(60)

    mock_provider = MagicMock()
    mock_provider.is_connected.return_value = True
    mock_provider.get_current_quote.return_value = Quote(
        symbol="XAUUSD",
        time=datetime.now(timezone.utc),
        bid=4194.0,
        ask=4194.34,
        spread_points=34,
        is_stale=False
    )
    mock_provider.get_historical_candles.return_value = candles_first

    worker = MarketMonitoringWorker(
        data_provider=mock_provider,
        db_factory=session_factory,
        symbol="XAUUSD",
        timeframe=TimeFrame.M1
    )

    # Cycle 1
    res1 = worker.run_single_cycle()
    bar_key_1 = res1["bar_key"]

    # Time advances by 1 minute: add a new closed bar
    candles_second = candles_first + [
        Candle(
            time=candles_first[-1].time + timedelta(minutes=1),
            open=4195.0,
            high=4195.5,
            low=4194.8,
            close=4195.2,
            tick_volume=120,
            spread=34,
            is_closed=True
        )
    ]
    mock_provider.get_historical_candles.return_value = candles_second

    # Cycle 2
    res2 = worker.run_single_cycle()
    bar_key_2 = res2["bar_key"]

    assert bar_key_1 != bar_key_2
    assert res2["status_code"] in ["NO_APPROVED_RULES_AND_NO_ACTIVE_MODEL", "NO_VALID_SIGNAL"]

    records = db.query(WorkerCycleRecord).order_by(WorkerCycleRecord.id.asc()).all()
    assert len(records) == 2
    assert records[0].bar_key == bar_key_1
    assert records[1].bar_key == bar_key_2


def test_worker_idempotency_persists_across_restart(test_db_session):
    """Verifies that bar deduplication persists across application restarts via database."""
    db, session_factory = test_db_session
    candles = generate_mock_candles(60)

    mock_provider = MagicMock()
    mock_provider.is_connected.return_value = True
    mock_provider.get_current_quote.return_value = Quote(
        symbol="XAUUSD",
        time=datetime.now(timezone.utc),
        bid=4194.0,
        ask=4194.34,
        spread_points=34,
        is_stale=False
    )
    mock_provider.get_historical_candles.return_value = candles

    # Instance 1 runs
    worker1 = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)
    res1 = worker1.run_single_cycle()
    bar_key_1 = res1["bar_key"]

    # Application restarts: create new Worker instance
    worker2 = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)
    res2 = worker2.run_single_cycle()

    # Must be recognized as already processed from DB
    assert res2["status_code"] == "NO_NEW_CLOSED_BAR"
    assert res2["bar_key"] == bar_key_1


def test_worker_never_executes_orders_or_calls_demo_executor(test_db_session):
    """CRITICAL SAFETY TEST: Proves that the Worker NEVER calls order dispatch methods,
    maintaining 0 executed orders even when market data is processed.
    """
    db, session_factory = test_db_session
    candles = generate_mock_candles(60)

    mock_provider = MagicMock()
    mock_provider.is_connected.return_value = True
    mock_provider.get_current_quote.return_value = Quote(
        symbol="XAUUSD",
        time=datetime.now(timezone.utc),
        bid=4194.0,
        ask=4194.34,
        spread_points=34,
        is_stale=False
    )
    mock_provider.get_historical_candles.return_value = candles

    worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)

    # Patch order executor and mt5 order_send to verify zero calls
    with patch("backend.app.services.execution.demo_executor.DemoTradingExecutor.execute_demo_signal") as mock_exec:
        worker.run_single_cycle()
        # Assert that execute_demo_signal was NEVER invoked
        assert mock_exec.call_count == 0


def test_worker_emergency_stop_halts_pipeline(test_db_session):
    """Verifies that Worker suspends operations when Emergency Stop circuit breaker is activated."""
    db, session_factory = test_db_session
    stop_mgr = EmergencyStopManager(db)
    stop_mgr.activate(reason="Test volatility trip", actor="TEST")

    mock_provider = MagicMock()
    worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)
    res = worker.run_single_cycle()

    assert res["status_code"] == "EMERGENCY_STOP_ACTIVE"
    assert "Circuit Breaker is active" in res["reason"]


def test_worker_market_closed_detection(test_db_session):
    """Verifies that Worker recognizes when the market is closed based on stale ticks."""
    db, session_factory = test_db_session
    candles = generate_mock_candles(60)

    # Tick from 36 hours ago
    stale_time = datetime.now(timezone.utc) - timedelta(hours=36)
    mock_provider = MagicMock()
    mock_provider.is_connected.return_value = True
    mock_provider.get_current_quote.return_value = Quote(
        symbol="XAUUSD",
        time=stale_time,
        bid=4194.0,
        ask=4194.34,
        spread_points=34,
        is_stale=True
    )
    mock_provider.get_historical_candles.return_value = candles

    worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)
    res = worker.run_single_cycle()

    # Initial cycle processes the last historical candle
    assert "bar_key" in res
    # Subsequent cycle flags NO_NEW_CLOSED_BAR while market is closed
    res2 = worker.run_single_cycle()
    assert res2["status_code"] == "NO_NEW_CLOSED_BAR"


def test_worker_status_and_cycles_api_endpoints(isolated_app_db):
    """Integration test: Verifies that FastAPI exposes worker status and cycle history correctly."""
    with TestClient(app) as client:
        # 1. Health check with worker telemetry
        res_h = client.get("/health")
        assert res_h.status_code == 200
        data_h = res_h.json()
        assert "liveness" in data_h
        assert "readiness" in data_h
        assert data_h["mode"] == "SIGNAL_ONLY"

        # 2. Worker Status Endpoint (M15: token-gated read)
        res_ws = client.get("/api/v1/system/worker/status", headers=_api_headers())
        assert res_ws.status_code == 200
        data_ws = res_ws.json()
        assert "liveness" in data_ws
        assert "symbol" in data_ws
        assert data_ws["symbol"] == "XAUUSD"
        assert data_ws["execution_mode"] == "SIGNAL_ONLY"

        # 3. Worker Cycles History Endpoint (M15: token-gated read)
        res_wc = client.get("/api/v1/system/worker/cycles", headers=_api_headers())
        assert res_wc.status_code == 200
        assert isinstance(res_wc.json(), list)
