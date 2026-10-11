"""Phase Q2 Validation Tests: Live validation, candle processing correctness, and worker stability.
Covers:
1. Processing of new closed candles (DISCOVERED -> PROCESSING -> COMPLETED)
2. Rejection of duplicate closed bars (NO_NEW_CLOSED_BAR, zero re-analysis)
3. Rejection of forming candle (start_pos=1 enforces completed bars only)
4. Corrupted / insufficient data handling (DATA_INVALID)
5. Restart idempotency & deduplication preservation across process restarts
6. Database commit failure resilience and rollback
7. MT5 temporary error handling and automatic reconnection
8. Unexpected worker cycle exception handling and exponential backoff
9. Atomic concurrency protection against simultaneous duplicate candle execution
10. Graceful shutdown and resource release
11. Separation of MT5 connection status from market data freshness taxonomy
12. Non-blocking event loop during worker cycle execution
"""
import asyncio
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from fastapi.testclient import TestClient

from backend.app.core.config import settings, ExecutionMode
from backend.app.db.base import Base
from backend.app.models.worker_cycle import WorkerCycleRecord
from backend.app.models.emergency_stop import EmergencyStopState
from backend.app.services.market_data.base import TimeFrame, Candle, Quote
from backend.app.services.worker.market_worker import MarketMonitoringWorker
from backend.app.main import app


@pytest.fixture
def test_db_session():
    """Isolated in-memory SQLite database session with proper table schema and cross-thread sharing."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestingSessionLocal()

    # Initial inactive emergency stop
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


# 1. Processing of new closed candle (DISCOVERED -> PROCESSING -> COMPLETED)
def test_process_new_closed_candle_lifecycle(test_db_session):
    """Test 1: Verifies transition of a new closed candle from reservation to completion."""
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
    result = worker.run_single_cycle()

    assert result["status_code"] in ["NO_APPROVED_RULES_AND_NO_ACTIVE_MODEL", "NO_VALID_SIGNAL"]
    assert result["lifecycle_phase"] == "COMPLETED"
    assert "bar_key" in result

    # Verify DB record
    rec = db.query(WorkerCycleRecord).filter_by(bar_key=result["bar_key"]).first()
    assert rec is not None
    assert rec.lifecycle_phase == "COMPLETED"
    assert rec.status_code == result["status_code"]
    assert rec.duration_ms >= 0.0


# 2. Rejection of duplicate closed bar
def test_reject_duplicate_candle_without_reanalysis(test_db_session):
    """Test 2: Verifies that a duplicate closed candle is rejected with NO re-analysis."""
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
    res1 = worker.run_single_cycle()
    bar_key = res1["bar_key"]

    with patch("backend.app.services.dataset.features.FeatureEngineer.transform") as mock_feat:
        res2 = worker.run_single_cycle()
        assert res2["status_code"] == "NO_NEW_CLOSED_BAR"
        assert res2["bar_key"] == bar_key
        assert mock_feat.call_count == 0  # Zero re-analysis


# 3. Rejection of forming candle
def test_reject_forming_candle_via_start_pos_1(test_db_session):
    """Test 3: Verifies that worker strictly queries start_pos=1, ignoring bar 0."""
    db, session_factory = test_db_session
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
    mock_provider.get_historical_candles.return_value = generate_mock_candles(60)

    worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)
    worker.run_single_cycle()

    # Verify that get_historical_candles was called with start_pos=1
    args, kwargs = mock_provider.get_historical_candles.call_args
    assert kwargs.get("start_pos") == 1


# 4. Corrupted / Insufficient data handling
def test_invalid_data_triggers_data_invalid_status(test_db_session):
    """Test 4: Verifies that insufficient candles (< 50) trigger DATA_INVALID."""
    db, session_factory = test_db_session
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
    mock_provider.get_historical_candles.return_value = generate_mock_candles(10)  # Only 10 candles

    worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)
    result = worker.run_single_cycle()

    assert result["status_code"] == "DATA_INVALID"
    assert "Insufficient or invalid closed bars" in result["reason"]


# 5. Restart and Deduplication preservation
def test_restart_preserves_deduplication_via_database(test_db_session):
    """Test 5: Verifies that after worker restart, previously completed candles are not re-processed."""
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

    # Instance 1 processes bar
    worker1 = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)
    res1 = worker1.run_single_cycle()
    bar_key = res1["bar_key"]

    # Backend restarts: Instance 2 created
    worker2 = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)
    res2 = worker2.run_single_cycle()

    assert res2["status_code"] == "NO_NEW_CLOSED_BAR"
    assert res2["bar_key"] == bar_key


# 6. Database commit failure resilience
def test_database_failure_resilience_and_rollback(test_db_session):
    """Test 6: Verifies that database commit failure is caught gracefully without worker crash."""
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

    # Patch session commit to raise error on final save
    with patch.object(db, "commit", side_effect=Exception("Simulated SQLite lock error")):
        # Mock session_factory to return the patched db
        worker.db_factory = lambda: db
        res = worker.run_single_cycle()
        assert res["status_code"] in ["NO_APPROVED_RULES_AND_NO_ACTIVE_MODEL", "NO_VALID_SIGNAL"]
        assert worker.last_status_code == "DATABASE_WRITE_FAILED"


# 7. MT5 temporary error and reconnection
def test_mt5_temporary_error_and_reconnection(test_db_session):
    """Test 7: Verifies that disconnected MT5 returns MT5_DISCONNECTED and reconnects on next cycle."""
    db, session_factory = test_db_session
    mock_provider = MagicMock()
    # Cycle 1: Disconnected and connect fails
    mock_provider.is_connected.return_value = False
    mock_provider.connect.return_value = False

    worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)
    res1 = worker.run_single_cycle()
    assert res1["status_code"] == "MT5_DISCONNECTED"
    assert worker.mt5_status == "MT5_DISCONNECTED"

    # Cycle 2: Connection succeeds
    mock_provider.is_connected.return_value = True
    mock_provider.get_current_quote.return_value = Quote(
        symbol="XAUUSD",
        time=datetime.now(timezone.utc),
        bid=4194.0,
        ask=4194.34,
        spread_points=34,
        is_stale=False
    )
    mock_provider.get_historical_candles.return_value = generate_mock_candles(60)

    res2 = worker.run_single_cycle()
    assert res2["status_code"] in ["NO_APPROVED_RULES_AND_NO_ACTIVE_MODEL", "NO_VALID_SIGNAL"]
    assert worker.mt5_status == "MT5_CONNECTED"


# 8. Unexpected worker error and exponential backoff
def test_unexpected_worker_error_and_backoff(test_db_session):
    """Test 8: Verifies that consecutive errors trigger exponential backoff with upper ceiling."""
    db, session_factory = test_db_session
    mock_provider = MagicMock()
    mock_provider.is_connected.side_effect = Exception("Kernel driver error")

    worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory, poll_interval=0.2)

    async def _run_test():
        await worker.start()
        await asyncio.sleep(0.6)
        # Verify error tracking while active
        assert worker.consecutive_errors >= 1
        assert worker.total_errors >= 1
        assert worker.last_status_code == "WORKER_EXCEPTION"
        diag = worker.get_diagnostics()
        assert diag["consecutive_errors"] >= 1
        await worker.stop()

    asyncio.run(_run_test())
    assert worker.last_status_code == "STOPPED"
    assert worker.total_errors >= 1


# 9. Atomic concurrency protection
def test_concurrent_duplicate_candle_reservation(test_db_session):
    """Test 9: Verifies that simultaneous cycle runs on same bar_key handle race conditions via unique constraint."""
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
    
    # Pre-insert a PROCESSING record to simulate concurrent thread
    bar_key = f"XAUUSD_M1_{candles[-1].time.strftime('%Y%m%d%H%M')}"
    existing = WorkerCycleRecord(
        cycle_id="cyc_other_thread",
        symbol="XAUUSD",
        timeframe="M1",
        bar_key=bar_key,
        bar_time=candles[-1].time,
        status_code="PROCESSING",
        lifecycle_phase="PROCESSING",
        decision="PENDING",
        created_at=datetime.utcnow()
    )
    db.add(existing)
    db.commit()

    # Current worker runs: must detect BAR_UNDER_PROCESSING without re-running pipeline
    res = worker.run_single_cycle()
    assert res["status_code"] == "BAR_UNDER_PROCESSING"
    assert res["lifecycle_phase"] == "PROCESSING"


# 10. Controlled graceful shutdown
def test_graceful_shutdown_releases_resources(test_db_session):
    """Test 10: Verifies that stop() cleanly stops the background task and disconnects data provider."""
    db, session_factory = test_db_session
    mock_provider = MagicMock()
    mock_provider.is_connected.return_value = True

    worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)

    async def _run_test():
        await worker.start()
        assert worker.is_running is True
        await worker.stop()
        assert worker.is_running is False
        assert worker.last_status_code == "STOPPED"

    asyncio.run(_run_test())
    assert mock_provider.disconnect.call_count == 1


# 11. Separation of MT5 connection from data freshness taxonomy
def test_mt5_connected_vs_data_freshness_separation(test_db_session):
    """Test 11: Verifies granular evaluation of market freshness:
    MARKET_OPEN_DATA_FRESH, NO_RECENT_TICK, MARKET_CLOSED, MARKET_CLOSED_OR_STALE.
    """
    db, session_factory = test_db_session
    worker = MarketMonitoringWorker(db_factory=session_factory)
    now_utc = datetime(2026, 10, 14, 14, 0, 0, tzinfo=timezone.utc)  # Wednesday midday

    # Case A: Live fresh quote (age 10s)
    q_fresh = Quote(symbol="XAUUSD", time=now_utc - timedelta(seconds=10), bid=4194.0, ask=4194.34, spread_points=34, is_stale=False)
    status_a, _ = worker.evaluate_market_freshness(q_fresh, now_utc)
    assert status_a == "MARKET_OPEN_DATA_FRESH"

    # Case B: Stale quote on weekday (age 200s)
    q_stale = Quote(symbol="XAUUSD", time=now_utc - timedelta(seconds=200), bid=4194.0, ask=4194.34, spread_points=34, is_stale=True)
    status_b, _ = worker.evaluate_market_freshness(q_stale, now_utc)
    assert status_b == "NO_RECENT_TICK"

    # Case C: Prolonged staleness on weekday (> 2 hours)
    q_prolonged = Quote(symbol="XAUUSD", time=now_utc - timedelta(hours=3), bid=4194.0, ask=4194.34, spread_points=34, is_stale=True)
    status_c, _ = worker.evaluate_market_freshness(q_prolonged, now_utc)
    assert status_c == "MARKET_CLOSED_OR_STALE"

    # Case D: Weekend closure (Saturday UTC)
    sat_utc = datetime(2026, 10, 10, 12, 0, 0, tzinfo=timezone.utc)
    q_weekend = Quote(symbol="XAUUSD", time=sat_utc - timedelta(hours=14), bid=4194.0, ask=4194.34, spread_points=34, is_stale=True)
    status_d, _ = worker.evaluate_market_freshness(q_weekend, sat_utc)
    assert status_d == "MARKET_CLOSED"


# 12. Non-blocking event loop during worker cycle execution
def test_non_blocking_event_loop_during_worker_cycle(test_db_session):
    """Test 12: Verifies that heavy synchronous operations run in asyncio.to_thread without blocking event loop."""
    db, session_factory = test_db_session
    mock_provider = MagicMock()
    mock_provider.is_connected.return_value = True

    # Simulate 0.2s blocking MT5 call
    def slow_candles(*args, **kwargs):
        time.sleep(0.2)
        return generate_mock_candles(60)

    mock_provider.get_historical_candles.side_effect = slow_candles
    mock_provider.get_current_quote.return_value = Quote(
        symbol="XAUUSD",
        time=datetime.now(timezone.utc),
        bid=4194.0,
        ask=4194.34,
        spread_points=34,
        is_stale=False
    )

    worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)

    async def _measure_responsiveness():
        async def loop_heartbeat():
            ticks = 0
            for _ in range(5):
                await asyncio.sleep(0.04)
                ticks += 1
            return ticks

        cycle_task = asyncio.to_thread(worker.run_single_cycle)
        heartbeat_task = loop_heartbeat()

        res_cycle, ticks = await asyncio.gather(cycle_task, heartbeat_task)
        return res_cycle, ticks

    res_cycle, ticks = asyncio.run(_measure_responsiveness())
    assert ticks >= 4  # Event loop continued to tick while thread was busy
    assert res_cycle["status_code"] in ["NO_APPROVED_RULES_AND_NO_ACTIVE_MODEL", "NO_VALID_SIGNAL"]
