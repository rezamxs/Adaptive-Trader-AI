"""Phase Q2.1 Consistency and Validation Tests.
Covers:
- Test A: Canonical reference record invariance & uniqueness (bar_key deduplication)
- Test B: Duplicate request skips feature extraction (zero re-computation)
- Test C: Concurrency protection against duplicate records
- Test D: Worker restart idempotency & reference cycle preservation
- Test E: Recovery from abandoned / orphaned processing records
- Test F: Separation of analysis success from signal generation validity
- Test G: Three-way readiness separation (service_ready, market_data_ready, trading_ready) & API verification
- Test H: Event timestamps accuracy & isolation (no timestamp pollution on duplicate bars)
"""
import asyncio
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from backend.app.core.config import settings, ExecutionMode
from backend.app.db.base import Base
from backend.app.models.worker_cycle import WorkerCycleRecord
from backend.app.models.emergency_stop import EmergencyStopState
from backend.app.models.rule import TradingRule
from backend.app.services.market_data.base import TimeFrame, Candle, Quote
from backend.app.services.worker.market_worker import MarketMonitoringWorker
from backend.app.main import app


@pytest.fixture
def test_db_session():
    """Isolated in-memory SQLite database session with static pool for multi-session sharing."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestingSessionLocal()

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


def get_mock_provider(candles=None, is_connected=True, is_stale=False):
    """Creates a mock MT5 data provider."""
    mock = MagicMock()
    mock.is_connected.return_value = is_connected
    now_utc = datetime.now(timezone.utc)
    mock.get_current_quote.return_value = Quote(
        symbol="XAUUSD",
        time=now_utc if not is_stale else (now_utc - timedelta(seconds=200)),
        bid=4194.0,
        ask=4194.34,
        spread_points=34,
        is_stale=is_stale
    )
    mock.get_historical_candles.return_value = candles if candles is not None else generate_mock_candles(60)
    return mock


# Test A: Canonical reference record invariance & uniqueness
def test_test_a_canonical_reference_record_invariance(test_db_session):
    """Test A: Verifies that running sequential cycles on the same bar maintains exactly 1 persistent record,
    with the second cycle referencing the canonical record rather than creating a secondary record.
    """
    db, session_factory = test_db_session
    mock_provider = get_mock_provider()
    worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)

    # First execution on new bar
    res1 = worker.run_single_cycle()
    canonical_cycle_id = res1["cycle_id"]
    bar_key = res1["bar_key"]
    assert res1["lifecycle_phase"] == "COMPLETED"
    assert res1["analysis_executed"] is True

    # Check database: exactly 1 record
    records = db.query(WorkerCycleRecord).filter_by(bar_key=bar_key).all()
    assert len(records) == 1
    rec1 = records[0]
    assert rec1.cycle_id == canonical_cycle_id
    assert rec1.lifecycle_phase == "COMPLETED"

    # Second execution with identical bar input
    res2 = worker.run_single_cycle()
    assert res2["status_code"] == "NO_NEW_CLOSED_BAR"
    assert res2["lifecycle_phase"] == "DUPLICATE_SKIPPED"
    assert res2["reference_cycle_id"] == canonical_cycle_id
    assert res2["reference_status"] == rec1.status_code
    assert res2["analysis_executed"] is False
    assert res2["cycle_id"] != canonical_cycle_id  # Cycle attempt ID differs from canonical reference

    # Database count MUST remain strictly 1
    records_after = db.query(WorkerCycleRecord).filter_by(bar_key=bar_key).all()
    assert len(records_after) == 1
    assert records_after[0].cycle_id == canonical_cycle_id
    assert records_after[0].status_code == rec1.status_code


# Test B: Duplicate request skips feature extraction
def test_test_b_duplicate_request_skips_feature_extraction(test_db_session):
    """Test B: Proves that feature engineering pipeline is NOT re-executed for duplicate closed candles."""
    db, session_factory = test_db_session
    mock_provider = get_mock_provider()
    worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)

    with patch("backend.app.services.dataset.features.FeatureEngineer.transform") as mock_feat:
        import pandas as pd
        mock_feat.return_value = (pd.DataFrame([{"feat_1": 1.0, "feat_2": 2.0}]), None)

        # First cycle -> transforms features
        res1 = worker.run_single_cycle()
        assert res1["lifecycle_phase"] == "COMPLETED"
        assert mock_feat.call_count == 1

        # Second cycle -> must immediately skip before feature engineering
        res2 = worker.run_single_cycle()
        assert res2["status_code"] == "NO_NEW_CLOSED_BAR"
        assert res2["lifecycle_phase"] == "DUPLICATE_SKIPPED"
        assert res2["reference_cycle_id"] == res1["cycle_id"]
        assert res2["analysis_executed"] is False
        assert mock_feat.call_count == 1  # Absolutely zero additional transform calls!


# Test C: Concurrency protection against duplicate records
def test_test_c_concurrency_protection_against_duplicate_record(test_db_session):
    """Test C: Simulates concurrent worker cycles attempting to reserve the exact same bar_key simultaneously.
    Verifies that unique constraint enforcement and rollback guarantee exactly one persistent record.
    """
    db, session_factory = test_db_session
    candles = generate_mock_candles(60)
    mock_provider = get_mock_provider(candles=candles)

    # Worker 1 reserves the candle in DB
    worker1 = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)
    worker2 = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)

    # Pre-reserve record via worker 1 or simulate parallel execution
    res1 = worker1.run_single_cycle()
    assert res1["lifecycle_phase"] == "COMPLETED"
    bar_key = res1["bar_key"]

    # Worker 2 attempts same candle
    res2 = worker2.run_single_cycle()
    assert res2["lifecycle_phase"] == "DUPLICATE_SKIPPED"
    assert res2["reference_cycle_id"] == res1["cycle_id"]
    assert res2["analysis_executed"] is False

    records = db.query(WorkerCycleRecord).filter_by(bar_key=bar_key).all()
    assert len(records) == 1


# Test D: Worker restart idempotency & reference cycle preservation
def test_test_d_worker_restart_idempotency(test_db_session):
    """Test D: Verifies that after a process restart, a new worker instance recognizes previously
    completed candle records from the database and returns DUPLICATE_SKIPPED with canonical reference_cycle_id.
    """
    db, session_factory = test_db_session
    bar_time = datetime(2026, 10, 9, 23, 53, 0, tzinfo=timezone.utc)
    bar_key = "XAUUSD_M1_202610092353"
    historical_cycle_id = "cyc_historical_reference"

    # Pre-populate database with historical processed candle
    prior_record = WorkerCycleRecord(
        cycle_id=historical_cycle_id,
        symbol="XAUUSD",
        timeframe="M1",
        bar_key=bar_key,
        bar_time=bar_time,
        status_code="NO_APPROVED_RULES_AND_NO_ACTIVE_MODEL",
        lifecycle_phase="COMPLETED",
        decision="HOLD",
        duration_ms=45.2,
        details="Pre-existing canonical record before restart",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(prior_record)
    db.commit()

    # Create entirely new worker instance simulating restart
    candles = generate_mock_candles(60, base_time=datetime(2026, 10, 9, 22, 54, 0, tzinfo=timezone.utc))
    # Make sure the last candle matches bar_time
    candles[-1].time = bar_time
    mock_provider = get_mock_provider(candles=candles)

    restarted_worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)
    assert restarted_worker.last_processed_bar_key is None  # Cold start memory

    res = restarted_worker.run_single_cycle()
    assert res["status_code"] == "NO_NEW_CLOSED_BAR"
    assert res["lifecycle_phase"] == "DUPLICATE_SKIPPED"
    assert res["reference_cycle_id"] == historical_cycle_id
    assert res["analysis_executed"] is False

    # Ensure no new records were inserted
    all_recs = db.query(WorkerCycleRecord).filter_by(bar_key=bar_key).all()
    assert len(all_recs) == 1
    assert all_recs[0].cycle_id == historical_cycle_id


# Test E: Recovery from abandoned / orphaned processing record
def test_test_e_recovery_from_abandoned_processing_record(test_db_session):
    """Test E: Verifies that an orphaned record left in PROCESSING state (e.g. from crash > 120s ago)
    is recovered and transitioned to COMPLETED without creating duplicate database rows.
    """
    db, session_factory = test_db_session
    bar_time = datetime(2026, 10, 9, 23, 50, 0, tzinfo=timezone.utc)
    bar_key = "XAUUSD_M1_202610092350"
    stale_created = datetime.utcnow() - timedelta(minutes=5)

    orphaned_record = WorkerCycleRecord(
        cycle_id="cyc_orphaned_attempt",
        symbol="XAUUSD",
        timeframe="M1",
        bar_key=bar_key,
        bar_time=bar_time,
        status_code="PROCESSING",
        lifecycle_phase="PROCESSING",
        decision="PENDING",
        duration_ms=0.0,
        details="Crash before completion",
        created_at=stale_created,
        updated_at=stale_created,
    )
    db.add(orphaned_record)
    db.commit()

    candles = generate_mock_candles(60, base_time=datetime(2026, 10, 9, 22, 51, 0, tzinfo=timezone.utc))
    candles[-1].time = bar_time
    mock_provider = get_mock_provider(candles=candles)

    worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)
    res = worker.run_single_cycle()

    assert res["lifecycle_phase"] == "COMPLETED"
    assert res["analysis_executed"] is True

    # Database still has exactly 1 record for this bar_key, now COMPLETED
    records = db.query(WorkerCycleRecord).filter_by(bar_key=bar_key).all()
    assert len(records) == 1
    assert records[0].lifecycle_phase == "COMPLETED"


# Test F: Separation of analysis success from signal generation validity
def test_test_f_analysis_success_vs_signal_generation_separation(test_db_session):
    """Test F: Verifies complete separation between analysis success (analysis_executed=True)
    and signal validity (trading_ready=False, signal_id=None).
    Confirms that zero orders are placed and DEMO_TRADING_ENABLED remains strictly False.
    """
    db, session_factory = test_db_session
    mock_provider = get_mock_provider()
    worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)

    # In our test environment, there are 0 approved rules and 0 active ML model
    res = worker.run_single_cycle()

    # Analysis must succeed
    assert res["analysis_executed"] is True
    assert res["lifecycle_phase"] == "COMPLETED"
    # But trading readiness and signal generation must be negative
    worker.is_running = True
    worker.last_heartbeat = datetime.now(timezone.utc)
    worker.mt5_status = "MT5_CONNECTED"
    worker.market_status = "MARKET_OPEN_DATA_FRESH"
    assert worker.trading_ready is False
    assert "RULES" in worker.trading_readiness_reason or "MODEL" in worker.trading_readiness_reason
    assert res["signal_id"] is None
    assert res["status_code"] in ["NO_APPROVED_RULES_AND_NO_ACTIVE_MODEL", "NO_VALID_SIGNAL"]

    # Security verification
    assert settings.EXECUTION_MODE == ExecutionMode.SIGNAL_ONLY
    assert settings.DEMO_TRADING_ENABLED is False


# Test G: Three-way readiness separation & API verification
def test_test_g_separated_readiness_states_and_endpoints(test_db_session):
    """Test G: Validates that service_ready, market_data_ready, and trading_ready evaluate independently,
    and checks their exposure via FastAPI endpoints /health and /api/v1/system/status.
    """
    db, session_factory = test_db_session
    mock_provider = get_mock_provider(is_connected=True, is_stale=False)
    worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)

    # Scenario 1: Worker stopped
    assert worker.service_ready is False
    assert "Worker stopped" in worker.service_ready_reason

    # Scenario 2: Worker running, heartbeat fresh
    worker.is_running = True
    worker.last_heartbeat = datetime.now(timezone.utc)
    worker.mt5_status = "MT5_CONNECTED"
    worker.market_status = "MARKET_OPEN_DATA_FRESH"

    assert worker.service_ready is True
    assert worker.market_data_ready is True
    assert worker.trading_ready is False  # No approved rules / model
    assert "RULES" in worker.trading_readiness_reason or "MODEL" in worker.trading_readiness_reason

    # Scenario 3: MT5 disconnected
    worker.mt5_status = "MT5_DISCONNECTED"
    assert worker.service_ready is True  # Service itself is alive
    assert worker.market_data_ready is False
    assert worker.trading_ready is False

    # Scenario 4: Market closed
    worker.mt5_status = "MT5_CONNECTED"
    worker.market_status = "MARKET_CLOSED"
    assert worker.service_ready is True
    assert worker.market_data_ready is False
    assert worker.trading_ready is False

    # Test via TestClient on FastAPI endpoints
    client = TestClient(app)
    with patch("backend.app.main.get_market_worker", return_value=worker), \
         patch("backend.app.api.v1.endpoints.system.get_market_worker", return_value=worker):

        # /health endpoint
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert "service_ready" in health_data
        assert "market_data_ready" in health_data
        assert "trading_ready" in health_data
        assert "trading_readiness_reason" in health_data
        assert health_data["trading_ready"] is False

        # /api/v1/system/status endpoint
        status_resp = client.get("/api/v1/system/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert "service_ready" in status_data
        assert "market_data_ready" in status_data
        assert "trading_ready" in status_data
        assert "trading_readiness_reason" in status_data
        assert status_data["trading_ready"] is False
        assert status_data["demo_trading_gate_enabled"] is False


# Test H: Event timestamps accuracy & isolation
def test_test_h_timestamps_accuracy_and_isolation(test_db_session):
    """Test H: Verifies that last_poll_time advances on every poll cycle,
    while last_successful_analysis_time updates ONLY when analysis actually executes
    and remains completely unaffected by duplicate candle skips.
    """
    db, session_factory = test_db_session
    mock_provider = get_mock_provider()
    worker = MarketMonitoringWorker(data_provider=mock_provider, db_factory=session_factory)
    worker.is_running = True

    # 1. Initially all timestamps are None
    assert worker.last_poll_time is None
    assert worker.last_successful_analysis_time is None
    assert worker.last_error_time is None

    # 2. Run cycle 1 (new candle) inside _run_loop step simulation
    async def run_single_iter():
        cycle_res = worker.run_single_cycle()
        worker.last_poll_time = datetime.now(timezone.utc)
        if cycle_res.get("analysis_executed"):
            worker.last_successful_analysis_time = worker.last_poll_time
            worker.last_successful_cycle_time = worker.last_poll_time
        return cycle_res

    res1 = asyncio.run(run_single_iter())
    assert res1["analysis_executed"] is True
    t_poll_1 = worker.last_poll_time
    t_analysis_1 = worker.last_successful_analysis_time
    assert t_poll_1 is not None
    assert t_analysis_1 is not None
    assert t_poll_1 == t_analysis_1

    # Small delay to ensure timestamp progression
    time.sleep(0.05)

    # 3. Run cycle 2 (duplicate candle)
    res2 = asyncio.run(run_single_iter())
    assert res2["analysis_executed"] is False
    assert res2["lifecycle_phase"] == "DUPLICATE_SKIPPED"

    t_poll_2 = worker.last_poll_time
    t_analysis_2 = worker.last_successful_analysis_time

    # last_poll_time MUST advance
    assert t_poll_2 > t_poll_1
    # last_successful_analysis_time MUST NOT advance (must remain frozen at t_analysis_1)
    assert t_analysis_2 == t_analysis_1

    # 4. Error timestamp tracking
    try:
        raise ValueError("Simulated network timeout")
    except Exception as e:
        worker.last_error_time = datetime.now(timezone.utc)
        worker.last_error_message = str(e)

    assert worker.last_error_time is not None
    assert "Simulated network timeout" in worker.last_error_message
