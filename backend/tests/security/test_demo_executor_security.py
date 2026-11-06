"""Tests for DemoTradingExecutor Fail-Closed Security and Safety Gates."""
import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings, ExecutionMode
from backend.app.db.base import Base, utc_now_naive
from backend.app.models.signal import TradeSignal
from backend.app.models.order import DemoOrder
from backend.app.services.market_data.mock_provider import MockMarketDataProvider
from backend.app.services.execution.demo_executor import DemoTradingExecutor


@pytest.fixture(autouse=True)
def _session_filter_disabled_for_security_tests(monkeypatch):
    """These tests verify OTHER rejection semantics (risk volume, idempotency)
    at arbitrary wall-clock times. The session-hours gate is now wired into
    the dispatch path; its behavior is covered separately in
    test_risk_gates_wiring.py."""
    monkeypatch.setattr(settings, "ENFORCE_SESSION_FILTER", False)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_approved_signal(db_session):
    sig = TradeSignal(
        signal_id="sig_test_001",
        idempotency_key="EURUSD_M15_202610101200_BUY",
        symbol="EURUSD",
        timeframe="M15",
        direction="BUY",
        strength=0.85,
        entry_price=1.0850,
        stop_loss=1.0800,
        take_profit=1.0925,
        rules_triggered=["rule_001"],
        features_snapshot={},
        risk_status="APPROVED",
        calculated_volume=0.10,
        execution_mode="DEMO_TRADING",
        status="GENERATED",
        bar_time=utc_now_naive(),
    )
    db_session.add(sig)
    db_session.commit()
    return sig


def test_executor_blocked_when_in_signal_only_mode(db_session, sample_approved_signal, monkeypatch):
    """Verify executor blocks order when EXECUTION_MODE is SIGNAL_ONLY (default)."""
    monkeypatch.setattr(settings, "EXECUTION_MODE", ExecutionMode.SIGNAL_ONLY)
    monkeypatch.setattr(settings, "DEMO_TRADING_ENABLED", True)

    provider = MockMarketDataProvider(is_demo_account=True)
    executor = DemoTradingExecutor(db_session, provider)

    success, order, msg = executor.execute_demo_signal(sample_approved_signal)
    assert success is False
    assert "Safety Gate Blocked" in msg
    assert "SIGNAL_ONLY" in msg


def test_executor_blocked_when_demo_gate_is_false(db_session, sample_approved_signal, monkeypatch):
    """Verify executor blocks order when DEMO_TRADING_ENABLED gate is False."""
    monkeypatch.setattr(settings, "EXECUTION_MODE", ExecutionMode.DEMO_TRADING)
    monkeypatch.setattr(settings, "DEMO_TRADING_ENABLED", False)

    provider = MockMarketDataProvider(is_demo_account=True)
    executor = DemoTradingExecutor(db_session, provider)

    success, order, msg = executor.execute_demo_signal(sample_approved_signal)
    assert success is False
    assert "DEMO_TRADING_ENABLED gate is False" in msg


def test_executor_blocked_when_emergency_stop_is_active(db_session, sample_approved_signal, monkeypatch):
    """Verify executor blocks order when Emergency Stop is active."""
    monkeypatch.setattr(settings, "EXECUTION_MODE", ExecutionMode.DEMO_TRADING)
    monkeypatch.setattr(settings, "DEMO_TRADING_ENABLED", True)

    provider = MockMarketDataProvider(is_demo_account=True)
    executor = DemoTradingExecutor(db_session, provider)
    executor.emergency_stop.activate("Market volatility spike", actor="RISK_SYSTEM")

    success, order, msg = executor.execute_demo_signal(sample_approved_signal)
    assert success is False
    assert "Emergency Stop is currently ACTIVE" in msg


def test_executor_blocks_on_real_account(db_session, sample_approved_signal, monkeypatch):
    """CRITICAL SECURITY TEST:
    Verify executor halts execution and fails closed if account is NOT a DEMO account.
    """
    monkeypatch.setattr(settings, "EXECUTION_MODE", ExecutionMode.DEMO_TRADING)
    monkeypatch.setattr(settings, "DEMO_TRADING_ENABLED", True)

    # Provider with REAL account (trade_mode = 2)
    provider = MockMarketDataProvider(is_demo_account=False)
    executor = DemoTradingExecutor(db_session, provider)

    success, order, msg = executor.execute_demo_signal(sample_approved_signal)
    assert success is False
    assert "Account could not be verified as a genuine DEMO account" in msg


def test_executor_rejects_signal_when_stop_loss_is_invalid_at_current_price(
    db_session, sample_approved_signal, monkeypatch
):
    """A past risk approval cannot bypass current-price SL/TP validation."""
    monkeypatch.setattr(settings, "EXECUTION_MODE", ExecutionMode.DEMO_TRADING)
    monkeypatch.setattr(settings, "DEMO_TRADING_ENABLED", True)

    sample_approved_signal.stop_loss = 1.0860  # Above the current BUY entry at 1.08515.
    db_session.commit()

    provider = MockMarketDataProvider(is_demo_account=True)
    executor = DemoTradingExecutor(db_session, provider)
    success, order, msg = executor.execute_demo_signal(sample_approved_signal)

    assert success is False
    assert order is None
    assert "REJECTED_INVALID_SL" in msg
    assert db_session.query(DemoOrder).count() == 0


def test_executor_rejects_volume_that_exceeds_current_risk_limit(
    db_session, sample_approved_signal, monkeypatch
):
    """A stored approval cannot send more volume than current risk permits."""
    monkeypatch.setattr(settings, "EXECUTION_MODE", ExecutionMode.DEMO_TRADING)
    monkeypatch.setattr(settings, "DEMO_TRADING_ENABLED", True)

    # The wider stop makes 0.10 lots exceed the current risk budget.
    sample_approved_signal.stop_loss = 1.0000
    db_session.commit()

    provider = MockMarketDataProvider(is_demo_account=True)
    executor = DemoTradingExecutor(db_session, provider)
    success, order, msg = executor.execute_demo_signal(sample_approved_signal)

    assert success is False
    assert order is None
    assert "current risk permits at most" in msg
    assert db_session.query(DemoOrder).count() == 0


def test_order_idempotency_prevents_duplicate_dispatch(db_session, sample_approved_signal, monkeypatch):
    """Verify order idempotency prevents duplicate fills on same signal."""
    monkeypatch.setattr(settings, "EXECUTION_MODE", ExecutionMode.DEMO_TRADING)
    monkeypatch.setattr(settings, "DEMO_TRADING_ENABLED", True)

    provider = MockMarketDataProvider(is_demo_account=True)
    executor = DemoTradingExecutor(db_session, provider)

    # First dispatch
    success1, order1, msg1 = executor.execute_demo_signal(sample_approved_signal)
    assert success1 is True
    assert order1 is not None

    # Immediate second dispatch on same signal
    success2, order2, msg2 = executor.execute_demo_signal(sample_approved_signal)
    assert success2 is False
    assert "Duplicate order prevented" in msg2

    # Database count must be exactly 1
    assert db_session.query(DemoOrder).count() == 1
