"""Tests for SIGNAL_ONLY execution mode, Idempotency, and Rule gating."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.db.base import Base
from backend.app.models.rule import TradingRule
from backend.app.models.signal import TradeSignal
from backend.app.models.order import DemoOrder
from backend.app.services.market_data.mock_provider import MockMarketDataProvider
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.signal.manager import SignalManager


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


def test_signal_only_mode_dispatches_zero_orders(db_session):
    """CRITICAL SAFETY TEST:
    Generating a signal in SIGNAL_ONLY mode MUST produce a TradeSignal in the DB
    but MUST NEVER create any DemoOrder or dispatch broker orders.
    """
    provider = MockMarketDataProvider(is_demo_account=True)
    manager = SignalManager(db_session, provider)

    # Add an approved entry rule
    rule = TradingRule(
        rule_id="rule_rsi_oversold",
        name="Buy on RSI Oversold",
        category="entry",
        description="Buy when RSI is below 70 (forced for testing)",
        conditions=[{"indicator": "RSI", "operator": "<", "threshold": 70.0}],
        parameters={},
        symbol_scope=["EURUSD"],
        timeframe_scope=["M15"],
        invalidation_conditions=[],
        source_pages=[],
        assumptions=[],
        ambiguities=[],
        approval_status="approved",  # Approved
    )
    db_session.add(rule)
    db_session.commit()

    sig = manager.process_latest_bar("EURUSD", TimeFrame.M15)
    assert sig is not None
    assert sig.direction == "BUY"
    assert sig.execution_mode == "SIGNAL_ONLY"

    # Verify TradeSignal exists in DB
    db_sig = db_session.query(TradeSignal).filter_by(signal_id=sig.signal_id).first()
    assert db_sig is not None

    # Verify DemoOrder table is COMPLETELY EMPTY
    orders_count = db_session.query(DemoOrder).count()
    assert orders_count == 0, "Orders were dispatched in SIGNAL_ONLY mode! Security violation!"


def test_signal_idempotency_prevents_duplicate_signals(db_session):
    """Verify that multiple evaluation cycles on the same bar produce the exact same signal
    without creating duplicate records.
    """
    provider = MockMarketDataProvider(is_demo_account=True)
    manager = SignalManager(db_session, provider)

    rule = TradingRule(
        rule_id="rule_rsi_oversold",
        name="Buy on RSI Oversold",
        category="entry",
        description="Buy rule",
        conditions=[{"indicator": "RSI", "operator": "<", "threshold": 70.0}],
        parameters={},
        symbol_scope=["EURUSD"],
        timeframe_scope=["M15"],
        invalidation_conditions=[],
        source_pages=[],
        assumptions=[],
        ambiguities=[],
        approval_status="approved",
    )
    db_session.add(rule)
    db_session.commit()

    # First run
    sig1 = manager.process_latest_bar("EURUSD", TimeFrame.M15)
    assert sig1 is not None

    # Flip the rule's direction, then evaluate the same completed bar again.
    # Idempotency must prevent a BUY and SELL signal for the same candle.
    rule.name = "Sell on RSI Oversold"
    db_session.commit()

    sig2 = manager.process_latest_bar("EURUSD", TimeFrame.M15)
    assert sig2 is not None

    # The earlier signal wins for this bar, regardless of a later direction change.
    assert sig1.signal_id == sig2.signal_id
    assert sig1.idempotency_key == sig2.idempotency_key
    assert sig1.direction == sig2.direction == "BUY"

    # Total signal rows in database must be strictly 1
    total_signals = db_session.query(TradeSignal).count()
    assert total_signals == 1


def test_unapproved_rules_do_not_generate_signals(db_session):
    """Verify that rules in 'pending' or 'rejected' status produce NO signals."""
    provider = MockMarketDataProvider(is_demo_account=True)
    manager = SignalManager(db_session, provider)

    pending_rule = TradingRule(
        rule_id="rule_pending_01",
        name="Buy Pending",
        category="entry",
        description="Unapproved rule",
        conditions=[{"indicator": "RSI", "operator": "<", "threshold": 90.0}],
        parameters={},
        symbol_scope=["EURUSD"],
        timeframe_scope=["M15"],
        invalidation_conditions=[],
        source_pages=[],
        assumptions=[],
        ambiguities=["Strong trend undefined"],
        approval_status="pending",  # NOT APPROVED
    )
    db_session.add(pending_rule)
    db_session.commit()

    sig = manager.process_latest_bar("EURUSD", TimeFrame.M15)
    # Must be None since rule is pending and no active model exists
    assert sig is None
