"""Integration tests for Database models and operations."""
import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.db.base import Base
from backend.app.models import (
    Document,
    TradingRule,
    DatasetVersion,
    MLModelRecord,
    BacktestRun,
    TradeSignal,
    DemoOrder,
    EmergencyStopState,
    AuditLog,
)


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_create_document_and_rule(in_memory_db):
    """Test creating a document and linked trading rule."""
    doc = Document(
        document_id="doc_test_001",
        file_name="strategy_guide.pdf",
        file_type="pdf",
        file_size_bytes=10240,
        source_description="Manual trading playbook",
        processing_status="completed",
        pages_count=5,
        raw_text="Rule 1: Buy when RSI < 30 and trend is strong.",
    )
    in_memory_db.add(doc)
    in_memory_db.commit()

    rule = TradingRule(
        rule_id="rule_rsi_oversold_01",
        name="RSI Oversold Entry",
        category="entry",
        description="Buy when RSI dips below 30",
        conditions=[{"indicator": "RSI", "operator": "<", "threshold": 30}],
        parameters={"rsi_period": 14},
        symbol_scope=["EURUSD"],
        timeframe_scope=["M15"],
        invalidation_conditions=["Price breaks below recent swing low"],
        source_document="doc_test_001",
        source_pages=[1, 2],
        assumptions=["Normal market volatility"],
        ambiguities=["'Trend is strong' is subjective and requires moving average filter definition"],
        confidence=0.85,
        approval_status="pending",
    )
    in_memory_db.add(rule)
    in_memory_db.commit()

    # Query back
    saved_rule = in_memory_db.query(TradingRule).filter_by(rule_id="rule_rsi_oversold_01").first()
    assert saved_rule is not None
    assert saved_rule.approval_status == "pending"
    assert len(saved_rule.conditions) == 1
    assert "Trend is strong" in saved_rule.ambiguities[0]


def test_emergency_stop_lifecycle(in_memory_db):
    """Test setting, persisting, and checking emergency stop state."""
    stop_state = EmergencyStopState(
        is_active=True,
        activated_by="RISK_ENGINE",
        reason="Daily drawdown threshold exceeded 3%",
    )
    in_memory_db.add(stop_state)
    in_memory_db.commit()

    latest = in_memory_db.query(EmergencyStopState).order_by(EmergencyStopState.id.desc()).first()
    assert latest.is_active is True
    assert latest.reason == "Daily drawdown threshold exceeded 3%"


def test_trade_signal_idempotency_constraint(in_memory_db):
    """Test that idempotency key prevents duplicate signals."""
    sig1 = TradeSignal(
        signal_id="sig_001",
        idempotency_key="EURUSD_M15_20261010_1200_BUY",
        symbol="EURUSD",
        timeframe="M15",
        direction="BUY",
        entry_price=1.0850,
        stop_loss=1.0820,
        take_profit=1.0910,
        rules_triggered=["rule_rsi_oversold_01"],
        features_snapshot={"rsi": 28.5},
        bar_time=datetime.utcnow(),
    )
    in_memory_db.add(sig1)
    in_memory_db.commit()

    # Attempt duplicate
    sig2 = TradeSignal(
        signal_id="sig_002",
        idempotency_key="EURUSD_M15_20261010_1200_BUY",
        symbol="EURUSD",
        timeframe="M15",
        direction="BUY",
        entry_price=1.0850,
        stop_loss=1.0820,
        take_profit=1.0910,
        rules_triggered=["rule_rsi_oversold_01"],
        features_snapshot={"rsi": 28.5},
        bar_time=datetime.utcnow(),
    )
    in_memory_db.add(sig2)
    with pytest.raises(Exception):  # IntegrityError
        in_memory_db.commit()
    in_memory_db.rollback()
