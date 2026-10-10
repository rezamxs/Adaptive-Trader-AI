"""Security and Safety Gate Tests: Risk Engine and Circuit Breaker."""
import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.db.base import Base
from backend.app.services.market_data.base import Quote, SymbolInfo, AccountDetails
from backend.app.services.risk.circuit_breaker import EmergencyStopManager
from backend.app.services.risk.risk_engine import RiskEngine
from backend.app.services.risk.position_sizer import PositionSizer


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
def mock_account():
    return AccountDetails(
        login_masked="******83",
        server="YourBroker-MT5",
        currency="USD",
        leverage=40,
        balance=10000.0,
        equity=10000.0,
        margin=0.0,
        free_margin=10000.0,
        trade_mode=0,
        is_demo=True,
    )


@pytest.fixture
def mock_symbol_info():
    return SymbolInfo(
        name="EURUSD",
        digits=5,
        point=0.00001,
        spread=15,
        tick_size=0.00001,
        tick_value=1.0,
        volume_min=0.01,
        volume_max=10.0,
        volume_step=0.01,
        trade_contract_size=100000.0,
        currency_base="EUR",
        currency_profit="USD",
        currency_margin="EUR",
    )


def test_emergency_stop_persists_across_restarts(db_session):
    """Verify that emergency stop state survives instance recreation / server restart."""
    mgr1 = EmergencyStopManager(db_session)
    assert mgr1.is_active() is False

    # Activate in instance 1
    mgr1.activate("Manual test trigger", actor="TRADER_ADMIN")
    assert mgr1.is_active() is True

    # Create new instance (simulating server reboot)
    mgr2 = EmergencyStopManager(db_session)
    assert mgr2.is_active() is True
    assert "Manual test trigger" in mgr2.get_status()[1]

    # Deactivate in instance 2
    mgr2.deactivate(actor="TRADER_ADMIN", notes="System cleared")
    assert mgr2.is_active() is False

    # Check instance 1 reflects deactivation
    assert mgr1.is_active() is False


def test_risk_engine_blocks_on_emergency_stop(db_session, mock_account, mock_symbol_info):
    """Verify RiskEngine strictly rejects any proposal when Emergency Stop is active."""
    risk_engine = RiskEngine(db_session)
    risk_engine.emergency_stop.activate("Emergency maintenance", actor="SYS")

    quote = Quote("EURUSD", datetime.now(timezone.utc), 1.0850, 1.08515, 15, is_stale=False)
    approved, lots, reason = risk_engine.assess_signal(
        symbol="EURUSD",
        direction="BUY",
        entry_price=1.0850,
        stop_loss=1.0800,
        take_profit=1.0950,
        account=mock_account,
        symbol_info=mock_symbol_info,
        quote=quote,
    )

    assert approved is False
    assert "REJECTED_EMERGENCY_STOP" in reason


def test_risk_engine_blocks_high_spread(db_session, mock_account, mock_symbol_info):
    """Verify RiskEngine rejects proposal when spread exceeds threshold (35 points)."""
    risk_engine = RiskEngine(db_session)
    # Quote with 50 points spread
    quote = Quote("EURUSD", datetime.now(timezone.utc), 1.0850, 1.0855, 50, is_stale=False)

    approved, lots, reason = risk_engine.assess_signal(
        symbol="EURUSD",
        direction="BUY",
        entry_price=1.0850,
        stop_loss=1.0800,
        take_profit=1.0950,
        account=mock_account,
        symbol_info=mock_symbol_info,
        quote=quote,
    )

    assert approved is False
    assert "REJECTED_HIGH_SPREAD" in reason


def test_risk_engine_blocks_stale_data(db_session, mock_account, mock_symbol_info):
    """Verify RiskEngine rejects proposal when quote data is flagged as stale."""
    risk_engine = RiskEngine(db_session)
    quote = Quote("EURUSD", datetime.now(timezone.utc), 1.0850, 1.08515, 15, is_stale=True)

    approved, lots, reason = risk_engine.assess_signal(
        symbol="EURUSD",
        direction="BUY",
        entry_price=1.0850,
        stop_loss=1.0800,
        take_profit=1.0950,
        account=mock_account,
        symbol_info=mock_symbol_info,
        quote=quote,
    )

    assert approved is False
    assert "REJECTED_STALE_DATA" in reason


def test_risk_engine_blocks_on_daily_drawdown_and_trips_breaker(db_session, mock_account, mock_symbol_info):
    """Verify daily drawdown breach auto-triggers Emergency Stop circuit breaker."""
    risk_engine = RiskEngine(db_session)
    quote = Quote("EURUSD", datetime.now(timezone.utc), 1.0850, 1.08515, 15, is_stale=False)

    # Daily loss = 3.5% (exceeds limit 3.0%)
    approved, lots, reason = risk_engine.assess_signal(
        symbol="EURUSD",
        direction="BUY",
        entry_price=1.0850,
        stop_loss=1.0800,
        take_profit=1.0950,
        account=mock_account,
        symbol_info=mock_symbol_info,
        quote=quote,
        daily_loss_percent=3.5,
    )

    assert approved is False
    assert "REJECTED_DAILY_DRAWDOWN" in reason
    # Breaker MUST be automatically tripped!
    assert risk_engine.emergency_stop.is_active() is True
