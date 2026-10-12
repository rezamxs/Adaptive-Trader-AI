"""Reject stale/future quote timestamps even when the provider flag is wrong."""
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from backend.app.db.base import Base
from backend.app.services.market_data.base import AccountDetails, Quote, SymbolInfo
from backend.app.services.risk.risk_engine import RiskEngine


def _case():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    account = AccountDetails(
        login_masked="******11", server="Test-Demo", currency="USD", leverage=100,
        balance=10000.0, equity=10000.0, margin=0.0, free_margin=10000.0,
        trade_mode=0, is_demo=True,
    )
    info = SymbolInfo(
        name="EURUSD", digits=5, point=0.00001, spread=10, tick_size=0.00001,
        tick_value=1.0, volume_min=0.01, volume_max=10.0, volume_step=0.01,
        trade_contract_size=100000.0, currency_base="EUR", currency_profit="USD",
        currency_margin="EUR",
    )
    return session, account, info


def _assess(session, account, info, quote):
    return RiskEngine(session).assess_signal(
        symbol="EURUSD", direction="BUY", entry_price=1.1000, stop_loss=1.0990,
        take_profit=1.1020, account=account, symbol_info=info, quote=quote,
    )


def test_stale_timestamp_rejected_even_if_flag_is_false():
    session, account, info = _case()
    try:
        quote = Quote(
            symbol="EURUSD",
            time=datetime.now(timezone.utc) - timedelta(seconds=settings.WORKER_STALE_TICK_SECONDS + 30),
            bid=1.1000, ask=1.1001, spread_points=10, is_stale=False,
        )
        approved, lots, reason = _assess(session, account, info, quote)
        assert not approved and lots == 0.0
        assert "REJECTED_STALE_DATA" in reason
    finally:
        session.close()


def test_future_timestamp_rejected():
    session, account, info = _case()
    try:
        quote = Quote(
            symbol="EURUSD", time=datetime.now(timezone.utc) + timedelta(seconds=30),
            bid=1.1000, ask=1.1001, spread_points=10, is_stale=False,
        )
        approved, lots, reason = _assess(session, account, info, quote)
        assert not approved and lots == 0.0
        assert "REJECTED_FUTURE_DATA" in reason
    finally:
        session.close()


def test_naive_timestamp_rejected():
    session, account, info = _case()
    try:
        quote = Quote(
            symbol="EURUSD", time=datetime.now(),
            bid=1.1000, ask=1.1001, spread_points=10, is_stale=False,
        )
        approved, lots, reason = _assess(session, account, info, quote)
        assert not approved and lots == 0.0
        assert "REJECTED_INVALID_QUOTE_TIME" in reason
    finally:
        session.close()



def test_missing_account_risk_context_is_rejected_by_default():
    session, account, info = _case()
    try:
        quote = Quote(
            symbol="EURUSD", time=datetime.now(timezone.utc),
            bid=1.1000, ask=1.1001, spread_points=10, is_stale=False,
        )
        approved, lots, reason = _assess(session, account, info, quote)
        assert not approved and lots == 0.0
        assert "REJECTED_RISK_CONTEXT_UNAVAILABLE" in reason
    finally:
        session.close()
