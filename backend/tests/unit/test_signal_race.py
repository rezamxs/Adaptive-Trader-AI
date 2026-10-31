"""Item 17 regression: a lost idempotency race between the worker and the API
must surface as the EXISTING record, never a 500 / WORKER_EXCEPTION.

The race window is reproduced deterministically: the losing manager's
idempotency pre-check runs BEFORE the winner's commit (simulated by making
the TradeSignal pre-check query return an empty result), then the real
IntegrityError path in SignalManager.process_latest_bar is exercised against
a real two-connection SQLite database. Nothing inside the logic under test
is mocked.
"""
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from backend.app.db.base import Base
import backend.app.models  # noqa: F401 - register all tables
from backend.app.models.rule import TradingRule
from backend.app.models.signal import TradeSignal
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.market_data.mock_provider import MockMarketDataProvider
from backend.app.services.signal.manager import SignalManager


@pytest.fixture
def shared_db(tmp_path):
    """File-backed SQLite shared by two independent connections (a real
    two-connection race, not one shared session)."""
    db_file = tmp_path / "race.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def approved_rule():
    def _make(session):
        rule = TradingRule(
            rule_id="rule_rsi_oversold_race",
            name="Buy on RSI Oversold",
            category="entry",
            description="Buy when RSI is below 70 (race test)",
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
        session.add(rule)
        session.commit()
        return rule
    return _make


def _manager(session):
    return SignalManager(session, MockMarketDataProvider(is_demo_account=True))


def test_lost_race_returns_existing_signal_not_500(shared_db, approved_rule, monkeypatch):
    session_a = shared_db()
    session_b = shared_db()
    approved_rule(session_a)

    # Winner: manager A records the signal for the current bar.
    manager_a = _manager(session_a)
    sig_a = manager_a.process_latest_bar("EURUSD", TimeFrame.M15)
    assert sig_a is not None and sig_a.risk_status in ("APPROVED", "REJECTED")

    # Loser: manager B's idempotency pre-check ran BEFORE A's commit landed
    # (the race window), so it sees nothing and proceeds to insert. Only the
    # FIRST TradeSignal query (the pre-check) is intercepted; every other
    # query — including the post-rollback lookup — hits the real database.
    manager_b = _manager(session_b)
    intercepted = {"used": False}
    real_query = manager_b.db.query

    class _EmptyResult:
        def filter(self, *args, **kwargs):
            return self

        def first(self):
            return None

        def all(self):
            return []

    def racing_query(*entities, **kwargs):
        if (
            not intercepted["used"]
            and entities
            and entities[0] is TradeSignal
        ):
            intercepted["used"] = True
            return _EmptyResult()
        return real_query(*entities, **kwargs)

    monkeypatch.setattr(manager_b.db, "query", racing_query)

    sig_b = manager_b.process_latest_bar("EURUSD", TimeFrame.M15)

    # The loser must return the WINNER'S committed record — no 500, no
    # WORKER_EXCEPTION, no duplicate row.
    assert intercepted["used"], "the race window must have been exercised"
    assert sig_b is not None
    assert sig_b.signal_id == sig_a.signal_id
    total = session_b.query(TradeSignal).count()
    assert total == 1, f"race must not create a duplicate signal, found {total}"


def test_guard_re_raises_when_no_winner_exists(shared_db, approved_rule, monkeypatch):
    """If the IntegrityError is NOT caused by the idempotency race (no matching
    record exists after rollback), the guard must re-raise instead of
    swallowing the failure."""
    session_a = shared_db()
    approved_rule(session_a)

    manager_a = _manager(session_a)
    intercepted = {"used": False}
    real_query = manager_a.db.query

    class _EmptyResult:
        def filter(self, *args, **kwargs):
            return self

        def first(self):
            return None

        def all(self):
            return []

    def always_empty_trade_signal_query(*entities, **kwargs):
        if entities and entities[0] is TradeSignal:
            intercepted["used"] = True
            return _EmptyResult()
        return real_query(*entities, **kwargs)

    monkeypatch.setattr(manager_a.db, "query", always_empty_trade_signal_query)

    # Force the commit to fail with an IntegrityError that is NOT resolved by
    # the race (the intercepted idempotency lookup keeps reporting no winner):
    # the guard must roll back and RE-RAISE, never swallow the failure.
    def failing_commit():
        raise IntegrityError(
            "INSERT INTO trade_signals ...", {},
            Exception("forced non-race unique violation"))

    monkeypatch.setattr(manager_a.db, "commit", failing_commit)

    with pytest.raises(IntegrityError):
        manager_a.process_latest_bar("EURUSD", TimeFrame.M15)

    assert intercepted["used"], "the TradeSignal lookup guard must have run"
