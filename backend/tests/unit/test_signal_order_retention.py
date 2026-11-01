"""Minor item 16 extension: retention windows for trade_signals and demo_orders.

Same contract as the worker_cycle/audit_log windows: a window of 0 (default)
DISABLES pruning — trade history is never deleted without an operator
explicitly setting a window.
"""
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.app.models  # noqa: F401 - register all ORM tables on Base
from backend.app.db.base import Base
from backend.app.models.order import DemoOrder
from backend.app.models.signal import TradeSignal
from backend.app.services.worker.market_worker import prune_operational_records


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session
    session.close()
    engine.dispose()


def _add_old_signal(session, signal_id: str, age_days: int):
    session.add(TradeSignal(
        signal_id=signal_id,
        idempotency_key=f"idem_{signal_id}",
        symbol="XAUUSD", timeframe="M1", direction="HOLD",
        entry_price=1.0, stop_loss=0.9, take_profit=1.2,
        bar_time=datetime.utcnow() - timedelta(days=age_days),
        created_at=datetime.utcnow() - timedelta(days=age_days),
    ))


def _add_old_order(session, order_id: str, age_days: int):
    session.add(DemoOrder(
        order_id=order_id,
        signal_id="sig_ref",
        idempotency_key=f"idem_{order_id}",
        symbol="XAUUSD", order_type="BUY",
        volume=0.01, requested_price=1.0, stop_loss=0.9, take_profit=1.2,
        status="FILLED",
        created_at=datetime.utcnow() - timedelta(days=age_days),
    ))


def test_default_windows_delete_nothing(db_session):
    _add_old_signal(db_session, "sig_keep", 400)
    _add_old_order(db_session, "ord_keep", 400)
    db_session.commit()

    deleted = prune_operational_records(db_session)

    assert deleted["trade_signals"] == 0
    assert deleted["demo_orders"] == 0
    assert db_session.query(TradeSignal).count() == 1
    assert db_session.query(DemoOrder).count() == 1


def test_signals_and_orders_pruned_only_older_than_window(db_session):
    now = datetime.utcnow()
    _add_old_signal(db_session, "sig_old", 40)
    _add_old_order(db_session, "ord_old", 40)
    fresh_signal = TradeSignal(
        signal_id="sig_new",
        idempotency_key="idem_sig_new",
        symbol="XAUUSD", timeframe="M1", direction="HOLD",
        entry_price=1.0, stop_loss=0.9, take_profit=1.2,
        bar_time=now - timedelta(days=1),
        created_at=now - timedelta(days=1),
    )
    db_session.add(fresh_signal)
    db_session.add(DemoOrder(
        order_id="ord_new",
        signal_id="sig_ref",
        idempotency_key="idem_ord_new",
        symbol="XAUUSD", order_type="BUY",
        volume=0.01, requested_price=1.0, stop_loss=0.9, take_profit=1.2,
        status="FILLED",
        created_at=now - timedelta(days=1),
    ))
    db_session.commit()

    deleted = prune_operational_records(db_session, trade_signal_days=30, demo_order_days=30)

    assert deleted["trade_signals"] == 1
    assert deleted["demo_orders"] == 1
    remaining_signals = {s.signal_id for s in db_session.query(TradeSignal).all()}
    remaining_orders = {o.order_id for o in db_session.query(DemoOrder).all()}
    assert remaining_signals == {"sig_new"}
    assert remaining_orders == {"ord_new"}
