"""TDD tests for audit fix H6: DemoOrder lifecycle must be PENDING -> FILLED/REJECTED.

The order record must be committed BEFORE mt5.order_send, so a crash between
dispatch and persistence can never orphan a live broker fill, and the
idempotency key is reserved before the broker is touched. The post-fill
commit must be guarded so a persistence failure cannot raise an unhandled
error while the order is already live at the broker.
"""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings, ExecutionMode
from backend.app.db.base import Base
from backend.app.models.order import DemoOrder
from backend.app.models.signal import TradeSignal
from backend.app.services.market_data.base import AccountDetails, Quote, SymbolInfo
from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider, MT5_AVAILABLE

# Item 24 (CI): this module builds REAL MetaTrader5.TradeOrder structseqs and
# monkeypatches the mt5 module — both require the Windows-only MetaTrader5
# package. On ubuntu CI the whole module skips instead of erroring at import.
pytestmark = pytest.mark.skipif(
    not MT5_AVAILABLE, reason="requires the Windows-only MetaTrader5 package"
)
from backend.app.services.execution import demo_executor
from backend.app.services.execution.demo_executor import DemoTradingExecutor


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
        signal_id="sig_h6_001",
        idempotency_key="XAUUSD_M15_202610240900_BUY",
        symbol="XAUUSD",
        timeframe="M15",
        direction="BUY",
        strength=0.85,
        entry_price=1900.0,
        stop_loss=1895.0,
        take_profit=1910.0,
        rules_triggered=["rule_h6"],
        features_snapshot={},
        risk_status="APPROVED",
        calculated_volume=0.10,
        execution_mode="DEMO_TRADING",
        status="GENERATED",
        bar_time=datetime.utcnow(),
    )
    db_session.add(sig)
    db_session.commit()
    return sig


class _FakeMT5Provider(MT5MarketDataProvider):
    """Looks like the real MT5 provider to the executor's isinstance gate,
    but every data method is a deterministic stub and no terminal is touched."""

    def __init__(self):  # bypass real provider's terminal init on purpose
        self._connected = True

    def connect(self):
        return True

    def is_connected(self):
        return True

    def get_account_info(self):
        return AccountDetails(
            login_masked="***",
            server="Demo-Server",
            currency="USD",
            leverage=100,
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            free_margin=10000.0,
            trade_mode=0,
            is_demo=True,
        )

    def get_current_quote(self, symbol):
        return Quote(symbol=symbol, time=datetime.now(timezone.utc), bid=1899.9, ask=1900.1, spread_points=20)

    def get_symbol_info(self, symbol):
        return SymbolInfo(
            name=symbol, digits=2, point=0.1, spread=20, tick_size=0.1, tick_value=1.0,
            volume_min=0.01, volume_max=10.0, volume_step=0.01,
            trade_contract_size=100.0, currency_base="XAU", currency_profit="USD", currency_margin="USD",
        )

    def get_open_positions_count(self):
        return 0

    def get_open_positions(self):
        return []  # no grid positions in these tests

    def get_daily_loss_percent(self):
        return 0.0

    def calculate_order_margin(self, *args, **kwargs):
        return 0.0


class _StubMT5Module:
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TRADE_RETCODE_DONE = 10009

    def __init__(self, order_send_impl):
        self._order_send_impl = order_send_impl
        # M5: deals evidence unavailable unless a test installs a stub — the
        # reconciler must then fall back to the order-row arithmetic.
        self.history_deals_get = lambda **kwargs: None

    def terminal_info(self):
        class _Term:
            trade_allowed = True
        return _Term()

    def order_send(self, request):
        return self._order_send_impl(request)


def _success_result(request):
    class _Res:
        retcode = 10009
        order = 123456
        volume = request["volume"]
        price = request["price"]
        comment = "done"
    return _Res()


def _enable_demo(monkeypatch):
    monkeypatch.setattr(settings, "EXECUTION_MODE", ExecutionMode.DEMO_TRADING)
    monkeypatch.setattr(settings, "DEMO_TRADING_ENABLED", True)


@pytest.fixture(autouse=True)
def _session_filter_disabled_for_dispatch_tests(monkeypatch):
    """Dispatch tests must be time-independent: the session-hours gate is
    wired into the production path (see test_risk_gates_wiring.py) and would
    otherwise reject entries outside 00:00-09:30 UTC."""
    monkeypatch.setattr(settings, "ENFORCE_SESSION_FILTER", False)


def test_pending_order_committed_before_order_send(db_session, sample_approved_signal, monkeypatch):
    _enable_demo(monkeypatch)
    observed = {}

    def order_send_impl(request):
        # At dispatch time the PENDING row must already be committed.
        observed["pending_at_send"] = (
            db_session.query(DemoOrder).filter_by(idempotency_key=sample_approved_signal.idempotency_key).first()
        )
        observed["pending_status"] = observed["pending_at_send"].status if observed["pending_at_send"] else None
        return _success_result(request)

    monkeypatch.setattr(demo_executor, "mt5", _StubMT5Module(order_send_impl))
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    success, order, msg = executor.execute_demo_signal(sample_approved_signal)

    assert success is True, msg
    assert observed["pending_at_send"] is not None, "no DemoOrder row committed before order_send"
    assert observed["pending_status"] == "PENDING"
    # Same physical record transitions PENDING -> FILLED; no second row created.
    assert order.status == "FILLED"
    assert order.order_id == observed["pending_at_send"].order_id
    assert db_session.query(DemoOrder).count() == 1


def test_rejected_order_updates_single_pending_record(db_session, sample_approved_signal, monkeypatch):
    _enable_demo(monkeypatch)

    def order_send_impl(request):
        class _Res:
            retcode = 10030
            comment = "Unsupported filling mode"
        return _Res()

    monkeypatch.setattr(demo_executor, "mt5", _StubMT5Module(order_send_impl))
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    success, order, msg = executor.execute_demo_signal(sample_approved_signal)

    assert success is False
    assert order.status == "REJECTED"
    assert "10030" in (order.error_message or "")
    assert db_session.query(DemoOrder).count() == 1, "rejection must update the PENDING row, not add a second one"


def test_post_fill_commit_failure_does_not_raise(db_session, sample_approved_signal, monkeypatch):
    _enable_demo(monkeypatch)
    commits = {"n": 0}
    real_commit = db_session.commit

    def flaky_commit():
        commits["n"] += 1
        real_commit()
        if commits["n"] >= 2:  # first commit = PENDING pre-send, second = post-fill update
            raise ConnectionError("simulated persistence outage")

    monkeypatch.setattr(db_session, "commit", flaky_commit)
    monkeypatch.setattr(demo_executor, "mt5", _StubMT5Module(lambda request: _success_result(request)))
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    # Must NOT raise: the fill is live at the broker; an unhandled exception here
    # would leave the operator without any response describing the live order.
    success, order, msg = executor.execute_demo_signal(sample_approved_signal)
    assert "persist" in msg.lower()


def test_idempotency_reserves_before_dispatch(db_session, sample_approved_signal, monkeypatch):
    _enable_demo(monkeypatch)
    send_calls = {"n": 0}

    def order_send_impl(request):
        send_calls["n"] += 1
        return _success_result(request)

    monkeypatch.setattr(demo_executor, "mt5", _StubMT5Module(order_send_impl))
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    success1, order1, msg1 = executor.execute_demo_signal(sample_approved_signal)
    assert success1 is True, msg1

    success2, order2, msg2 = executor.execute_demo_signal(sample_approved_signal)
    assert success2 is False
    assert "Duplicate order prevented" in msg2
    assert send_calls["n"] == 1, "second dispatch must never reach the broker"
    assert db_session.query(DemoOrder).count() == 1


def test_order_send_exception_leaves_outcome_unknown(db_session, sample_approved_signal, monkeypatch):
    """Final-safety 2: an order_send exception means the broker outcome is
    UNKNOWN (the request may have executed). The row must be EXECUTION_UNKNOWN,
    NOT REJECTED, and the idempotency key stays reserved."""
    _enable_demo(monkeypatch)

    def order_send_impl(request):
        raise RuntimeError("IPC timeout: terminal is not responding")

    monkeypatch.setattr(demo_executor, "mt5", _StubMT5Module(order_send_impl))
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    success, order, msg = executor.execute_demo_signal(sample_approved_signal)

    assert success is False
    assert order is not None
    assert order.status == "EXECUTION_UNKNOWN"
    assert "IPC timeout" in (order.error_message or "")
    assert "UNKNOWN" in msg
    assert db_session.query(DemoOrder).filter_by(status="REJECTED").count() == 0


def test_unknown_outcome_blocks_retry_without_second_dispatch(db_session, sample_approved_signal, monkeypatch):
    """Scenario 4 + safety: after an unknown outcome, retrying the same signal
    must return the blocking message and must never send a second order."""
    _enable_demo(monkeypatch)
    send_calls = {"n": 0}

    def order_send_impl(request):
        send_calls["n"] += 1
        if send_calls["n"] == 1:
            raise RuntimeError("connection dropped mid-send")
        return _success_result(request)

    monkeypatch.setattr(demo_executor, "mt5", _StubMT5Module(order_send_impl))
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    s1, o1, m1 = executor.execute_demo_signal(sample_approved_signal)
    assert s1 is False
    assert o1.status == "EXECUTION_UNKNOWN"

    s2, o2, m2 = executor.execute_demo_signal(sample_approved_signal)
    assert s2 is False, m2
    assert "UNKNOWN" in m2 and "reconciliation" in m2
    assert send_calls["n"] == 1, "a retry after unknown outcome must never reach the broker"
    assert db_session.query(DemoOrder).count() == 1


# Documented TradeOrder field order (mql5.com docs / repr of the real structseq).
_TRADE_ORDER_FIELDS = [
    "ticket", "time_setup", "time_setup_msc", "time_done", "time_done_msc",
    "time_expiration", "type", "type_time", "type_filling", "state", "magic",
    "position_id", "position_by_id", "reason", "volume_initial", "volume_current",
    "price_open", "sl", "tp", "price_current", "price_stoplimit",
    "symbol", "comment", "external_id",
]


def _make_trade_order(**overrides):
    """Build a real MT5 TradeOrder structseq with standard field values.

    history_orders_get(date_from, date_to) returns a tuple of these directly
    (mql5.com docs); there is no wrapper list. The structseq is immutable and
    built from a positional sequence; only ticket/comment/state are read by
    the reconciler, other fields keep 0 defaults exactly as the docs example
    shows zero-valued rows for untouched fields.
    """
    import MetaTrader5 as real_mt5
    values = {name: 0 for name in _TRADE_ORDER_FIELDS}
    values.update({"symbol": "XAUUSD"})
    values.update(overrides)
    return real_mt5.TradeOrder([values[name] for name in _TRADE_ORDER_FIELDS])


def _date_range_history(rows):
    """Stub matching the documented date-range call signature and return shape."""
    def history_impl(date_from, date_to):
        return tuple(rows)
    return history_impl


def test_reconcile_promotes_unknown_to_filled_from_history(db_session, sample_approved_signal, monkeypatch):
    """If MT5 history shows the unknown order actually filled, reconciliation
    must mark it FILLED with the broker ticket. Uses the STANDARD MT5 return
    shape: a tuple of TradeOrder namedtuples, not a wrapper list."""
    _enable_demo(monkeypatch)
    sent = {"comment": None}

    def order_send_impl(request):
        sent["comment"] = request["comment"]
        raise RuntimeError("connection dropped mid-send")

    stub = _StubMT5Module(order_send_impl)
    stub.ORDER_STATE_FILLED = 4
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    def history_impl(date_from, date_to):
        stub._saw_comment = sent["comment"]
        # M5/mql5 semantics: volume_initial=requested 0.01, volume_current=
        # REMAINING volume (0.0 on a full fill); executed = initial - current.
        return (_make_trade_order(ticket=777001, comment=sent["comment"], state=4,
                                  volume_initial=0.01, volume_current=0.0),)

    stub.history_orders_get = history_impl

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    success, order, msg = executor.execute_demo_signal(sample_approved_signal)
    assert success is False
    assert stub._saw_comment and "Demo AI:" in stub._saw_comment, (
        "reconciler must match on the request comment that carried the order_id"
    )
    assert order.status == "FILLED", "reconciliation must promote the row to FILLED"
    assert order.mt5_ticket == 777001
    assert order.volume == pytest.approx(0.01), (
        "a full fill (volume_current=0) must record the requested volume as executed")
    assert "FILLED" not in (order.error_message or "")


def test_reconcile_leaves_unknown_when_history_has_no_match(db_session, sample_approved_signal, monkeypatch):
    """Standard MT5 returns None when nothing is found; the row must stay
    EXECUTION_UNKNOWN, untouched."""
    _enable_demo(monkeypatch)
    send_calls = {"n": 0}

    def order_send_impl(request):
        send_calls["n"] += 1
        raise RuntimeError("connection dropped mid-send")

    stub = _StubMT5Module(order_send_impl)
    stub.ORDER_STATE_FILLED = 4
    stub.history_orders_get = _date_range_history([])  # empty tuple: nothing found
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    success, order, msg = executor.execute_demo_signal(sample_approved_signal)
    assert success is False
    assert send_calls["n"] == 1
    assert order.status == "EXECUTION_UNKNOWN", "no history match must not change the status"
    assert order.mt5_ticket is None


def test_reconcile_survives_history_errors(db_session, sample_approved_signal, monkeypatch):
    """A failing history call (terminal unreachable, error return, or raised
    exception) must not crash dispatch and must leave EXECUTION_UNKNOWN."""
    _enable_demo(monkeypatch)

    def order_send_impl(request):
        raise RuntimeError("connection dropped mid-send")

    stub = _StubMT5Module(order_send_impl)
    stub.ORDER_STATE_FILLED = 4
    # Standard MT5 error shape: None result
    stub.history_orders_get = _date_range_history(None)
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    success, order, msg = executor.execute_demo_signal(sample_approved_signal)
    assert success is False
    assert order.status == "EXECUTION_UNKNOWN"

    # Raised exception inside the history call
    def broken_history(date_from, date_to):
        raise OSError("history call failed")
    stub2 = _StubMT5Module(order_send_impl)
    stub2.ORDER_STATE_FILLED = 4
    stub2.history_orders_get = broken_history
    monkeypatch.setattr(demo_executor, "mt5", stub2)

    db_session.query(DemoOrder).delete()
    db_session.commit()
    success2, order2, msg2 = executor.execute_demo_signal(sample_approved_signal)
    assert success2 is False
    assert order2.status == "EXECUTION_UNKNOWN"


def test_definitive_pre_send_block_stays_rejected_not_unknown(db_session, sample_approved_signal, monkeypatch):
    """Scenario 1: a definitive failure BEFORE dispatch (terminal Algo Trading
    disabled) is a certain REJECTED path; it must not be labelled unknown."""
    _enable_demo(monkeypatch)
    send_calls = {"n": 0}

    def order_send_impl(request):
        send_calls["n"] += 1
        return _success_result(request)

    class _TermDisabled:
        trade_allowed = False

    stub = _StubMT5Module(order_send_impl)
    stub.terminal_info = lambda: _TermDisabled()
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    success, order, msg = executor.execute_demo_signal(sample_approved_signal)

    assert success is False
    assert order is None
    assert "Algo Trading" in msg
    assert send_calls["n"] == 0, "definitive pre-send block must never dispatch"
    assert db_session.query(DemoOrder).count() == 0
    assert db_session.query(DemoOrder).filter_by(status="EXECUTION_UNKNOWN").count() == 0


def test_broker_rejection_stays_rejected_not_unknown(db_session, sample_approved_signal, monkeypatch):
    """A definitive broker refusal (failure retcode) must stay REJECTED."""
    _enable_demo(monkeypatch)

    def order_send_impl(request):
        class _Res:
            retcode = 10030
            comment = "Unsupported filling mode"
        return _Res()

    monkeypatch.setattr(demo_executor, "mt5", _StubMT5Module(order_send_impl))
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    success, order, msg = executor.execute_demo_signal(sample_approved_signal)

    assert success is False
    assert order.status == "REJECTED"
    assert "10030" in (order.error_message or "")
    assert db_session.query(DemoOrder).filter_by(status="EXECUTION_UNKNOWN").count() == 0


def test_concurrent_duplicate_dispatch_returns_duplicate_message(db_session, sample_approved_signal, monkeypatch):
    """Finding 6: two requests passing the pre-send check at once — the loser's
    PENDING commit hits the unique idempotency_key constraint and must return a
    clean duplicate failure, not an unhandled IntegrityError (500)."""
    from sqlalchemy.exc import IntegrityError

    _enable_demo(monkeypatch)
    send_calls = {"n": 0}
    commits = {"n": 0}
    real_commit = db_session.commit

    def order_send_impl(request):
        send_calls["n"] += 1
        return _success_result(request)

    def racing_commit():
        commits["n"] += 1
        # Commit #1 = winner's PENDING row (first dispatch). Commit #2 = the
        # loser's PENDING row, which collides on the unique key exactly as a
        # concurrent duplicate would at the database.
        if commits["n"] == 2:
            raise IntegrityError(
                "INSERT INTO demo_orders ...", None,
                Exception("UNIQUE constraint failed: demo_orders.idempotency_key"),
            )
        real_commit()

    monkeypatch.setattr(demo_executor, "mt5", _StubMT5Module(order_send_impl))
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)
    monkeypatch.setattr(db_session, "commit", racing_commit)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    success1, order1, msg1 = executor.execute_demo_signal(sample_approved_signal)
    assert success1 is True, msg1

    success2, order2, msg2 = executor.execute_demo_signal(sample_approved_signal)
    assert success2 is False, msg2
    assert "Duplicate order prevented" in msg2
    assert send_calls["n"] == 1, "loser of the race must never reach the broker"
    # Exactly one live order survives the collision.
    assert db_session.query(DemoOrder).count() == 1


def test_post_fill_persistence_failure_reports_pending_status(db_session, sample_approved_signal, monkeypatch):
    """Finding 11: after a post-fill commit failure the returned record must
    reflect the store (PENDING), not the in-memory FILLED value."""
    _enable_demo(monkeypatch)
    commits = {"n": 0}
    real_commit = db_session.commit

    def flaky_commit():
        commits["n"] += 1
        if commits["n"] >= 2:
            # Fail BEFORE touching the real store: commit #1 (PENDING) is durable,
            # commit #2 (the FILLED update) never lands, matching a real outage.
            raise ConnectionError("simulated persistence outage")
        real_commit()

    monkeypatch.setattr(db_session, "commit", flaky_commit)
    monkeypatch.setattr(demo_executor, "mt5", _StubMT5Module(lambda request: _success_result(request)))
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    success, order, msg = executor.execute_demo_signal(sample_approved_signal)
    assert success is True, msg
    assert order.status == "PENDING", (
        "response record must match the store after a persistence failure, not claim FILLED"
    )


# ---------------------------------------------------------------------------
# Item 6/37: ambiguous retcodes, None results, PENDING sweeper, partial fill
# ---------------------------------------------------------------------------

def test_none_result_marks_unknown_and_reconciles_to_fill(db_session, sample_approved_signal, monkeypatch):
    """order_send returning None is transport-ambiguous: the order may be live
    at the broker. Must go to EXECUTION_UNKNOWN (never REJECTED) and then
    reconcile to FILLED from broker history."""
    _enable_demo(monkeypatch)
    sent = {"comment": None}

    def order_send_impl(request):
        sent["comment"] = request["comment"]
        return None  # documented MT5 failure shape

    stub = _StubMT5Module(order_send_impl)
    stub.ORDER_STATE_FILLED = 4
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=888001, comment=sent["comment"], state=4,
                          volume_initial=0.10, volume_current=0.0),)
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    success, order, msg = executor.execute_demo_signal(sample_approved_signal)

    assert success is False
    assert order.status == "FILLED", "reconciliation must promote the broker fill"
    assert order.mt5_ticket == 888001
    assert order.volume == pytest.approx(0.10)
    assert db_session.query(DemoOrder).count() == 1


def test_ambiguous_retcode_timeout_marks_unknown_not_rejected(db_session, sample_approved_signal, monkeypatch):
    """TRADE_RETCODE_TIMEOUT is transport-ambiguous: the order may have been
    executed. Must be EXECUTION_UNKNOWN (with idempotency reserved), never
    REJECTED, and blocked from a blind retry."""
    _enable_demo(monkeypatch)

    def order_send_impl(request):
        class _Res:
            retcode = 10012  # TRADE_RETCODE_TIMEOUT
            comment = "Request timed out"
        return _Res()

    stub = _StubMT5Module(order_send_impl)
    stub.ORDER_STATE_FILLED = 4
    stub.TRADE_RETCODE_TIMEOUT = 10012
    stub.history_orders_get = _date_range_history([])  # no confirmed outcome
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    success, order, msg = executor.execute_demo_signal(sample_approved_signal)

    assert success is False
    assert order.status == "EXECUTION_UNKNOWN", (
        f"ambiguous retcode must be UNKNOWN, got {order.status}")
    retry_ok, retry_order, retry_msg = executor.execute_demo_signal(sample_approved_signal)
    assert retry_ok is False and "UNKNOWN" in retry_msg, (
        "blind retry on an ambiguous outcome must be blocked")


def test_partial_fill_volume_recorded_from_broker_history(db_session, sample_approved_signal, monkeypatch):
    """A partial IOC fill must record the EXECUTED volume. Per mql5 docs the
    TradeOrder carries volume_initial (requested) and volume_current
    (REMAINING); a terminal IOC partial shows state FILLED with the unfilled
    remainder left in volume_current — executed = initial - current."""
    _enable_demo(monkeypatch)
    sent = {"comment": None}

    def order_send_impl(request):
        sent["comment"] = request["comment"]
        raise RuntimeError("connection dropped mid-send")

    stub = _StubMT5Module(order_send_impl)
    stub.ORDER_STATE_FILLED = 4
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=888002, comment=sent["comment"], state=4,
                          volume_initial=0.10, volume_current=0.06),)
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    success, order, msg = executor.execute_demo_signal(sample_approved_signal)

    assert order.status == "FILLED"
    assert order.volume == pytest.approx(0.04), (
        f"partial fill must record executed volume (0.10 requested - 0.06 "
        f"remaining), got {order.volume}")


def test_done_partial_retcode_records_fill_not_rejection(db_session, sample_approved_signal, monkeypatch):
    """M5: TRADE_RETCODE_DONE_PARTIAL (10010) confirms a partial IOC
    execution — the row must become FILLED with the broker-confirmed executed
    volume (result.volume), never REJECTED with the fill discarded."""
    _enable_demo(monkeypatch)

    def order_send_impl(request):
        class _Res:
            retcode = 10010  # TRADE_RETCODE_DONE_PARTIAL
            order = 555001
            volume = 0.04  # broker-confirmed executed volume
            price = request["price"]
            comment = "done partial"
        return _Res()

    stub = _StubMT5Module(order_send_impl)
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    success, order, msg = executor.execute_demo_signal(sample_approved_signal)

    assert order.status == "FILLED", (
        f"DONE_PARTIAL is a confirmed (partial) execution, got {order.status}")
    assert order.volume == pytest.approx(0.04), (
        "the executed volume from the broker result must be recorded")
    assert success is True


def test_sweeper_also_recovers_execution_unknown_rows(db_session, sample_approved_signal, monkeypatch):
    """M5: an EXECUTION_UNKNOWN row stuck because the terminal was unreachable
    at dispatch time must be re-swept by the recovery sweep, and a repeated
    non-fill sweep pass must not grow error_message unboundedly."""
    stale_unknown = DemoOrder(
        order_id="ord_unknown_001",
        signal_id=sample_approved_signal.signal_id,
        idempotency_key="XAUUSD_M15_UNKNOWN_1",
        symbol="XAUUSD",
        order_type="BUY",
        volume=0.10,
        requested_price=1900.0,
        stop_loss=1895.0,
        take_profit=1910.0,
        status="EXECUTION_UNKNOWN",
        created_at=datetime.utcnow() - timedelta(minutes=120),
    )
    db_session.add(stale_unknown)
    db_session.commit()

    def order_send_impl(request):
        raise AssertionError("sweeper must not dispatch orders")

    stub = _StubMT5Module(order_send_impl)
    stub.ORDER_STATE_FILLED = 4
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=888004,
                          comment=f"Demo AI: {stale_unknown.order_id}",
                          state=1,  # ORDER_STATE_STARTED: no confirmed fill yet
                          volume_initial=0.10, volume_current=0.10),)
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    reconciled = executor.reconcile_stale_pending_orders(max_age_minutes=30)

    assert reconciled == 0
    db_session.refresh(stale_unknown)
    assert stale_unknown.status == "EXECUTION_UNKNOWN"
    marker_count = (stale_unknown.error_message or "").count("broker history state=")
    assert marker_count == 1, "repeated sweeps must not append the marker again"

    # Now the terminal shows a confirmed full fill: the same sweep must
    # promote the EXECUTION_UNKNOWN row to FILLED.
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=888004,
                          comment=f"Demo AI: {stale_unknown.order_id}",
                          state=4, volume_initial=0.10, volume_current=0.0),)
    reconciled = executor.reconcile_stale_pending_orders(max_age_minutes=30)
    assert reconciled == 1
    db_session.refresh(stale_unknown)
    assert stale_unknown.status == "FILLED" and stale_unknown.mt5_ticket == 888004
    assert stale_unknown.volume == pytest.approx(0.10)


def test_stale_pending_sweeper_recovers_fill(db_session, sample_approved_signal, monkeypatch):
    """A PENDING row orphaned by a failed post-fill commit is recovered by the
    sweeper from broker history; fresh PENDING rows are left alone."""
    _enable_demo(monkeypatch)
    stale_order = DemoOrder(
        order_id="ord_sweep_001",
        signal_id=sample_approved_signal.signal_id,
        idempotency_key="XAUUSD_M15_SWEEP_1",
        symbol="XAUUSD",
        order_type="BUY",
        volume=0.10,
        requested_price=1900.0,
        stop_loss=1895.0,
        take_profit=1910.0,
        status="PENDING",
        created_at=datetime.utcnow() - timedelta(minutes=60),
    )
    db_session.add(stale_order)
    fresh_order = DemoOrder(
        order_id="ord_sweep_002",
        signal_id=sample_approved_signal.signal_id,
        idempotency_key="XAUUSD_M15_SWEEP_2",
        symbol="XAUUSD",
        order_type="BUY",
        volume=0.10,
        requested_price=1900.0,
        stop_loss=1895.0,
        take_profit=1910.0,
        status="PENDING",
        created_at=datetime.utcnow(),
    )
    db_session.add(fresh_order)
    db_session.commit()

    def order_send_impl(request):  # never dispatched in this test
        raise AssertionError("sweeper must not dispatch orders")

    stub = _StubMT5Module(order_send_impl)
    stub.ORDER_STATE_FILLED = 4
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=888003,
                          comment=f"Demo AI: {stale_order.order_id}", state=4,
                          volume_initial=0.10, volume_current=0.0),)
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    reconciled = executor.reconcile_stale_pending_orders(max_age_minutes=30)

    assert reconciled == 1
    db_session.refresh(stale_order)
    db_session.refresh(fresh_order)
    assert stale_order.status == "FILLED" and stale_order.mt5_ticket == 888003
    assert fresh_order.status == "PENDING", "fresh PENDING rows must not be swept"


# ---------------------------------------------------------------------------
# M5 round 2: deals-based reconciliation per the REAL MT5 contract.
# TradeDeal field order verified positionally against the installed package
# (18 fields): ticket, order, time, time_msc, type, entry, magic,
# position_id, reason, volume, price, commission, swap, profit, fee, symbol,
# comment, external_id.
# ---------------------------------------------------------------------------

_TRADE_DEAL_FIELDS = [
    "ticket", "order", "time", "time_msc", "type", "entry", "magic",
    "position_id", "reason", "volume", "price", "commission", "swap",
    "profit", "fee", "symbol", "comment", "external_id",
]


def _make_trade_deal(**overrides):
    """Build a real MT5 TradeDeal structseq; entry defaults to DEAL_ENTRY_IN."""
    import MetaTrader5 as real_mt5
    values = {name: 0 for name in _TRADE_DEAL_FIELDS}
    values.update({"symbol": "XAUUSD", "entry": real_mt5.DEAL_ENTRY_IN})
    values.update(overrides)
    return real_mt5.TradeDeal([values[name] for name in _TRADE_DEAL_FIELDS])


def _stale_row(db_session, sample_approved_signal, order_id, idem_suffix, status="EXECUTION_UNKNOWN"):
    row = DemoOrder(
        order_id=order_id,
        signal_id=sample_approved_signal.signal_id,
        idempotency_key=f"XAUUSD_M15_{idem_suffix}",
        symbol="XAUUSD",
        order_type="BUY",
        volume=0.10,
        requested_price=1900.0,
        stop_loss=1895.0,
        take_profit=1910.0,
        status=status,
        created_at=datetime.utcnow() - timedelta(minutes=120),
    )
    db_session.add(row)
    db_session.commit()
    return row


def test_full_fill_confirmed_by_deals(db_session, sample_approved_signal, monkeypatch):
    """M5 evidence ladder step 1: immutable DEAL records are the authoritative
    executed volume — history_deals_get(position=ticket) with a DEAL_ENTRY_IN
    deal for our order ticket proves the fill and its size."""
    row = _stale_row(db_session, sample_approved_signal, "ord_deals_full", "DEALS_FULL")

    def order_send_impl(request):
        raise AssertionError("sweeper must not dispatch orders")

    stub = _StubMT5Module(order_send_impl)
    stub.ORDER_STATE_FILLED = 4
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=911001, position_id=900001,
                          comment=f"Demo AI: {row.order_id}",
                          state=4, volume_initial=0.10, volume_current=0.0),)
    stub.history_deals_get = lambda **kw: (
        _make_trade_deal(ticket=910100, order=911001, position_id=900001, volume=0.10),)
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    assert executor.reconcile_stale_pending_orders(max_age_minutes=30) == 1

    db_session.refresh(row)
    assert row.status == "FILLED"
    assert row.volume == pytest.approx(0.10)
    assert row.mt5_ticket == 911001


def test_partial_fill_then_cancel_resolved_from_deals_ignoring_exit_deals(db_session, sample_approved_signal, monkeypatch):
    """Partial fill then cancel: the executed volume is the sum of ENTRY_IN
    deals only (0.04). A later EXIT deal on the same position (0.04) must NOT
    be summed, and the canceled remainder must not erase the executed part."""
    row = _stale_row(db_session, sample_approved_signal, "ord_deals_part", "DEALS_PART")

    def order_send_impl(request):
        raise AssertionError("sweeper must not dispatch orders")

    stub = _StubMT5Module(order_send_impl)
    import MetaTrader5 as real_mt5
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=911002, position_id=900002,
                          comment=f"Demo AI: {row.order_id}",
                          state=real_mt5.ORDER_STATE_CANCELED,
                          volume_initial=0.10, volume_current=0.06),)
    stub.history_deals_get = lambda **kw: (
        # The exit deal shares the SAME order ticket — only the
        # DEAL_ENTRY_IN filter must exclude it from the executed volume.
        _make_trade_deal(ticket=910200, order=911002, position_id=900002,
                         entry=real_mt5.DEAL_ENTRY_IN, volume=0.04),
        _make_trade_deal(ticket=910201, order=911002, position_id=900002,
                         entry=real_mt5.DEAL_ENTRY_OUT, volume=0.04),
    )
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    executor.reconcile_stale_pending_orders(max_age_minutes=30)

    db_session.refresh(row)
    assert row.status == "FILLED"
    assert row.volume == pytest.approx(0.04), (
        "executed volume must come from ENTRY_IN deals of OUR order only")


def test_order_without_execution_is_terminal_not_unknown(db_session, sample_approved_signal, monkeypatch):
    """A broker row that exists but never executed (CANCELED/REJECTED with
    zero fills and an empty deal list) must resolve to a DEFINITIVE terminal
    state, not stay EXECUTION_UNKNOWN forever, and must never become FILLED."""
    canceled = _stale_row(db_session, sample_approved_signal, "ord_noexec_c", "NOEXEC_C")
    rejected = _stale_row(db_session, sample_approved_signal, "ord_noexec_r", "NOEXEC_R")

    def order_send_impl(request):
        raise AssertionError("sweeper must not dispatch orders")

    import MetaTrader5 as real_mt5
    stub = _StubMT5Module(order_send_impl)
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=911003, position_id=900003,
                          comment=f"Demo AI: {canceled.order_id}",
                          state=real_mt5.ORDER_STATE_CANCELED,
                          volume_initial=0.10, volume_current=0.10),
        _make_trade_order(ticket=911004, position_id=900004,
                          comment=f"Demo AI: {rejected.order_id}",
                          state=real_mt5.ORDER_STATE_REJECTED,
                          volume_initial=0.10, volume_current=0.10),
    )
    stub.history_deals_get = lambda **kw: ()  # no execution records at all
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    executor.reconcile_stale_pending_orders(max_age_minutes=30)

    db_session.refresh(canceled)
    db_session.refresh(rejected)
    assert canceled.status == "CANCELLED", f"got {canceled.status}"
    assert rejected.status == "REJECTED", f"got {rejected.status}"
    assert canceled.volume == pytest.approx(0.10), "no fill must not alter the requested volume"


def test_deals_unavailable_falls_back_to_order_row_arithmetic(db_session, sample_approved_signal, monkeypatch):
    """When history_deals_get is unavailable (None/error), the order-row
    fallback (volume_initial - volume_current) still resolves a fill — the
    documented contract pinned by the earlier M5 tests keeps working."""
    row = _stale_row(db_session, sample_approved_signal, "ord_deals_none", "DEALS_NONE")

    def order_send_impl(request):
        raise AssertionError("sweeper must not dispatch orders")

    stub = _StubMT5Module(order_send_impl)
    stub.ORDER_STATE_FILLED = 4
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=911005, position_id=900005,
                          comment=f"Demo AI: {row.order_id}",
                          state=4, volume_initial=0.10, volume_current=0.0),)
    stub.history_deals_get = lambda **kw: None  # deals evidence unavailable
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    executor.reconcile_stale_pending_orders(max_age_minutes=30)

    db_session.refresh(row)
    assert row.status == "FILLED" and row.volume == pytest.approx(0.10)


def test_repeated_sweeps_are_idempotent(db_session, sample_approved_signal, monkeypatch):
    """Reconciliation must be repeatable without duplicate recording: once a
    row is terminal it leaves the sweep filter, and re-running the sweep
    neither changes the recorded volume nor re-promotes anything."""
    row = _stale_row(db_session, sample_approved_signal, "ord_idem", "IDEM")

    def order_send_impl(request):
        raise AssertionError("sweeper must not dispatch orders")

    stub = _StubMT5Module(order_send_impl)
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=911006, position_id=900006,
                          comment=f"Demo AI: {row.order_id}",
                          state=4, volume_initial=0.10, volume_current=0.0),)
    stub.history_deals_get = lambda **kw: (
        _make_trade_deal(ticket=910600, order=911006, position_id=900006, volume=0.10),)
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    assert executor.reconcile_stale_pending_orders(max_age_minutes=30) == 1
    db_session.refresh(row)
    first = (row.status, row.volume, row.mt5_ticket, row.executed_at)

    # Second and third sweep passes: nothing left to resolve.
    assert executor.reconcile_stale_pending_orders(max_age_minutes=30) == 0
    assert executor.reconcile_stale_pending_orders(max_age_minutes=30) == 0
    db_session.refresh(row)
    assert (row.status, row.volume, row.mt5_ticket) == first[:3], (
        "repeat sweeps must not alter a resolved row")
    assert row.executed_at == first[3], "executed_at must be stamped exactly once"


def test_multiple_entry_deals_are_summed(db_session, sample_approved_signal, monkeypatch):
    """An order filled across multiple IN deals records the SUM of those
    deals — each deal is one partial execution of the same request."""
    row = _stale_row(db_session, sample_approved_signal, "ord_deals_multi", "DEALS_MULTI")

    def order_send_impl(request):
        raise AssertionError("sweeper must not dispatch orders")

    stub = _StubMT5Module(order_send_impl)
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=911007, position_id=900007,
                          comment=f"Demo AI: {row.order_id}",
                          state=4, volume_initial=0.10, volume_current=0.0),)
    stub.history_deals_get = lambda **kw: (
        _make_trade_deal(ticket=910700, order=911007, position_id=900007, volume=0.03),
        _make_trade_deal(ticket=910701, order=911007, position_id=900007, volume=0.04),
        _make_trade_deal(ticket=910702, order=911007, position_id=900007, volume=0.03),
    )
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    executor.reconcile_stale_pending_orders(max_age_minutes=30)
    db_session.refresh(row)
    assert row.status == "FILLED"
    assert row.volume == pytest.approx(0.10), "0.03 + 0.04 + 0.03 IN deals"


def test_no_evidence_at_all_stays_unknown(db_session, sample_approved_signal, monkeypatch):
    """Both evidence sources unusable (deals unavailable AND order-row fields
    non-numeric) must keep the row unresolved — the None-evidence branch of
    the decision ladder is live, never coerced to 'zero execution'."""
    row = _stale_row(db_session, sample_approved_signal, "ord_no_evidence", "NO_EVID")

    def order_send_impl(request):
        raise AssertionError("sweeper must not dispatch orders")

    stub = _StubMT5Module(order_send_impl)

    class _BrokenRow:  # not a structseq: non-numeric volume fields
        ticket = 910008
        state = 4
        comment = f"Demo AI: {row.order_id}"
        volume_initial = "not-a-number"
        volume_current = "also-not"

    stub.history_orders_get = lambda f, t: (_BrokenRow(),)
    stub.history_deals_get = lambda **kw: None
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    executor.reconcile_stale_pending_orders(max_age_minutes=30)
    db_session.refresh(row)
    assert row.status == "EXECUTION_UNKNOWN", (
        "no usable evidence must never resolve to FILLED/CANCELLED/REJECTED")


def test_missing_history_and_api_error_stay_unknown(db_session, sample_approved_signal, monkeypatch):
    """When the terminal is unreachable (history None) or the history call
    raises, the row must keep EXECUTION_UNKNOWN — an uncertain outcome is
    never promoted, cancelled, or rejected."""
    missing = _stale_row(db_session, sample_approved_signal, "ord_hist_none", "HIST_NONE")

    def order_send_impl(request):
        raise AssertionError("sweeper must not dispatch orders")

    stub = _StubMT5Module(order_send_impl)
    stub.history_orders_get = lambda f, t: None
    stub.history_deals_get = lambda **kw: (_make_trade_deal(volume=0.10),)
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    executor.reconcile_stale_pending_orders(max_age_minutes=30)
    db_session.refresh(missing)
    assert missing.status == "EXECUTION_UNKNOWN"

    # API error: the history call itself raises.
    broken = _stale_row(db_session, sample_approved_signal, "ord_hist_err", "HIST_ERR")

    def broken_history(f, t):
        raise OSError("terminal unreachable")

    stub2 = _StubMT5Module(order_send_impl)
    stub2.history_orders_get = broken_history
    monkeypatch.setattr(demo_executor, "mt5", stub2)
    executor.reconcile_stale_pending_orders(max_age_minutes=30)
    db_session.refresh(broken)
    assert broken.status == "EXECUTION_UNKNOWN"


def test_history_deals_get_is_called_with_order_record_position_id(db_session, sample_approved_signal, monkeypatch):
    """M5 mandatory regression: the reconciler MUST query deals with the
    documented `position=<position-ticket>` keyword form (filters by
    DEAL_POSITION_ID) — never positionally, never with `ticket=`, and never
    with a value other than the order's own broker ticket. Stubs that
    silently accept any arguments do not prove this, so every call's exact
    signature is captured and asserted."""
    row = _stale_row(db_session, sample_approved_signal, "ord_deals_kw", "DEALS_KW")

    def order_send_impl(request):
        raise AssertionError("sweeper must not dispatch orders")

    calls = []

    stub = _StubMT5Module(order_send_impl)
    stub.ORDER_STATE_FILLED = 4
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=911777, position_id=900777,
                          comment=f"Demo AI: {row.order_id}",
                          state=4, volume_initial=0.10, volume_current=0.0),)

    def capturing_deals_query(*args, **kwargs):
        calls.append({"args": args, "kwargs": dict(kwargs)})
        return (
            _make_trade_deal(ticket=910800, order=911777, position_id=900777, volume=0.10),
        )

    stub.history_deals_get = capturing_deals_query
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    executor.reconcile_stale_pending_orders(max_age_minutes=30)

    assert calls, "reconciler must consult broker deals evidence"
    for call in calls:
        assert call["args"] == (), (
            f"history_deals_get must only use named parameters, got positional {call['args']}"
        )
        assert set(call["kwargs"]) == {"position"}, (
            f"history_deals_get must filter by DEAL_POSITION_ID via 'position=', got {call['kwargs']}"
        )
        # order_ticket (911777) and position_id (900777) are deliberately
        # DIFFERENT identifiers: the query must carry the ORDER RECORD's
        # position_id, never the order ticket (a ticket is not a position id).
        assert call["kwargs"]["position"] == 900777, (
            "history_deals_get(position=...) must receive the ORDER RECORD's position_id"
        )
        assert call["kwargs"]["position"] != 911777, (
            "an order ticket must never be passed as a position identifier"
        )

    db_session.refresh(row)
    assert row.status == "FILLED"
    assert row.mt5_ticket == 911777


def test_reconciliation_records_deal_costs_and_vwap_fill_price(db_session, sample_approved_signal, monkeypatch):
    """Partial fills across two prices: the reconciler must record the
    volume-weighted average fill price and the summed commission/swap of the
    ENTRY_IN deals of OUR order. EXIT-deal costs belong to the closing order
    and must be excluded."""
    row = _stale_row(db_session, sample_approved_signal, "ord_deals_cost", "DEALS_COST")

    def order_send_impl(request):
        raise AssertionError("sweeper must not dispatch orders")

    import MetaTrader5 as real_mt5
    stub = _StubMT5Module(order_send_impl)
    stub.ORDER_STATE_FILLED = 4
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=911901, position_id=900901,
                          comment=f"Demo AI: {row.order_id}",
                          state=4, volume_initial=0.10, volume_current=0.0),)
    stub.history_deals_get = lambda **kw: (
        _make_trade_deal(ticket=910902, order=911901, position_id=900901,
                         volume=0.04, price=1900.5, commission=-0.20, swap=0.01),
        _make_trade_deal(ticket=910903, order=911901, position_id=900901,
                         volume=0.06, price=1901.5, commission=-0.30, swap=0.02),
        # EXIT deal of the same position — different order; its volume,
        # price, commission and swap must all be excluded.
        _make_trade_deal(ticket=910904, order=999999, position_id=900901,
                         entry=real_mt5.DEAL_ENTRY_OUT, volume=0.10,
                         price=1905.0, commission=-0.99, swap=9.99),
    )
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    executor.reconcile_stale_pending_orders(max_age_minutes=30)

    db_session.refresh(row)
    assert row.status == "FILLED"
    assert row.volume == pytest.approx(0.10)
    expected_vwap = (0.04 * 1900.5 + 0.06 * 1901.5) / 0.10  # 1901.10
    assert row.executed_price == pytest.approx(expected_vwap)
    assert row.commission == pytest.approx(-0.50)
    assert row.swap == pytest.approx(0.03)


def test_order_row_fallback_does_not_fabricate_cost_values(db_session, sample_approved_signal, monkeypatch):
    """When deals evidence is unavailable the order-row arithmetic may still
    resolve the fill, but it has no price/cost fields — the reconciler must
    leave executed_price/commission/swap exactly as stored instead of writing
    fabricated zeros over them."""
    row = _stale_row(db_session, sample_approved_signal, "ord_fallback_cost", "FB_COST")
    row.executed_price = None
    row.commission = 5.0
    row.swap = 7.0
    db_session.commit()

    def order_send_impl(request):
        raise AssertionError("sweeper must not dispatch orders")

    stub = _StubMT5Module(order_send_impl)
    stub.ORDER_STATE_FILLED = 4
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=911950, position_id=900950,
                          comment=f"Demo AI: {row.order_id}",
                          state=4, volume_initial=0.10, volume_current=0.0),)
    stub.history_deals_get = lambda **kw: None  # deals evidence unavailable
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    executor.reconcile_stale_pending_orders(max_age_minutes=30)

    db_session.refresh(row)
    assert row.status == "FILLED"
    assert row.volume == pytest.approx(0.10)
    assert row.executed_price is None, "fallback must not invent a fill price"
    assert row.commission == pytest.approx(5.0), "fallback must not overwrite stored commission"
    assert row.swap == pytest.approx(7.0), "fallback must not overwrite stored swap"


def test_history_deals_get_exception_is_distinct_from_empty_evidence(db_session, sample_approved_signal, monkeypatch):
    """A history_deals_get API exception is a distinct evidence state: the
    reconciler must log it, treat deals evidence as UNAVAILABLE (not empty),
    and fall back to the order-row arithmetic — a raising deals call must not
    strand the order unresolved nor fabricate FILLED from nothing."""
    row = _stale_row(db_session, sample_approved_signal, "ord_deals_raise", "DEALS_RAISE")

    def order_send_impl(request):
        raise AssertionError("sweeper must not dispatch orders")

    stub = _StubMT5Module(order_send_impl)
    stub.ORDER_STATE_FILLED = 4
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=911960, position_id=900960,
                          comment=f"Demo AI: {row.order_id}",
                          state=4, volume_initial=0.10, volume_current=0.0),)

    def raising_deals_query(**kwargs):
        raise OSError("terminal history unavailable")

    stub.history_deals_get = raising_deals_query
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    executor.reconcile_stale_pending_orders(max_age_minutes=30)

    db_session.refresh(row)
    assert row.status == "FILLED", "order-row arithmetic must still resolve the fill"
    assert row.volume == pytest.approx(0.10)
    assert row.executed_price is None, "no deals evidence -> no fabricated fill price"
    assert row.commission == pytest.approx(0.0) and row.swap == pytest.approx(0.0)


def test_unrelated_entry_deal_on_same_position_is_excluded(db_session, sample_approved_signal, monkeypatch):
    """Another order's ENTRY deal living on the SAME position (e.g. a netting
    add-on from a later request) must be excluded from our volume, VWAP and
    costs — order identity (deal.order) is the attribution key, not the
    position membership."""
    row = _stale_row(db_session, sample_approved_signal, "ord_unrel_deal", "UNREL_DEAL")

    def order_send_impl(request):
        raise AssertionError("sweeper must not dispatch orders")

    import MetaTrader5 as real_mt5
    stub = _StubMT5Module(order_send_impl)
    stub.ORDER_STATE_FILLED = 4
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=911970, position_id=900970,
                          comment=f"Demo AI: {row.order_id}",
                          state=4, volume_initial=0.05, volume_current=0.0),)
    stub.history_deals_get = lambda **kw: (
        _make_trade_deal(ticket=910970, order=911970, position_id=900970,
                         volume=0.05, price=1900.0, commission=-0.25, swap=0.0),
        # A DIFFERENT order's ENTRY deal on the same position: not ours.
        _make_trade_deal(ticket=910971, order=888888, position_id=900970,
                         entry=real_mt5.DEAL_ENTRY_IN, volume=5.0,
                         price=2100.0, commission=-9.99, swap=1.23),
    )
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    executor.reconcile_stale_pending_orders(max_age_minutes=30)

    db_session.refresh(row)
    assert row.status == "FILLED"
    assert row.volume == pytest.approx(0.05), "unrelated order's deal must not inflate the volume"
    assert row.executed_price == pytest.approx(1900.0)
    assert row.commission == pytest.approx(-0.25), "unrelated order's commission must be excluded"
    assert row.swap == pytest.approx(0.0)


def test_zero_position_id_order_row_skips_deals_query(db_session, sample_approved_signal, monkeypatch):
    """A broker order row with position_id=0 (order unrelated to any position
    yet, or server that does not populate it) provides NO valid position
    identifier: the deals query must be skipped entirely and the order-row
    arithmetic decides — passing the order ticket as a position id is
    forbidden."""
    row = _stale_row(db_session, sample_approved_signal, "ord_zero_pos", "ZERO_POS")

    def order_send_impl(request):
        raise AssertionError("sweeper must not dispatch orders")

    calls = []
    stub = _StubMT5Module(order_send_impl)
    stub.ORDER_STATE_FILLED = 4
    stub.history_orders_get = lambda f, t: (
        _make_trade_order(ticket=911980, position_id=0,
                          comment=f"Demo AI: {row.order_id}",
                          state=4, volume_initial=0.10, volume_current=0.0),)

    def forbidden_deals_query(**kwargs):
        calls.append(dict(kwargs))
        return (_make_trade_deal(ticket=910980, order=911980, position_id=911980, volume=0.10),)

    stub.history_deals_get = forbidden_deals_query
    monkeypatch.setattr(demo_executor, "mt5", stub)
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    executor.reconcile_stale_pending_orders(max_age_minutes=30)

    assert calls == [], (
        "history_deals_get must never be called without a real position_id "
        "from the order record (order ticket is not a position identifier)")
    db_session.refresh(row)
    assert row.status == "FILLED", "order-row arithmetic must still resolve the fill"
    assert row.volume == pytest.approx(0.10)
    assert row.executed_price is None
