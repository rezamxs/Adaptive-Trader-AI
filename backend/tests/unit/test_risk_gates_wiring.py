"""Stage-4.5 regression tests: the strategy session/martingale gates are
WIRED into the production risk path (signal manager + demo dispatch), and the
martingale drawdown is computed from REAL account floating loss.

The manager/executor call sites pass `current_time_utc=datetime.now()`, so
tests freeze THAT module's clock for deterministic session-hour scenarios.
All DB access is in-memory; no broker connection, no orders.
"""
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from types import SimpleNamespace

from backend.app.core.config import settings, ExecutionMode
from backend.app.db.base import Base
import backend.app.models  # noqa: F401 - register all tables
from backend.app.models.rule import TradingRule
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.market_data.mock_provider import MockMarketDataProvider
from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider
from backend.app.services.market_data.base import AccountDetails, Quote, SymbolInfo
from backend.app.services.signal.manager import SignalManager
from backend.app.services.execution import demo_executor
from backend.app.services.execution.demo_executor import DemoTradingExecutor
from backend.app.models.signal import TradeSignal


class _FrozenClockMeta(type):
    """Keep isinstance(x, datetime) true for REAL datetime instances even while
    the module-level datetime name is the frozen test double."""

    def __instancecheck__(cls, instance):
        return isinstance(instance, datetime) or type.__instancecheck__(cls, instance)


class _FixedDatetime(datetime, metaclass=_FrozenClockMeta):
    """Clock frozen to Wednesday 15:00 UTC (outside the 00:00-09:30 window)."""

    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 28, 15, 0, tzinfo=timezone.utc)


class _FixedDatetimeAllowed(datetime, metaclass=_FrozenClockMeta):
    """Clock frozen to Wednesday 05:00 UTC (inside the allowed window)."""

    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 28, 5, 0, tzinfo=timezone.utc)


class _ControlledAccountProvider(MockMarketDataProvider):
    """Mock candles for feature computation + controlled real account and
    controlled open-position list for grid-drawdown computation."""

    def __init__(self, positions=None):
        super().__init__(is_demo_account=True)
        self._positions = positions  # None = provider cannot verify

    def get_account_info(self):
        return AccountDetails(
            login_masked="***", server="Test-Server", currency="USD",
            leverage=100, balance=10000.0, equity=10000.0,
            margin=0.0, free_margin=10000.0, trade_mode=0, is_demo=True,
        )

    def get_current_quote(self, symbol):
        return Quote(symbol=symbol, time=datetime.now(timezone.utc),
                     bid=1.0999, ask=1.1001, spread_points=20)

    def get_symbol_info(self, symbol):
        return SymbolInfo(
            name=symbol, digits=5, point=0.00001, spread=20,
            tick_size=0.00001, tick_value=1.0,
            volume_min=0.01, volume_max=100.0, volume_step=0.01,
            trade_contract_size=100000.0, currency_base="EUR",
            currency_profit="USD", currency_margin="EUR",
        )

    def get_open_positions_count(self):
        return 0

    def get_open_positions(self):
        return self._positions

    def get_daily_loss_percent(self):
        return 0.0


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://", poolclass=__import__("sqlalchemy.pool", fromlist=["StaticPool"]).StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def approved_buy_rule(db_session):
    rule = TradingRule(
        rule_id="rule_rsi_oversold_wiring",
        name="Buy on RSI Oversold",
        category="entry",
        description="Buy when RSI is below 70 (wiring test)",
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
    return rule


def test_session_gate_rejects_production_signal_outside_allowed_hours(
        db_session, approved_buy_rule, monkeypatch):
    """The wired production path must consult the session filter: at 15:00 UTC
    (frozen manager clock) the signal is REJECTED with the session reason."""
    monkeypatch.setattr(settings, "ENFORCE_SESSION_FILTER", True)
    monkeypatch.setattr("backend.app.services.signal.manager.datetime", _FixedDatetime)
    manager = SignalManager(db_session, _ControlledAccountProvider(positions=[]))

    sig = manager.process_latest_bar("EURUSD", TimeFrame.M15)

    assert sig is not None, "a directional signal must be produced and risk-assessed"
    assert sig.risk_status == "REJECTED"
    assert "SESSION_OUTSIDE_ALLOWED_HOURS" in (sig.risk_rejection_reason or ""), (
        f"session gate must fire on the production path, got: {sig.risk_rejection_reason}")


def test_martingale_gate_uses_real_grid_floating_loss(db_session, approved_buy_rule, monkeypatch):
    """$300 floating loss on a GRID position (Demo AI comment) trips the $200
    ceiling; an unrelated MANUAL position's $500 loss is NOT counted."""
    from types import SimpleNamespace
    monkeypatch.setattr(settings, "ENFORCE_SESSION_FILTER", True)
    monkeypatch.setattr("backend.app.services.signal.manager.datetime", _FixedDatetime)
    positions = [
        SimpleNamespace(comment="Demo AI: ord_grid_1", profit=-300.0, type=0),
        SimpleNamespace(comment="manual trade by operator", profit=-500.0, type=0),
    ]
    manager = SignalManager(db_session, _ControlledAccountProvider(positions=positions))

    sig = manager.process_latest_bar("EURUSD", TimeFrame.M15)

    assert sig is not None
    assert sig.risk_status == "REJECTED"
    assert "MARTINGALE_LOSS_LIMIT" in (sig.risk_rejection_reason or ""), (
        f"real grid floating loss must trip the martingale ceiling, got: {sig.risk_rejection_reason}")


def test_grid_gate_ignores_unrelated_manual_positions(db_session, approved_buy_rule, monkeypatch):
    """ONLY manual (non-grid) positions are open: the martingale ceiling must
    NOT trip on their floating loss — the gate is grid-scoped."""
    from types import SimpleNamespace
    monkeypatch.setattr(settings, "ENFORCE_SESSION_FILTER", True)
    monkeypatch.setattr("backend.app.services.signal.manager.datetime", _FixedDatetimeAllowed)
    positions = [
        SimpleNamespace(comment="manual trade by operator", profit=-500.0, type=0),
        SimpleNamespace(comment="manual trade by operator", profit=-900.0, type=0),
    ]
    manager = SignalManager(db_session, _ControlledAccountProvider(positions=positions))

    sig = manager.process_latest_bar("EURUSD", TimeFrame.M15)

    assert sig is not None
    assert sig.risk_status == "APPROVED", (
        f"manual positions must not trip the grid ceiling, got: {sig.risk_rejection_reason}")


def test_martingale_data_unavailable_fails_closed(db_session, approved_buy_rule, monkeypatch):
    """When open positions cannot be verified (None), the path is REJECTED —
    never approved with a zero default."""
    monkeypatch.setattr(settings, "ENFORCE_SESSION_FILTER", True)
    monkeypatch.setattr("backend.app.services.signal.manager.datetime", _FixedDatetimeAllowed)
    manager = SignalManager(db_session, _ControlledAccountProvider(positions=None))

    sig = manager.process_latest_bar("EURUSD", TimeFrame.M15)

    assert sig is not None
    assert sig.risk_status == "REJECTED"
    assert "MARTINGALE_DATA_UNAVAILABLE" in (sig.risk_rejection_reason or ""), (
        f"unverifiable position data must fail closed, got: {sig.risk_rejection_reason}")


def test_grid_position_with_invalid_profit_fails_closed(db_session, approved_buy_rule, monkeypatch):
    """A grid position with an unusable profit value (None / non-numeric) makes
    the drawdown unverifiable -> fail closed, no zero substitution."""
    from types import SimpleNamespace
    monkeypatch.setattr(settings, "ENFORCE_SESSION_FILTER", True)
    monkeypatch.setattr("backend.app.services.signal.manager.datetime", _FixedDatetimeAllowed)
    positions = [SimpleNamespace(comment="Demo AI: ord_bad", profit=None, type=0)]
    manager = SignalManager(db_session, _ControlledAccountProvider(positions=positions))

    sig = manager.process_latest_bar("EURUSD", TimeFrame.M15)

    assert sig is not None
    assert sig.risk_status == "REJECTED"
    assert "MARTINGALE_DATA_UNAVAILABLE" in (sig.risk_rejection_reason or "")


def test_empty_grid_is_zero_drawdown(db_session, approved_buy_rule, monkeypatch):
    """No open positions at all = genuinely zero grid drawdown (an empty list
    is verifiable data, unlike None)."""
    monkeypatch.setattr(settings, "ENFORCE_SESSION_FILTER", True)
    monkeypatch.setattr("backend.app.services.signal.manager.datetime", _FixedDatetimeAllowed)
    manager = SignalManager(db_session, _ControlledAccountProvider(positions=[]))

    sig = manager.process_latest_bar("EURUSD", TimeFrame.M15)

    assert sig is not None
    assert sig.risk_status == "APPROVED", f"got: {sig.risk_rejection_reason}"


# ---------------------------------------------------------------------------
# Item 4 (executor side): the demo dispatch path consults the same gates.
# ---------------------------------------------------------------------------

class _FakeExecutorProvider(MT5MarketDataProvider):
    """Terminal-free provider stub with controlled open positions (same
    pattern as test_audit_fixes_h6._FakeMT5Provider)."""

    def __init__(self, positions=None):
        self._positions = positions  # None = provider cannot verify
        self._connected = True

    def connect(self):
        return True

    def is_connected(self):
        return True

    def get_account_info(self):
        return AccountDetails(
            login_masked="***", server="Test-Server", currency="USD",
            leverage=100, balance=10000.0, equity=10000.0,
            margin=0.0, free_margin=10000.0, trade_mode=0, is_demo=True,
        )

    def get_current_quote(self, symbol):
        return Quote(symbol=symbol, time=datetime.now(timezone.utc),
                     bid=1.0999, ask=1.1001, spread_points=20)

    def get_symbol_info(self, symbol):
        return SymbolInfo(
            name=symbol, digits=5, point=0.00001, spread=20,
            tick_size=0.00001, tick_value=1.0,
            volume_min=0.01, volume_max=100.0, volume_step=0.01,
            trade_contract_size=100000.0, currency_base="EUR",
            currency_profit="USD", currency_margin="EUR",
        )

    def get_open_positions_count(self):
        return 0

    def get_open_positions(self):
        return self._positions

    def get_daily_loss_percent(self):
        return 0.0

    def calculate_order_margin(self, *args, **kwargs):
        return 0.0


def _approved_signal(db_session, key_suffix="001"):
    sig = TradeSignal(
        signal_id=f"sig_wiring_{key_suffix}",
        idempotency_key=f"EURUSD_M15_202610280900_{key_suffix}",
        symbol="EURUSD",
        timeframe="M15",
        direction="BUY",
        strength=0.8,
        entry_price=1.1001,
        stop_loss=1.0951,
        take_profit=1.1101,
        rules_triggered=["rule_rsi_oversold_wiring"],
        features_snapshot={},
        risk_status="APPROVED",
        calculated_volume=0.10,
        execution_mode=ExecutionMode.DEMO_TRADING.value,
        status="GENERATED",
        bar_time=datetime(2026, 10, 28, 9, 0),
    )
    db_session.add(sig)
    db_session.commit()
    return sig


def test_executor_dispatch_rejects_outside_allowed_hours(db_session, approved_buy_rule, monkeypatch):
    """The demo dispatch path must consult the session filter too: at 15:00
    UTC the order is refused BEFORE any broker interaction."""
    monkeypatch.setattr(settings, "EXECUTION_MODE", ExecutionMode.DEMO_TRADING)
    monkeypatch.setattr(settings, "DEMO_TRADING_ENABLED", True)
    monkeypatch.setattr(settings, "ENFORCE_SESSION_FILTER", True)
    monkeypatch.setattr("backend.app.services.execution.demo_executor.datetime", _FixedDatetime)

    def never_send(request):
        raise AssertionError("order_send must not be reached when the session gate rejects")

    class _StubMT5:
        TRADE_RETCODE_DONE = 10009
        ORDER_TYPE_BUY = 0
        ORDER_TIME_GTC = 0
        ORDER_FILLING_IOC = 1
        TRADE_ACTION_DEAL = 1

        def terminal_info(self):
            return type("T", (), {"trade_allowed": True})()

        def order_send(self, request):
            return never_send(request)

    monkeypatch.setattr(demo_executor, "mt5", _StubMT5())
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    sig = _approved_signal(db_session, "sess")
    executor = DemoTradingExecutor(db_session, _FakeExecutorProvider(positions=[]))
    success, order, msg = executor.execute_demo_signal(sig)

    assert success is False
    assert "SESSION_OUTSIDE_ALLOWED_HOURS" in msg, f"session gate must fire: {msg}"


def test_executor_dispatch_rejects_on_real_floating_loss(db_session, approved_buy_rule, monkeypatch):
    """$300 real floating loss trips the martingale ceiling in the dispatch
    path; session filter is disabled here to isolate the martingale gate."""
    monkeypatch.setattr(settings, "EXECUTION_MODE", ExecutionMode.DEMO_TRADING)
    monkeypatch.setattr(settings, "DEMO_TRADING_ENABLED", True)
    monkeypatch.setattr(settings, "ENFORCE_SESSION_FILTER", False)
    monkeypatch.setattr("backend.app.services.execution.demo_executor.datetime", _FixedDatetimeAllowed)

    class _StubMT5:
        TRADE_RETCODE_DONE = 10009
        ORDER_TYPE_BUY = 0
        ORDER_TIME_GTC = 0
        ORDER_FILLING_IOC = 1
        TRADE_ACTION_DEAL = 1

        def terminal_info(self):
            return type("T", (), {"trade_allowed": True})()

        def order_send(self, request):
            raise AssertionError("order_send must not be reached when the martingale gate rejects")

    monkeypatch.setattr(demo_executor, "mt5", _StubMT5())
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    sig = _approved_signal(db_session, "mart")
    executor = DemoTradingExecutor(db_session, _FakeExecutorProvider(positions=[
        SimpleNamespace(comment="Demo AI: ord_grid_exec", profit=-300.0, type=0),
        SimpleNamespace(comment="manual by operator", profit=-900.0, type=0),
    ]))
    success, order, msg = executor.execute_demo_signal(sig)

    assert success is False
    assert "MARTINGALE_LOSS_LIMIT" in msg, f"martingale gate must fire: {msg}"
