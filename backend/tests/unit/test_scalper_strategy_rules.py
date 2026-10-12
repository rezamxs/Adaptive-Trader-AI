"""Tests for Scalper Strategy specific risk rules and document parsers.
Validates Gold minimum capital, Session time windows, Martingale drawdown ceiling, and DOCX parser.
"""
import os
from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.db.base import Base
from backend.app.services.market_data.base import AccountDetails, Quote, SymbolInfo
from backend.app.services.risk.risk_engine import RiskEngine
from backend.app.services.knowledge.parsers import DocumentParser


@pytest.fixture
def memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def dummy_symbol_gold():
    return SymbolInfo(
        name="XAUUSD",
        digits=2,
        point=0.01,
        spread=20,
        tick_size=0.01,
        tick_value=1.0,
        volume_min=0.01,
        volume_max=10.0,
        volume_step=0.01,
        trade_contract_size=100.0,
        currency_base="XAU",
        currency_profit="USD",
        currency_margin="USD",
    )


@pytest.fixture
def dummy_quote_gold():
    return Quote(
        symbol="XAUUSD",
        time=datetime.now(timezone.utc),
        bid=2650.00,
        ask=2650.20,
        spread_points=20,
        is_stale=False,
    )


def test_gold_minimum_equity_gate(memory_db, dummy_symbol_gold, dummy_quote_gold):
    """Test strategy rule: Gold trading requires at least $5,000 equity."""
    risk_engine = RiskEngine(memory_db)

    # Sub-5000 account (e.g., $4,900)
    low_account = AccountDetails(
        login_masked="******11",
        server="DemoServer",
        currency="USD",
        leverage=500,
        balance=4900.0,
        equity=4900.0,
        margin=0.0,
        free_margin=4900.0,
        trade_mode=0,
        is_demo=True,
    )

    is_approved, lots, reason = risk_engine.assess_signal(
        symbol="XAUUSD",
        direction="BUY",
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2660.0,
        account=low_account,
        symbol_info=dummy_symbol_gold,
        quote=dummy_quote_gold,
        open_trades_count=0,
        daily_loss_percent=0.0,
        enforce_gold_min_equity=True,
    )

    assert not is_approved
    assert "REJECTED_GOLD_MIN_EQUITY" in reason
    assert lots == 0.0

    # Adequate capital account ($10,000)
    ok_account = AccountDetails(
        login_masked="******11",
        server="DemoServer",
        currency="USD",
        leverage=500,
        balance=10000.0,
        equity=10000.0,
        margin=0.0,
        free_margin=10000.0,
        trade_mode=0,
        is_demo=True,
    )

    is_approved_ok, lots_ok, reason_ok = risk_engine.assess_signal(
        symbol="XAUUSD",
        direction="BUY",
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2660.0,
        account=ok_account,
        symbol_info=dummy_symbol_gold,
        quote=dummy_quote_gold,
        open_trades_count=0,
        daily_loss_percent=0.0,
        enforce_gold_min_equity=True,
    )

    assert is_approved_ok
    assert lots_ok > 0.0
    assert "APPROVED" in reason_ok


def test_session_hours_gate(memory_db, dummy_symbol_gold, dummy_quote_gold):
    """Test strategy rule: Asian and early European session only (00:00-09:30 UTC / 03:30-13:00 Iran)."""
    risk_engine = RiskEngine(memory_db)
    account = AccountDetails(
        login_masked="******11",
        server="DemoServer",
        currency="USD",
        leverage=500,
        balance=10000.0,
        equity=10000.0,
        margin=0.0,
        free_margin=10000.0,
        trade_mode=0,
        is_demo=True,
    )

    # Allowed time: 05:00 UTC (Tuesday)
    valid_time = datetime(2026, 10, 13, 5, 0, 0, tzinfo=timezone.utc)
    ok, lots, reason = risk_engine.assess_signal(
        symbol="XAUUSD",
        direction="BUY",
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2660.0,
        account=account,
        symbol_info=dummy_symbol_gold,
        quote=dummy_quote_gold,
        open_trades_count=0,
        daily_loss_percent=0.0,
        current_time_utc=valid_time,
        enforce_session_filter=True,
    )
    assert ok
    assert "APPROVED" in reason

    # Disallowed time: 14:00 UTC (New York session - Tuesday)
    ny_time = datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)
    blocked, _, blocked_reason = risk_engine.assess_signal(
        symbol="XAUUSD",
        direction="BUY",
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2660.0,
        account=account,
        symbol_info=dummy_symbol_gold,
        quote=dummy_quote_gold,
        open_trades_count=0,
        daily_loss_percent=0.0,
        current_time_utc=ny_time,
        enforce_session_filter=True,
    )
    assert not blocked
    assert "REJECTED_SESSION_OUTSIDE_ALLOWED_HOURS" in blocked_reason


def test_friday_close_weekend_risk(memory_db, dummy_symbol_gold, dummy_quote_gold):
    """Test strategy rule: Late Friday entries blocked to prevent weekend risk."""
    risk_engine = RiskEngine(memory_db)
    account = AccountDetails(
        login_masked="******11",
        server="DemoServer",
        currency="USD",
        leverage=500,
        balance=10000.0,
        equity=10000.0,
        margin=0.0,
        free_margin=10000.0,
        trade_mode=0,
        is_demo=True,
    )

    # Friday at 19:30 UTC
    friday_night = datetime(2026, 10, 16, 19, 30, 0, tzinfo=timezone.utc)
    blocked, _, reason = risk_engine.assess_signal(
        symbol="XAUUSD",
        direction="BUY",
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2660.0,
        account=account,
        symbol_info=dummy_symbol_gold,
        quote=dummy_quote_gold,
        open_trades_count=0,
        daily_loss_percent=0.0,
        current_time_utc=friday_night,
        enforce_session_filter=True,
    )
    assert not blocked
    assert "REJECTED_FRIDAY_WEEKEND_RISK" in reason


def test_martingale_drawdown_ceiling(memory_db, dummy_symbol_gold, dummy_quote_gold):
    """Test strategy rule: Manual cutoff at $100-$200 floating loss."""
    risk_engine = RiskEngine(memory_db)
    account = AccountDetails(
        login_masked="******11",
        server="DemoServer",
        currency="USD",
        leverage=500,
        balance=10000.0,
        equity=10000.0,
        margin=0.0,
        free_margin=10000.0,
        trade_mode=0,
        is_demo=True,
    )

    # Under ceiling ($150)
    ok, _, reason_ok = risk_engine.assess_signal(
        symbol="XAUUSD",
        direction="BUY",
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2660.0,
        account=account,
        symbol_info=dummy_symbol_gold,
        quote=dummy_quote_gold,
        open_trades_count=0,
        daily_loss_percent=0.0,
        active_martingale_drawdown=150.0,
    )
    assert ok
    assert "APPROVED" in reason_ok

    # At or above ceiling ($200)
    breached, _, reason_breach = risk_engine.assess_signal(
        symbol="XAUUSD",
        direction="BUY",
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2660.0,
        account=account,
        symbol_info=dummy_symbol_gold,
        quote=dummy_quote_gold,
        open_trades_count=0,
        daily_loss_percent=0.0,
        active_martingale_drawdown=200.0,
    )
    assert not breached
    assert "REJECTED_MARTINGALE_LOSS_LIMIT" in reason_breach


def test_document_parser_docx():
    """Test DocumentParser native support for DOCX files."""
    test_docx_path = r"<OWNER_HOME>\Downloads\فایل ها\اصطلاح پرکاربرد در ترید.docx"
    if os.path.exists(test_docx_path):
        with open(test_docx_path, "rb") as f:
            content_bytes = f.read()
        full_text, pages, chunks = DocumentParser.parse_file("اصطلاح پرکاربرد در ترید.docx", content_bytes)
        assert len(full_text) > 100
        assert pages >= 1
        assert len(chunks) >= 1
        assert "Market" in full_text
