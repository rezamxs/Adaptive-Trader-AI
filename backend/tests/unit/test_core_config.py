"""Unit tests for Core Configuration and Security mode invariants."""
import pytest
from backend.app.core.config import ExecutionMode, Settings
from backend.app.core.errors import LiveTradingProhibitedError, InvalidExecutionModeError


def test_default_execution_mode_is_signal_only():
    """Verify system default mode is strictly SIGNAL_ONLY."""
    s = Settings()
    assert s.EXECUTION_MODE == ExecutionMode.SIGNAL_ONLY


def test_live_trading_is_strictly_prohibited():
    """Verify any attempt to set LIVE or REAL mode raises LiveTradingProhibitedError."""
    with pytest.raises(LiveTradingProhibitedError):
        Settings(EXECUTION_MODE="LIVE_TRADING")

    with pytest.raises(LiveTradingProhibitedError):
        Settings(EXECUTION_MODE="real_account")

    with pytest.raises(LiveTradingProhibitedError):
        Settings(EXECUTION_MODE="LIVE")


def test_invalid_mode_raises_error():
    """Verify unknown modes are rejected."""
    with pytest.raises(InvalidExecutionModeError):
        Settings(EXECUTION_MODE="EXPERIMENTAL_HIGH_FREQUENCY")


def test_permitted_modes_succeed():
    """Verify only BACKTEST, SIGNAL_ONLY, and DEMO_TRADING are valid."""
    s1 = Settings(EXECUTION_MODE="BACKTEST")
    assert s1.EXECUTION_MODE == ExecutionMode.BACKTEST

    s2 = Settings(EXECUTION_MODE="SIGNAL_ONLY")
    assert s2.EXECUTION_MODE == ExecutionMode.SIGNAL_ONLY

    s3 = Settings(EXECUTION_MODE="DEMO_TRADING")
    assert s3.EXECUTION_MODE == ExecutionMode.DEMO_TRADING


def test_demo_trading_gate_default_is_false():
    """Verify DEMO_TRADING_ENABLED gate defaults to False for safety."""
    s = Settings()
    assert s.DEMO_TRADING_ENABLED is False
