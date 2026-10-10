"""Unit tests for Security Gates, Account Verification, and Fail-Closed semantics."""
import pytest
from types import SimpleNamespace
from backend.app.core.config import ExecutionMode
from backend.app.core.security import (
    verify_account_is_demo,
    assert_safe_to_execute_demo_order,
    mask_login,
    MT5_ACCOUNT_TRADE_MODE_DEMO,
    MT5_ACCOUNT_TRADE_MODE_REAL,
)
from backend.app.core.errors import (
    EmergencyStopActiveError,
    SecurityViolationError,
    AccountAuthenticationError,
)


def test_mask_login():
    """Verify login number masking."""
    assert mask_login(12345678) == "******78"
    assert mask_login("87654321") == "******21"
    assert mask_login(12) == "**"


def test_verify_account_is_demo_success():
    """Verify successful verification of legitimate DEMO account."""
    mock_acc = SimpleNamespace(
        trade_mode=MT5_ACCOUNT_TRADE_MODE_DEMO,
        server="YourBroker-MT5",
        login=12345683,
        currency="USD",
        leverage=40,
        balance=10000.0,
        equity=10000.0,
    )
    is_demo, reason, details = verify_account_is_demo(mock_acc, expected_server="YourBroker-MT5")
    assert is_demo is True
    assert details["login_masked"] == "******83"
    assert "DEMO" in reason


def test_verify_account_rejects_real_account():
    """Verify REAL account is immediately flagged as a critical security rejection."""
    mock_acc = SimpleNamespace(
        trade_mode=MT5_ACCOUNT_TRADE_MODE_REAL,
        server="the configured broker-Real",
        login=99887766,
        currency="USD",
        leverage=100,
        balance=50000.0,
        equity=50000.0,
    )
    is_demo, reason, details = verify_account_is_demo(mock_acc)
    assert is_demo is False
    assert "SECURITY ALERT: Account is REAL" in reason


def test_assert_safe_to_execute_demo_order_fails_closed():
    """Verify fail-closed triggers under various risk/safety conditions."""
    # Emergency stop active
    with pytest.raises(EmergencyStopActiveError):
        assert_safe_to_execute_demo_order(
            current_mode=ExecutionMode.DEMO_TRADING,
            is_demo_enabled=True,
            is_emergency_stop_active=True,
            is_connected=True,
            account_is_demo=True,
        )

    # In SIGNAL_ONLY mode
    with pytest.raises(SecurityViolationError):
        assert_safe_to_execute_demo_order(
            current_mode=ExecutionMode.SIGNAL_ONLY,
            is_demo_enabled=True,
            is_emergency_stop_active=False,
            is_connected=True,
            account_is_demo=True,
        )

    # Gate is False
    with pytest.raises(SecurityViolationError):
        assert_safe_to_execute_demo_order(
            current_mode=ExecutionMode.DEMO_TRADING,
            is_demo_enabled=False,
            is_emergency_stop_active=False,
            is_connected=True,
            account_is_demo=True,
        )

    # Disconnected
    with pytest.raises(SecurityViolationError):
        assert_safe_to_execute_demo_order(
            current_mode=ExecutionMode.DEMO_TRADING,
            is_demo_enabled=True,
            is_emergency_stop_active=False,
            is_connected=False,
            account_is_demo=True,
        )

    # Account not demo
    with pytest.raises(AccountAuthenticationError):
        assert_safe_to_execute_demo_order(
            current_mode=ExecutionMode.DEMO_TRADING,
            is_demo_enabled=True,
            is_emergency_stop_active=False,
            is_connected=True,
            account_is_demo=False,
        )
