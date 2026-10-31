"""Core Security and Safety Gate Enforcement.
Implements dual verification for Demo accounts and blocks any unauthorized actions.
"""
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from backend.app.core.config import ExecutionMode, settings
from backend.app.core.errors import (
    AccountAuthenticationError,
    EmergencyStopActiveError,
    SecurityViolationError,
)

# MetaTrader 5 trade mode constants
MT5_ACCOUNT_TRADE_MODE_DEMO = 0
MT5_ACCOUNT_TRADE_MODE_CONTEST = 1
MT5_ACCOUNT_TRADE_MODE_REAL = 2

# Filename safety: reject path separators, Windows-reserved metacharacters,
# traversal sequences, and control characters. Persian/Unicode letters are
# legitimate in uploaded document names and remain allowed.
_UNSAFE_FILENAME_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]|\.\.')

# Windows reserved device names (case-insensitive, any extension) — writing
# to e.g. "nul" silently targets the NUL device instead of a file.
_WINDOWS_RESERVED_RE = re.compile(
    r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$", re.IGNORECASE)


def mask_login(login: Any) -> str:
    """Masks an account login number to prevent sensitive credential disclosure."""
    s = str(login).strip()
    if len(s) <= 2:
        return "**"
    return "*" * (len(s) - 2) + s[-2:]


def verify_account_is_demo(account_info: Any, expected_server: Optional[str] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """Strictly verifies that an account object is an authentic DEMO account.
    
    Returns:
        (is_valid_demo, reason, details_dict)
    """
    if account_info is None:
        return False, "Account info object is None. Terminal might be disconnected.", {}

    trade_mode = getattr(account_info, "trade_mode", None)
    server = getattr(account_info, "server", "")
    login = getattr(account_info, "login", 0)
    currency = getattr(account_info, "currency", "")
    leverage = getattr(account_info, "leverage", 0)
    balance = getattr(account_info, "balance", 0.0)
    equity = getattr(account_info, "equity", 0.0)

    details = {
        "server": server,
        "login_masked": mask_login(login),
        "currency": currency,
        "leverage": leverage,
        "balance": balance,
        "equity": equity,
        "trade_mode": trade_mode,
    }

    # Strict check: MT5 trade_mode must be explicitly 0 (DEMO)
    if trade_mode != MT5_ACCOUNT_TRADE_MODE_DEMO:
        if trade_mode == MT5_ACCOUNT_TRADE_MODE_REAL:
            return False, f"SECURITY ALERT: Account is REAL (trade_mode={trade_mode}). Execution permanently blocked.", details
        return False, f"Account trade_mode is {trade_mode}, which is not DEMO ({MT5_ACCOUNT_TRADE_MODE_DEMO}).", details

    # Optional server validation if expected_server is specified
    if expected_server and expected_server.lower() not in server.lower():
        return False, f"Server mismatch: current server '{server}' does not match expected '{expected_server}'.", details

    return True, "Account successfully verified as DEMO.", details


def assert_safe_to_execute_demo_order(
    current_mode: ExecutionMode,
    is_demo_enabled: bool,
    is_emergency_stop_active: bool,
    is_connected: bool,
    account_is_demo: bool,
) -> None:
    """Gatekeeper assertion before any demo order can be formed or sent.
    Fails closed on any inconsistency.
    """
    if is_emergency_stop_active:
        raise EmergencyStopActiveError("Execution blocked: Emergency Stop is currently ACTIVE.")

    if current_mode != ExecutionMode.DEMO_TRADING:
        raise SecurityViolationError(
            f"Execution blocked: System execution mode is '{current_mode.value}', not 'DEMO_TRADING'."
        )

    if not is_demo_enabled:
        raise SecurityViolationError(
            "Execution blocked: DEMO_TRADING_ENABLED gate is False in settings."
        )

    if not is_connected:
        raise SecurityViolationError("Execution blocked: MT5 terminal is not connected.")

    if not account_is_demo:
        raise AccountAuthenticationError(
            "Execution blocked: Account could not be verified as a genuine DEMO account."
        )


def build_contained_path(root, *segments: str) -> Path:
    """Build a filesystem path under `root` from validated segments (path
    traversal hardening, centralised so every dynamic file construction in
    the application enforces the same contract).

    Segments are joined with "_" and must be non-empty basenames: path
    separators, traversal sequences (".."), reserved metacharacters and
    control characters are refused BEFORE anything touches the filesystem.
    The resolved candidate must stay inside the resolved root — the return
    value is therefore provably contained and safe to open for I/O.

    Raises:
        ValueError: if any segment is empty, unsafe, or the joined path
            escapes `root`.
    """
    cleaned = [str(segment) for segment in segments if str(segment)]
    if not cleaned:
        raise ValueError("build_contained_path requires at least one non-empty segment")
    joined = "_".join(cleaned)
    if _UNSAFE_FILENAME_RE.search(joined):
        raise ValueError(f"Unsafe filename component: {joined!r}")
    for segment in cleaned:
        if _WINDOWS_RESERVED_RE.match(segment.split(".")[0] or segment):
            raise ValueError(f"Filename uses a Windows reserved device name: {segment!r}")
    root_resolved = Path(root).resolve()
    candidate = (root_resolved / joined).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ValueError(f"Resolved path escaped the storage root: {candidate}")
    return candidate
