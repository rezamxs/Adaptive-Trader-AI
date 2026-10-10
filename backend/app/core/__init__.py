"""Core configuration, security, errors, and logging module."""
from backend.app.core.config import ExecutionMode, settings
from backend.app.core.errors import (
    AdaptiveTraderError,
    SecurityViolationError,
    EmergencyStopActiveError,
    AccountAuthenticationError,
    RiskLimitExceededError,
)
from backend.app.core.security import verify_account_is_demo, mask_login
from backend.app.core.logging import logger, log_security_event

__all__ = [
    "ExecutionMode",
    "settings",
    "AdaptiveTraderError",
    "SecurityViolationError",
    "EmergencyStopActiveError",
    "AccountAuthenticationError",
    "RiskLimitExceededError",
    "verify_account_is_demo",
    "mask_login",
    "logger",
    "log_security_event",
]
