"""Custom application exceptions for Adaptive Trader AI.
All trading and security exceptions fail closed.
"""

class AdaptiveTraderError(Exception):
    """Base exception for all system errors."""
    pass


class SecurityViolationError(AdaptiveTraderError):
    """Raised when an operation attempts an insecure, unauthorized, or live trading action."""
    pass


class InvalidExecutionModeError(SecurityViolationError):
    """Raised when an unsupported or prohibited execution mode is requested."""
    pass


class LiveTradingProhibitedError(SecurityViolationError):
    """Raised if any attempt is made to configure or trigger live trading."""
    def __init__(self, message: str = "Live trading is strictly prohibited in this version. Only BACKTEST, SIGNAL_ONLY, and DEMO_TRADING are permitted."):
        super().__init__(message)


class EmergencyStopActiveError(AdaptiveTraderError):
    """Raised when an operation is blocked because the emergency stop is active."""
    pass


class AccountAuthenticationError(SecurityViolationError):
    """Raised when MT5 account validation or demo proof fails."""
    pass


class RiskLimitExceededError(AdaptiveTraderError):
    """Raised when a proposed trade or operation violates risk parameters."""
    pass


class MarketDataUnavailableError(AdaptiveTraderError):
    """Raised when MT5 connection is down or market rates are stale/missing."""
    pass


class RuleValidationError(AdaptiveTraderError):
    """Raised when a trading rule definition is invalid or malformed."""
    pass


class ModelGovernanceError(AdaptiveTraderError):
    """Raised when model promotion or versioning validation fails."""
    pass
