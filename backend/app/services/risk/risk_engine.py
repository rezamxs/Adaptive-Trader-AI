"""Independent Risk Management Engine with Fail-Closed Constraints.
Validates all signal and trade proposals against strict safety thresholds.
"""
from typing import Any, Dict, Optional, Tuple
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.logging import logger, log_security_event
from backend.app.services.market_data.base import Quote, SymbolInfo, AccountDetails
from backend.app.services.risk.circuit_breaker import EmergencyStopManager
from backend.app.services.risk.position_sizer import PositionSizer


class RiskEngine:
    """Evaluates trading proposals against comprehensive risk policies."""

    def __init__(self, db: Session):
        self.db = db
        self.emergency_stop = EmergencyStopManager(db)

    def assess_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        account: AccountDetails,
        symbol_info: SymbolInfo,
        quote: Quote,
        open_trades_count: int = 0,
        daily_loss_percent: float = 0.0,
    ) -> Tuple[bool, float, str]:
        """Comprehensive risk gatekeeper check.
        
        Returns:
            (is_approved, calculated_lots, reason_or_status)
        """
        # 1. EMERGENCY STOP CHECK
        if self.emergency_stop.is_active():
            return False, 0.0, "REJECTED_EMERGENCY_STOP: Emergency stop circuit breaker is currently ACTIVE."

        # 2. DATA QUALITY & STALENESS CHECK
        if quote is None or quote.is_stale:
            return False, 0.0, "REJECTED_STALE_DATA: Market quote is stale or unavailable."

        # 3. SPREAD FILTER
        if quote.spread_points > settings.MAX_ALLOWED_SPREAD_POINTS:
            return False, 0.0, f"REJECTED_HIGH_SPREAD: Current spread ({quote.spread_points} pts) exceeds maximum permitted ({settings.MAX_ALLOWED_SPREAD_POINTS} pts)."

        # 4. MAX OPEN TRADES LIMIT
        if open_trades_count >= settings.MAX_OPEN_TRADES:
            return False, 0.0, f"REJECTED_MAX_TRADES: Open trades count ({open_trades_count}) has reached the maximum ceiling ({settings.MAX_OPEN_TRADES})."

        # 5. DAILY LOSS / DRAWDOWN LIMIT
        if daily_loss_percent >= settings.MAX_DAILY_LOSS_PERCENT:
            # Trigger Emergency Stop automatically upon daily drawdown breach!
            self.emergency_stop.activate(
                reason=f"Daily drawdown limit breached: {daily_loss_percent:.2f}% >= {settings.MAX_DAILY_LOSS_PERCENT:.2f}%",
                actor="RISK_ENGINE"
            )
            return False, 0.0, f"REJECTED_DAILY_DRAWDOWN: Daily drawdown ({daily_loss_percent:.2f}%) breached limit ({settings.MAX_DAILY_LOSS_PERCENT:.2f}%). Emergency stop triggered."

        # 6. STOP LOSS SANITY CHECK
        if direction == "BUY" and stop_loss >= entry_price:
            return False, 0.0, "REJECTED_INVALID_SL: Stop loss must be strictly below entry price for BUY orders."
        if direction == "SELL" and stop_loss <= entry_price:
            return False, 0.0, "REJECTED_INVALID_SL: Stop loss must be strictly above entry price for SELL orders."

        # 7. POSITION SIZING & POST-ROUNDING VALIDATION
        is_ok, lots, reason = PositionSizer.calculate_and_verify_lots(
            equity=account.equity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            symbol_info=symbol_info,
            risk_percent=settings.DEFAULT_RISK_PER_TRADE_PERCENT,
            max_allowed_risk_percent=settings.DEFAULT_RISK_PER_TRADE_PERCENT * 1.5
        )

        if not is_ok:
            return False, 0.0, f"REJECTED_POSITION_SIZING: {reason}"

        # 8. MARGIN ADEQUACY CHECK
        estimated_margin = (lots * (symbol_info.trade_contract_size or 100000.0)) / (account.leverage or 1)
        if estimated_margin > account.free_margin:
            return False, 0.0, f"REJECTED_INSUFFICIENT_MARGIN: Estimated required margin (${estimated_margin:.2f}) exceeds free margin (${account.free_margin:.2f})."

        return True, lots, f"APPROVED: {reason}"
