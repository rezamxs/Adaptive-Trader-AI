"""Independent Risk Management Engine with Fail-Closed Constraints.
Validates all signal and trade proposals against strict safety thresholds.
"""
from datetime import datetime, timezone
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
        current_time_utc: Optional[datetime] = None,
        active_martingale_drawdown: float = 0.0,
        enforce_session_filter: bool = False,
        enforce_gold_min_equity: bool = True,
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

        # 9. GOLD MINIMUM CAPITAL CHECK (Strategy Rule - Page 4 & 11)
        # Gold scalping requires at least $5,000 equity; operating below $5,000 has catastrophic margin call risk.
        if enforce_gold_min_equity and symbol.upper() in ["XAUUSD", "GOLD"]:
            if account.equity < 5000.0:
                return False, 0.0, f"REJECTED_GOLD_MIN_EQUITY: Minimum equity of $5,000.00 required for Gold scalping (Current: ${account.equity:.2f}). Trading Gold under $5,000 is prohibited."

        # 10. MARTINGALE GRID LOSS CEILING (Strategy Rule - Page 14)
        # In grid phases, if floating drawdown reaches the $100-$200 threshold, reject new entries / enforce cutoff.
        if active_martingale_drawdown >= 200.0:
            return False, 0.0, f"REJECTED_MARTINGALE_LOSS_LIMIT: Active grid drawdown (${active_martingale_drawdown:.2f}) reached maximum safety cutoff ceiling ($200.00). Mandatory manual closure / loss acceptance required."

        # 11. SESSION HOURS & WEEKEND FILTERS (Strategy Rule - Page 14 & 15)
        if enforce_session_filter:
            eval_time = current_time_utc or datetime.now(timezone.utc)
            # Weekend gap protection: Friday closing / afternoon entries blocked
            if eval_time.weekday() == 4 and eval_time.hour >= 9:
                return False, 0.0, "REJECTED_FRIDAY_WEEKEND_RISK: New entries prohibited on Friday closing hours to avoid weekend gap exposure."

            # Asian + Early European window: 03:30 to 13:00 Iran time (00:00 to 09:30 UTC)
            hour_float = eval_time.hour + (eval_time.minute / 60.0)
            if not (0.0 <= hour_float <= 9.5):
                return False, 0.0, f"REJECTED_SESSION_OUTSIDE_ALLOWED_HOURS: Current time ({eval_time.strftime('%H:%M')} UTC) is outside permitted Asian/early European session (00:00-09:30 UTC / 03:30-13:00 Iran). NY session prohibited."

        return True, lots, f"APPROVED: {reason}"

