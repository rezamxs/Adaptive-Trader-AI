"""Independent Risk Management Engine with Fail-Closed Constraints.
Validates all signal and trade proposals against strict safety thresholds.
"""
from datetime import datetime, timezone
import math
from typing import Any, Callable, Dict, Optional, Tuple
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.logging import logger, log_security_event
from backend.app.services.market_data.base import Quote, SymbolInfo, AccountDetails
from backend.app.services.risk.circuit_breaker import EmergencyStopManager
from backend.app.services.risk.position_sizer import PositionSizer


class RiskEngine:
    """Evaluates trading proposals against comprehensive risk policies."""

    GRID_COMMENT_PREFIX = "Demo AI: "

    @staticmethod
    def compute_grid_martingale_drawdown(positions) -> Optional[float]:
        """Martingale-ceiling input = the floating loss of OUR demo-grid
        positions ONLY (identified by the broker comment prefix the executor
        writes), NOT the whole-account floating loss (manual positions on the
        same account are excluded — item 4).

        Returns None (fail closed) when the position list itself is
        unverifiable or a grid position carries an unusable profit value.
        Never substitutes zero for missing data.
        """
        if positions is None:
            return None
        total_floating = 0.0
        grid_positions = 0
        for position in positions:
            comment = str(getattr(position, "comment", "") or "")
            if RiskEngine.GRID_COMMENT_PREFIX not in comment:
                continue  # manual/unrelated position: not part of the grid
            grid_positions += 1
            profit = getattr(position, "profit", None)
            if profit is None:
                return None
            try:
                floating = float(profit)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(floating):
                return None
            total_floating += floating
        if grid_positions == 0:
            return 0.0
        return max(0.0, -total_floating)

    def __init__(self, db: Session, margin_estimator: Optional[Callable[..., Optional[float]]] = None):
        self.db = db
        self.margin_estimator = margin_estimator
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
        open_trades_count: Optional[int] = None,
        daily_loss_percent: Optional[float] = None,
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

        # Independently validate freshness. A provider's is_stale flag can be wrong.
        quote_time = getattr(quote, "time", None)
        if (
            not isinstance(quote_time, datetime)
            or quote_time.tzinfo is None
            or quote_time.utcoffset() is None
        ):
            return False, 0.0, "REJECTED_INVALID_QUOTE_TIME: Quote timestamp must be timezone-aware."

        age_seconds = (datetime.now(timezone.utc) - quote_time.astimezone(timezone.utc)).total_seconds()
        if age_seconds < -5.0:
            return False, 0.0, (
                f"REJECTED_FUTURE_DATA: Quote timestamp is {-age_seconds:.1f}s in the future."
            )
        if age_seconds > settings.WORKER_STALE_TICK_SECONDS:
            return False, 0.0, (
                f"REJECTED_STALE_DATA: Quote age {age_seconds:.1f}s exceeds "
                f"{settings.WORKER_STALE_TICK_SECONDS}s."
            )

        # Validate identifiers and all prices before any numeric risk decision.
        if quote.symbol.upper() != symbol.upper() or symbol_info.name.upper() != symbol.upper():
            return False, 0.0, "REJECTED_SYMBOL_MISMATCH: Quote or broker specification does not match requested symbol."
        price_values = (entry_price, stop_loss, take_profit, quote.bid, quote.ask, quote.spread_points)
        try:
            prices_are_finite = all(math.isfinite(float(value)) for value in price_values)
            numeric_prices = tuple(float(value) for value in price_values)
        except (TypeError, ValueError, OverflowError):
            return False, 0.0, "REJECTED_INVALID_PRICE: Price or spread contains a non-numeric value."
        if not prices_are_finite:
            return False, 0.0, "REJECTED_INVALID_PRICE: Price or spread contains a non-finite value."
        entry_value, sl_value, tp_value, bid_value, ask_value, spread_value = numeric_prices
        if min(entry_value, sl_value, tp_value, bid_value, ask_value) <= 0 or ask_value < bid_value or spread_value < 0:
            return False, 0.0, "REJECTED_INVALID_PRICE: Non-positive price or inverted quote."
        if direction not in {"BUY", "SELL"}:
            return False, 0.0, "REJECTED_INVALID_DIRECTION: Direction must be BUY or SELL."
        # Fail-closed on missing account context: AttributeError on a None
        # account would bypass the REJECTED contract instead of honoring it.
        if account is None:
            return False, 0.0, "REJECTED_INVALID_ACCOUNT: Account data is unavailable (fail-closed)."
        try:
            account_values = (account.equity, account.free_margin, account.leverage)
            if not all(math.isfinite(float(value)) for value in account_values):
                return False, 0.0, "REJECTED_INVALID_ACCOUNT: Equity, free margin, or leverage is non-finite."
            if float(account.equity) <= 0 or float(account.free_margin) < 0 or int(account.leverage) <= 0:
                return False, 0.0, "REJECTED_INVALID_ACCOUNT: Equity, free margin, or leverage is invalid."
        except (TypeError, ValueError, OverflowError):
            return False, 0.0, "REJECTED_INVALID_ACCOUNT: Equity, free margin, or leverage is invalid."

        # 3. SPREAD FILTER
        if quote.spread_points > settings.MAX_ALLOWED_SPREAD_POINTS:
            return False, 0.0, f"REJECTED_HIGH_SPREAD: Current spread ({quote.spread_points} pts) exceeds maximum permitted ({settings.MAX_ALLOWED_SPREAD_POINTS} pts)."

        # Unknown position/drawdown data must never be replaced by zero.
        if open_trades_count is None or daily_loss_percent is None:
            return False, 0.0, (
                "REJECTED_RISK_CONTEXT_UNAVAILABLE: Open positions and daily drawdown "
                "could not be verified from the account."
            )
        if (
            not isinstance(open_trades_count, int)
            or open_trades_count < 0
            or not math.isfinite(float(daily_loss_percent))
            or daily_loss_percent < 0
        ):
            return False, 0.0, "REJECTED_INVALID_RISK_CONTEXT: Position count or daily drawdown is invalid."

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
        if direction == "BUY" and take_profit <= entry_price:
            return False, 0.0, "REJECTED_INVALID_TP: Take profit must be strictly above entry price for BUY orders."
        if direction == "SELL" and take_profit >= entry_price:
            return False, 0.0, "REJECTED_INVALID_TP: Take profit must be strictly below entry price for SELL orders."

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

        # 8. MARGIN ADEQUACY CHECK. Production signal flow injects MT5's
        # order_calc_margin; fallback formula exists for isolated service tests.
        if self.margin_estimator is not None:
            try:
                estimated_margin = self.margin_estimator(symbol, direction, lots, entry_price)
            except Exception as margin_error:
                logger.warning(f"Broker margin estimate failed for {symbol}: {margin_error}")
                estimated_margin = None
            if estimated_margin is None:
                return False, 0.0, (
                    "REJECTED_MARGIN_ESTIMATE_UNAVAILABLE: Broker could not verify required margin."
                )
            try:
                estimated_margin = float(estimated_margin)
            except (TypeError, ValueError, OverflowError):
                return False, 0.0, "REJECTED_MARGIN_ESTIMATE_INVALID: Broker margin estimate is invalid."
            if not math.isfinite(estimated_margin) or estimated_margin < 0:
                return False, 0.0, "REJECTED_MARGIN_ESTIMATE_INVALID: Broker margin estimate is invalid."
        else:
            leverage = int(account.leverage or 0)
            contract_size = float(symbol_info.trade_contract_size or 0.0)
            if leverage <= 0 or contract_size <= 0:
                return False, 0.0, "REJECTED_MARGIN_ESTIMATE_UNAVAILABLE: Leverage or contract size is invalid."
            estimated_margin = lots * contract_size * entry_price / leverage

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

