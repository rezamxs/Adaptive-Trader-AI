"""Position Sizing and Post-Rounding Risk Validation."""
import math
from decimal import Decimal
from typing import Tuple
from backend.app.core.logging import logger
from backend.app.services.market_data.base import SymbolInfo


class PositionSizer:
    """Calculates safe order lot sizing and performs post-rounding risk checks."""

    @staticmethod
    def calculate_and_verify_lots(
        equity: float,
        entry_price: float,
        stop_loss: float,
        symbol_info: SymbolInfo,
        risk_percent: float = 1.0,
        max_allowed_risk_percent: float = 2.0
    ) -> Tuple[bool, float, str]:
        """Calculates order volume and verifies post-rounding risk bounds.
        
        Returns:
            (is_approved, calculated_lots, reason_or_status)
        """
        try:
            numeric_inputs = (
                float(equity), float(entry_price), float(stop_loss),
                float(risk_percent), float(max_allowed_risk_percent),
            )
        except (TypeError, ValueError, OverflowError):
            return False, 0.0, "Risk sizing inputs must be numeric."
        if not all(math.isfinite(value) for value in numeric_inputs):
            return False, 0.0, "Risk sizing inputs must be finite."
        equity, entry_price, stop_loss, risk_percent, max_allowed_risk_percent = numeric_inputs
        if equity <= 0:
            return False, 0.0, "Account equity is zero or negative."

        sl_distance = abs(entry_price - stop_loss)
        if sl_distance <= 0:
            return False, 0.0, "Invalid stop loss: distance to entry price is zero."

        if risk_percent <= 0 or max_allowed_risk_percent < risk_percent:
            return False, 0.0, "Risk percentages are invalid."

        target_risk_amount = equity * (risk_percent / 100.0)
        hard_max_risk_amount = equity * (max_allowed_risk_percent / 100.0)

        # Prefer broker tick economics. MT5 reports tick value in the account
        # deposit currency, which is more appropriate for metals and CFDs.
        tick_size = float(symbol_info.tick_size or 0.0)
        tick_value = float(symbol_info.tick_value or 0.0)
        if tick_size > 0 and tick_value > 0:
            risk_per_full_lot = (sl_distance / tick_size) * tick_value
        else:
            # Compatibility fallback for providers without valid tick economics.
            contract_size = float(symbol_info.trade_contract_size or 0.0)
            if contract_size <= 0:
                return False, 0.0, "Missing valid tick economics and contract size."
            risk_per_full_lot = sl_distance * contract_size

        if not math.isfinite(risk_per_full_lot) or risk_per_full_lot <= 0:
            return False, 0.0, "Calculated risk per full lot is non-positive or invalid."

        raw_lots = target_risk_amount / risk_per_full_lot

        # Round DOWN: nearest-step rounding may exceed the desired risk target.
        step = float(symbol_info.volume_step or 0.01)
        min_lot = float(symbol_info.volume_min or 0.01)
        max_lot = float(symbol_info.volume_max or 100.0)
        if step <= 0 or min_lot <= 0 or max_lot < min_lot:
            return False, 0.0, "Broker volume limits are invalid."

        step_precision = max(0, -Decimal(str(step)).normalize().as_tuple().exponent)
        effective_max_lot = math.floor((max_lot / step) + 1e-12) * step
        effective_max_lot = round(effective_max_lot, step_precision)
        if effective_max_lot < min_lot:
            return False, 0.0, "Broker maximum volume cannot satisfy minimum volume and step constraints."
        quantized_lots = math.floor((raw_lots / step) + 1e-12) * step
        quantized_lots = round(quantized_lots, step_precision)

        # If quantized is 0 or less than min_lot, check if min_lot violates max risk
        if quantized_lots < min_lot:
            min_lot_risk = min_lot * risk_per_full_lot
            if min_lot_risk > hard_max_risk_amount:
                return False, 0.0, f"Minimum lot size ({min_lot}) incurs risk (${min_lot_risk:.2f}) exceeding hard limit (${hard_max_risk_amount:.2f})."
            quantized_lots = min_lot

        # Clip to max_lot
        final_lots = min(effective_max_lot, quantized_lots)
        final_lots = round(final_lots, step_precision)

        # POST-ROUNDING VERIFICATION:
        actual_risk_amount = final_lots * risk_per_full_lot
        if actual_risk_amount > hard_max_risk_amount:
            # Step down by one volume step
            stepped_down = round(final_lots - step, step_precision)
            if stepped_down >= min_lot:
                final_lots = stepped_down
                actual_risk_amount = final_lots * risk_per_full_lot
            else:
                return False, 0.0, f"Post-rounding risk (${actual_risk_amount:.2f}) exceeds hard limit (${hard_max_risk_amount:.2f})."

        return True, final_lots, f"Approved {final_lots} lots (Actual risk: ${actual_risk_amount:.2f} / {actual_risk_amount/equity*100:.2f}%)"
