"""Position Sizing and Post-Rounding Risk Validation."""
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
        if equity <= 0:
            return False, 0.0, "Account equity is zero or negative."

        sl_distance = abs(entry_price - stop_loss)
        if sl_distance <= 0:
            return False, 0.0, "Invalid stop loss: distance to entry price is zero."

        target_risk_amount = equity * (risk_percent / 100.0)
        hard_max_risk_amount = equity * (max_allowed_risk_percent / 100.0)

        # Risk per 1.0 standard lot
        contract_size = symbol_info.trade_contract_size or 100000.0
        risk_per_full_lot = sl_distance * contract_size

        if risk_per_full_lot <= 0:
            return False, 0.0, "Calculated risk per full lot is non-positive."

        raw_lots = target_risk_amount / risk_per_full_lot

        # Quantize to broker's volume step
        step = symbol_info.volume_step or 0.01
        min_lot = symbol_info.volume_min or 0.01
        max_lot = symbol_info.volume_max or 100.0

        quantized_lots = round(raw_lots / step) * step

        # If quantized is 0 or less than min_lot, check if min_lot violates max risk
        if quantized_lots < min_lot:
            min_lot_risk = min_lot * risk_per_full_lot
            if min_lot_risk > hard_max_risk_amount:
                return False, 0.0, f"Minimum lot size ({min_lot}) incurs risk (${min_lot_risk:.2f}) exceeding hard limit (${hard_max_risk_amount:.2f})."
            quantized_lots = min_lot

        # Clip to max_lot
        final_lots = min(max_lot, quantized_lots)
        final_lots = round(final_lots, 2)

        # POST-ROUNDING VERIFICATION:
        actual_risk_amount = final_lots * risk_per_full_lot
        if actual_risk_amount > hard_max_risk_amount:
            # Step down by one volume step
            stepped_down = round(final_lots - step, 2)
            if stepped_down >= min_lot:
                final_lots = stepped_down
                actual_risk_amount = final_lots * risk_per_full_lot
            else:
                return False, 0.0, f"Post-rounding risk (${actual_risk_amount:.2f}) exceeds hard limit (${hard_max_risk_amount:.2f})."

        return True, final_lots, f"Approved {final_lots} lots (Actual risk: ${actual_risk_amount:.2f} / {actual_risk_amount/equity*100:.2f}%)"
