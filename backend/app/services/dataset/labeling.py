"""Conservative Barrier Labeling for ML Strategy Training.
Evaluates realistic TP/SL barrier touches and handles same-bar collisions conservatively.
"""
from typing import Optional, Tuple
import numpy as np
import pandas as pd


class BarrierLabeler:
    """Labels events based on barrier arrival order with conservative collision handling."""

    @staticmethod
    def label_buy_barrier(
        df: pd.DataFrame,
        tp_atr_mult: float = 1.5,
        sl_atr_mult: float = 1.0,
        max_horizon_bars: int = 24,
        atr_col: str = "atr_14",
        spread_points: int = 15,
        point_value: float = 0.00001
    ) -> pd.Series:
        """Labels BUY opportunities:
        1: Take Profit reached strictly before Stop Loss within horizon.
        0: Stop Loss hit first, same-bar collision, or horizon expired without reaching TP.
        """
        if max_horizon_bars < 1:
            raise ValueError("max_horizon_bars must be at least 1.")
        if not np.isfinite(point_value) or point_value <= 0:
            raise ValueError("point_value must be finite and positive.")
        if spread_points < 0:
            raise ValueError("spread_points must not be negative.")

        n = len(df)
        labels = np.full(n, fill_value=np.nan)
        close_prices = df["close"].values
        high_prices = df["high"].values
        low_prices = df["low"].values
        if atr_col not in df.columns:
            # Fail loud instead of a hardcoded 0.0010 fallback ATR: on XAUUSD
            # (~$2000) that barrier is sub-tick and degenerates every label
            # to 0 with no warning — silently corrupt training targets.
            raise ValueError(
                f"Barrier labeling requires the ATR column '{atr_col}' in the frame; "
                "refusing to substitute a hardcoded fallback ATR."
            )
        atrs = df[atr_col].values

        # Prefer historical per-bar spread, available in MT5 rates, over today's
        # broker spread. This avoids labeling all history with the current spread.
        if "spread" in df.columns:
            try:
                historical_spreads = pd.to_numeric(df["spread"], errors="coerce").to_numpy(dtype=float)
            except (TypeError, ValueError):
                historical_spreads = np.full(n, float(spread_points))
        else:
            historical_spreads = np.full(n, float(spread_points))

        for i in range(n - max_horizon_bars):
            bar_spread = historical_spreads[i]
            if not np.isfinite(bar_spread) or bar_spread < 0:
                bar_spread = float(spread_points)
            spread_cost = bar_spread * point_value
            entry_price = close_prices[i] + spread_cost  # Pay ask price to buy
            atr = atrs[i]
            if np.isnan(atr) or atr <= 0:
                continue

            take_profit = entry_price + (tp_atr_mult * atr)
            stop_loss = entry_price - (sl_atr_mult * atr)

            label = 0  # Default to 0 (conservative)
            for h in range(1, max_horizon_bars + 1):
                future_idx = i + h
                future_high = high_prices[future_idx]
                future_low = low_prices[future_idx]

                tp_hit = (future_high >= take_profit)
                sl_hit = (future_low <= stop_loss)

                if tp_hit and sl_hit:
                    # CONSERVATIVE COLLISION RESOLUTION:
                    # If both boundaries breached within the same bar, conservatively assume
                    # stop loss was hit first.
                    label = 0
                    break
                elif sl_hit:
                    label = 0
                    break
                elif tp_hit:
                    label = 1
                    break

            labels[i] = label

        return pd.Series(labels, index=df.index, name="target_tp_before_sl")
