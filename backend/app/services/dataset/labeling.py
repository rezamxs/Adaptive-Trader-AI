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
        n = len(df)
        labels = np.full(n, fill_value=np.nan)
        close_prices = df["close"].values
        high_prices = df["high"].values
        low_prices = df["low"].values
        atrs = df[atr_col].values if atr_col in df.columns else np.full(n, 0.0010)

        spread_cost = spread_points * point_value

        for i in range(n - max_horizon_bars):
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
