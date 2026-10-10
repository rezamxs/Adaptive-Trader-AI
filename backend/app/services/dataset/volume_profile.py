"""Volume Profile & Auction Market Value Area Calculator.
Computes Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL).
Accurately distributes candle volume across the price span [Low, High],
avoiding the discrete Close-only bias of legacy libraries.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


class VolumeProfileCalculator:
    """Calculates Point of Control (POC) and 70% Value Area (VAH/VAL) from OHLCV bar data."""

    @staticmethod
    def calculate_profile(
        df: pd.DataFrame,
        price_step: float = 0.10,
        value_area_pct: float = 0.70,
        volume_col: str = "tick_volume"
    ) -> Dict[str, Any]:
        """Calculates Volume Profile for a discrete slice of OHLCV bars.
        
        Args:
            df: DataFrame containing 'low', 'high', and volume columns.
            price_step: Bin size in price units (e.g., 0.10 for XAUUSD, 0.0001 for EURUSD).
            value_area_pct: Target percentage of volume for Value Area (default 0.70).
            volume_col: Name of volume column ('tick_volume' for MT5, or 'volume').
            
        Returns:
            Dict containing 'poc', 'val', 'vah', 'total_volume', and 'profile' Series.
        """
        if df.empty:
            return {
                "poc": None,
                "val": None,
                "vah": None,
                "total_volume": 0.0,
                "profile": pd.Series(dtype=float)
            }

        # Normalize column names
        lows = df["low"].values if "low" in df.columns else df["Low"].values
        highs = df["high"].values if "high" in df.columns else df["High"].values
        
        vol_col_name = volume_col
        if vol_col_name not in df.columns:
            if "tick_volume" in df.columns:
                vol_col_name = "tick_volume"
            elif "volume" in df.columns:
                vol_col_name = "volume"
            elif "Volume" in df.columns:
                vol_col_name = "Volume"
            else:
                vol_col_name = None

        if vol_col_name:
            volumes = df[vol_col_name].values.astype(float)
        else:
            # Fallback to unit volume
            volumes = np.ones(len(df), dtype=float)

        min_price = float(np.min(lows))
        max_price = float(np.max(highs))

        if price_step <= 0:
            price_step = 0.01

        min_bin = np.floor(min_price / price_step) * price_step
        max_bin = np.ceil(max_price / price_step) * price_step

        n_bins = int(np.round((max_bin - min_bin) / price_step)) + 1
        bin_edges = np.linspace(min_bin, max_bin, n_bins)
        bin_volumes = np.zeros(n_bins, dtype=float)

        # Distribute each bar's volume uniformly across its spanned price bins
        for l, h, v in zip(lows, highs, volumes):
            if v <= 0:
                continue
            if h <= l:
                # Zero-range / flat bar: assign volume to exact price level
                idx = int(np.round((l - min_bin) / price_step))
                idx = min(max(0, idx), n_bins - 1)
                bin_volumes[idx] += v
            else:
                start_idx = int(np.floor((l - min_bin) / price_step))
                end_idx = int(np.ceil((h - min_bin) / price_step))
                start_idx = max(0, start_idx)
                end_idx = min(n_bins - 1, end_idx)
                span_count = max(1, end_idx - start_idx + 1)
                bin_volumes[start_idx:end_idx + 1] += (v / span_count)

        profile_series = pd.Series(bin_volumes, index=np.round(bin_edges, 5))
        total_vol = float(profile_series.sum())

        if total_vol <= 0:
            return {
                "poc": None,
                "val": None,
                "vah": None,
                "total_volume": 0.0,
                "profile": profile_series
            }

        poc_idx = int(np.argmax(bin_volumes))
        poc_price = float(profile_series.index[poc_idx])

        # Dual-directional expansion from POC to enclose target volume (Value Area)
        target_vol = total_vol * value_area_pct
        cum_vol = bin_volumes[poc_idx]
        low_i = poc_idx
        high_i = poc_idx

        while cum_vol < target_vol and (low_i > 0 or high_i < n_bins - 1):
            next_low = bin_volumes[low_i - 1] if low_i > 0 else -1.0
            next_high = bin_volumes[high_i + 1] if high_i < n_bins - 1 else -1.0

            if next_high >= next_low and high_i < n_bins - 1:
                cum_vol += next_high
                high_i += 1
            elif low_i > 0:
                cum_vol += next_low
                low_i -= 1
            else:
                break

        val_price = float(profile_series.index[low_i])
        vah_price = float(profile_series.index[high_i])

        return {
            "poc": poc_price,
            "val": val_price,
            "vah": vah_price,
            "total_volume": total_vol,
            "profile": profile_series
        }

    @staticmethod
    def compute_rolling_profile_features(
        df: pd.DataFrame,
        window: int = 24,
        price_step: float = 0.10,
        value_area_pct: float = 0.70
    ) -> pd.DataFrame:
        """Computes rolling backward-looking Volume Profile features with ZERO lookahead bias.
        
        For bar i, the profile is strictly computed from bars [i - window : i] (prior closed bars),
        ensuring features at decision bar i never leak future price information.
        """
        df = df.copy()
        n = len(df)
        poc_arr = np.full(n, np.nan)
        val_arr = np.full(n, np.nan)
        vah_arr = np.full(n, np.nan)

        close = df["close"].values if "close" in df.columns else df["Close"].values

        for i in range(window, n):
            sub_slice = df.iloc[i - window : i]
            res = VolumeProfileCalculator.calculate_profile(
                sub_slice,
                price_step=price_step,
                value_area_pct=value_area_pct
            )
            poc_arr[i] = res["poc"] if res["poc"] is not None else np.nan
            val_arr[i] = res["val"] if res["val"] is not None else np.nan
            vah_arr[i] = res["vah"] if res["vah"] is not None else np.nan

        df["vp_poc"] = poc_arr
        df["vp_val"] = val_arr
        df["vp_vah"] = vah_arr
        df["dist_vp_poc"] = (close - poc_arr) / close
        df["dist_vp_vah"] = (close - vah_arr) / close
        df["dist_vp_val"] = (close - val_arr) / close
        df["in_value_area"] = ((close >= val_arr) & (close <= vah_arr)).astype(float)
        return df
