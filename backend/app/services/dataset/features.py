"""Feature Engineering module with strict zero look-ahead bias guarantees.
All features are computed solely from data available at or before the decision bar.
"""
from typing import List, Tuple
import numpy as np
import pandas as pd


class FeatureEngineer:
    """Computes technical, price-action, and volatility features from OHLCV data."""

    @staticmethod
    def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Computes backward-looking price returns."""
        df = df.copy()
        df["return_1"] = df["close"].pct_change(1)
        df["return_3"] = df["close"].pct_change(3)
        df["return_5"] = df["close"].pct_change(5)
        df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
        return df

    @staticmethod
    def compute_candle_geometry(df: pd.DataFrame) -> pd.DataFrame:
        """Computes candle body, shadows, and range relative to price."""
        df = df.copy()
        high = df["high"]
        low = df["low"]
        close = df["close"]
        open_p = df["open"]

        candle_range = (high - low).replace(0, 1e-6)
        body = (close - open_p).abs()
        upper_shadow = high - np.maximum(open_p, close)
        lower_shadow = np.minimum(open_p, close) - low

        df["candle_range"] = candle_range
        df["body_to_range"] = body / candle_range
        df["upper_shadow_ratio"] = upper_shadow / candle_range
        df["lower_shadow_ratio"] = lower_shadow / candle_range
        df["is_bullish_bar"] = (close > open_p).astype(int)
        return df

    @staticmethod
    def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Computes Wilder's Relative Strength Index (RSI)."""
        df = df.copy()
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().replace(0, 1e-6)

        rs = avg_gain / avg_loss
        df[f"rsi_{period}"] = 100.0 - (100.0 / (1.0 + rs))
        return df

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Computes Average True Range (ATR)."""
        df = df.copy()
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df[f"atr_{period}"] = tr.rolling(window=period, min_periods=period).mean()
        df[f"atr_pct_{period}"] = df[f"atr_{period}"] / df["close"]
        return df

    @staticmethod
    def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Computes exponential and simple moving averages and relative price distances."""
        df = df.copy()
        close = df["close"]

        ema_9 = close.ewm(span=9, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        sma_50 = close.rolling(window=50, min_periods=50).mean()

        df["ema_9"] = ema_9
        df["ema_21"] = ema_21
        df["sma_50"] = sma_50

        # Distance features (normalized by ATR or price)
        df["dist_ema_9"] = (close - ema_9) / close
        df["dist_ema_21"] = (close - ema_21) / close
        df["dist_sma_50"] = (close - sma_50) / close
        df["ema_9_21_diff"] = (ema_9 - ema_21) / close
        return df

    @staticmethod
    def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Computes rolling price volatility."""
        df = df.copy()
        returns = df["close"].pct_change()
        df[f"volatility_{window}"] = returns.rolling(window=window, min_periods=window).std()
        return df

    @staticmethod
    def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extracts cyclical hour-of-day and day-of-week session features."""
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            # If not DatetimeIndex, try converting or pass
            return df

        hour = df.index.hour
        dayofweek = df.index.dayofweek

        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df["day_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)
        return df

    @classmethod
    def transform(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Executes full feature engineering pipeline.
        
        Returns:
            (feature_dataframe, list_of_feature_column_names)
        """
        initial_cols = set(df.columns)
        
        df = cls.compute_returns(df)
        df = cls.compute_candle_geometry(df)
        df = cls.compute_rsi(df, period=14)
        df = cls.compute_atr(df, period=14)
        df = cls.compute_moving_averages(df)
        df = cls.compute_volatility(df, window=20)
        df = cls.compute_time_features(df)

        feature_cols = [c for c in df.columns if c not in initial_cols]
        return df, feature_cols
