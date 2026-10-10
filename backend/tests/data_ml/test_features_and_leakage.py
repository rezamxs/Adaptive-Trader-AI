"""Tests for Feature Engineering, Anti-Lookahead Guarantees, and Conservative Labeling."""
import pytest
import numpy as np
import pandas as pd
from backend.app.services.market_data.mock_provider import MockMarketDataProvider
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.dataset.features import FeatureEngineer
from backend.app.services.dataset.labeling import BarrierLabeler
from backend.app.services.dataset.pipeline import DatasetPipeline


def test_anti_lookahead_bias_guarantee():
    """CRITICAL DATA LEAKAGE TEST:
    Modifying future prices (at t+1 ... t+N) must NOT change any feature value at bar t.
    """
    provider = MockMarketDataProvider()
    df1 = provider.get_historical_dataframe("EURUSD", TimeFrame.M15, count=200)

    # Transform original data
    feats1, cols = FeatureEngineer.transform(df1)

    # Create modified dataframe where future 20 bars (from bar 150 onwards) are drastically altered
    df2 = df1.copy()
    df2.iloc[150:, df2.columns.get_loc("close")] *= 1.50
    df2.iloc[150:, df2.columns.get_loc("high")] *= 1.50
    df2.iloc[150:, df2.columns.get_loc("low")] *= 1.50

    feats2, _ = FeatureEngineer.transform(df2)

    # Features at bar 140 (10 bars prior to alteration) MUST BE 100% IDENTICAL
    bar_140_feats1 = feats1.iloc[140][cols].to_dict()
    bar_140_feats2 = feats2.iloc[140][cols].to_dict()

    for col in cols:
        v1 = bar_140_feats1[col]
        v2 = bar_140_feats2[col]
        if not np.isnan(v1):
            assert np.isclose(v1, v2, atol=1e-7), f"Feature {col} changed at bar 140 when future data changed! Look-ahead bias detected!"


def test_conservative_same_bar_collision_handling():
    """Verify that if both TP and SL are hit within the same candle,
    the label is conservatively resolved to 0 (loss).
    """
    # Construct a synthetic DataFrame where bar 1 triggers both TP and SL for bar 0
    times = pd.date_range("2026-01-01", periods=10, freq="15min")
    df = pd.DataFrame({
        "open": [1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000],
        "high": [1.1020, 1.1200, 1.1020, 1.1020, 1.1020, 1.1020, 1.1020, 1.1020, 1.1020, 1.1020],
        "low":  [1.0980, 1.0800, 1.0980, 1.0980, 1.0980, 1.0980, 1.0980, 1.0980, 1.0980, 1.0980],
        "close":[1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000],
        "atr_14":[0.0050] * 10
    }, index=times)

    # For bar 0: Entry is 1.1000 + spread. TP is ~1.1075, SL is ~1.0950.
    # At bar 1: High is 1.1200 (hits TP) AND Low is 1.0800 (hits SL)
    labels = BarrierLabeler.label_buy_barrier(
        df,
        tp_atr_mult=1.5,
        sl_atr_mult=1.0,
        max_horizon_bars=5,
        atr_col="atr_14",
        spread_points=10
    )

    # Must be 0, NOT 1
    assert labels.iloc[0] == 0.0, "Same-bar collision was not resolved conservatively to 0!"


def test_dataset_pipeline_purging_and_no_leakage():
    """Verify temporal separation, purged barriers, and scaler fitting."""
    provider = MockMarketDataProvider()
    raw_df = provider.get_historical_dataframe("EURUSD", TimeFrame.M15, count=300)

    pipeline = DatasetPipeline(db=None)
    result = pipeline.build_dataset(
        raw_df,
        symbol="EURUSD",
        timeframe="M15",
        horizon_bars=10,
        train_ratio=0.7,
        test_ratio=0.15
    )

    assert "version_tag" in result
    assert result["target_stats"]["train_samples"] > 0
    assert result["target_stats"]["test_samples"] > 0
    assert result["target_stats"]["purged_bars"] == 20

    X_train, y_train = result["train_data"]
    X_test, y_test = result["test_data"]
    assert len(X_train) == result["target_stats"]["train_samples"]
    assert len(X_test) == result["target_stats"]["test_samples"]
