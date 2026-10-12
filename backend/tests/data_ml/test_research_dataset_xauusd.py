"""Tests verifying the integrity and quality of the XAUUSD M1 ML Research Dataset."""
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from backend.app.core.config import settings

SPEC_PATH = settings.DATA_PROCESSED_DIR / "dataset_spec_xauusd_m1.json"
DATASET_PATH = settings.DATA_PROCESSED_DIR / "research_xauusd_m1_latest.csv"


def test_research_dataset_spec_exists_and_valid():
    """Verify specification JSON artifact is present and adheres to structure."""
    assert SPEC_PATH.exists(), f"Dataset spec not found at {SPEC_PATH}"
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        spec = json.load(f)
        
    assert spec["symbol"] == "XAUUSD"
    assert spec["timeframe"] == "M1"
    assert spec["broker_timezone"] == "UTC+3"
    assert spec["target_specification"]["direction"] == "BUY"
    assert spec["target_specification"]["tp_multiplier_atr"] == 1.5
    assert spec["target_specification"]["sl_multiplier_atr"] == 1.0
    assert spec["target_specification"]["horizon_bars"] == 24
    assert spec["feature_specification"]["feature_count"] == 24
    assert spec["partition_specification"]["total_clean_samples"] > 90000


def test_research_dataset_file_integrity():
    """Verify dataset CSV has correct columns, no NaNs, and binary target."""
    assert DATASET_PATH.exists(), f"Processed CSV not found at {DATASET_PATH}"
    df = pd.read_csv(DATASET_PATH, index_col=0, parse_dates=True)
    
    assert len(df) > 90000
    assert df.index.is_monotonic_increasing
    assert df.index.duplicated().sum() == 0
    
    # Check nulls
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected NaN values!"
    
    # Target integrity
    assert "target" in df.columns
    unique_targets = set(df["target"].unique())
    assert unique_targets.issubset({0, 1})
    pos_rate = df["target"].mean()
    assert 0.20 <= pos_rate <= 0.50, f"Unrealistic positive rate: {pos_rate}"


def test_dataset_features_presence():
    """Verify all 24 required features are present."""
    df = pd.read_csv(DATASET_PATH, nrows=10, index_col=0)
    expected_features = [
        "return_1", "return_3", "return_5", "log_return_1",
        "candle_range", "body_to_range", "upper_shadow_ratio", "lower_shadow_ratio", "is_bullish_bar",
        "rsi_14", "atr_14", "atr_pct_14",
        "ema_9", "ema_21", "sma_50", "dist_ema_9", "dist_ema_21", "dist_sma_50", "ema_9_21_diff",
        "volatility_20", "hour_sin", "hour_cos", "day_sin", "day_cos"
    ]
    for feat in expected_features:
        assert feat in df.columns, f"Missing feature: {feat}"
