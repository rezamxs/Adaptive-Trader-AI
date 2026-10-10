"""Unit tests for VolumeProfileCalculator and Auction Market Value Area.
Validates mathematical correctness, edge case handling, zero look-ahead bias,
and comparison against discrete Close-only method.
"""
import numpy as np
import pandas as pd
import pytest

from backend.app.services.dataset.volume_profile import VolumeProfileCalculator


@pytest.fixture
def sample_ohlcv():
    """Generates a controlled, deterministic synthetic OHLCV dataset."""
    np.random.seed(123)
    n = 60
    prices = 2650.0 + np.cumsum(np.random.randn(n) * 0.4)
    data = {
        "open": prices,
        "high": prices + np.abs(np.random.randn(n) * 0.6) + 0.1,
        "low": prices - np.abs(np.random.randn(n) * 0.6) - 0.1,
        "close": prices + np.random.randn(n) * 0.3,
        "tick_volume": np.random.randint(50, 500, n),
    }
    df = pd.DataFrame(data)
    # Ensure High is strictly >= max(Open, Close) and Low <= min(Open, Close)
    df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
    df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))
    return df


def test_volume_profile_basic_structure(sample_ohlcv):
    """Verifies that POC, VAL, and VAH adhere to mathematical ordering: VAL <= POC <= VAH."""
    res = VolumeProfileCalculator.calculate_profile(sample_ohlcv, price_step=0.20, value_area_pct=0.70)
    
    assert res["poc"] is not None
    assert res["val"] is not None
    assert res["vah"] is not None
    assert res["total_volume"] > 0
    assert res["val"] <= res["poc"] <= res["vah"]
    
    # Value Area should capture ~70% of total volume
    profile = res["profile"]
    va_mask = (profile.index >= res["val"]) & (profile.index <= res["vah"])
    va_vol = profile[va_mask].sum()
    va_ratio = va_vol / res["total_volume"]
    assert 0.65 <= va_ratio <= 0.85


def test_volume_profile_empty_dataframe():
    """Verifies graceful handling of empty inputs."""
    empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "tick_volume"])
    res = VolumeProfileCalculator.calculate_profile(empty_df)
    assert res["poc"] is None
    assert res["val"] is None
    assert res["vah"] is None
    assert res["total_volume"] == 0.0


def test_volume_profile_flat_candles():
    """Verifies handling of flat candles where high == low without division-by-zero."""
    df = pd.DataFrame({
        "low": [100.0, 100.0, 100.0],
        "high": [100.0, 100.0, 100.0],
        "close": [100.0, 100.0, 100.0],
        "tick_volume": [10, 20, 30]
    })
    res = VolumeProfileCalculator.calculate_profile(df, price_step=1.0)
    assert res["poc"] == 100.0
    assert res["val"] == 100.0
    assert res["vah"] == 100.0
    assert res["total_volume"] == 60.0


def test_zero_lookahead_bias(sample_ohlcv):
    """Guarantees that rolling volume profile features at bar i depend strictly on past data [i-W : i]."""
    df1 = sample_ohlcv.copy()
    window = 20
    feat1 = VolumeProfileCalculator.compute_rolling_profile_features(df1, window=window, price_step=0.20)
    
    # Save values at test index (e.g. bar 35)
    test_idx = 35
    poc_orig = feat1.loc[test_idx, "vp_poc"]
    val_orig = feat1.loc[test_idx, "vp_val"]
    vah_orig = feat1.loc[test_idx, "vp_vah"]
    
    # Mutate future bars (e.g. bars 40 to 59)
    df2 = sample_ohlcv.copy()
    df2.loc[40:, "high"] = df2.loc[40:, "high"] + 50.0
    df2.loc[40:, "tick_volume"] = 9999
    
    feat2 = VolumeProfileCalculator.compute_rolling_profile_features(df2, window=window, price_step=0.20)
    
    # Assert values at bar 35 are 100% identical despite extreme future alterations
    assert feat2.loc[test_idx, "vp_poc"] == poc_orig
    assert feat2.loc[test_idx, "vp_val"] == val_orig
    assert feat2.loc[test_idx, "vp_vah"] == vah_orig


def test_side_by_side_comparison_with_discrete_method():
    """Compares range allocation against discrete close-only allocation on a known benchmark."""
    # Bar where price traversed from 10 to 20, but closed at 20
    df = pd.DataFrame({
        "low": [10.0],
        "high": [20.0],
        "close": [20.0],
        "tick_volume": [100.0]
    })
    
    # Range-distributed calculator
    res_range = VolumeProfileCalculator.calculate_profile(df, price_step=1.0)
    
    # Discrete method (allocates 100% to Close=20.0)
    # The range calculator distributes volume across bins [10.0, ..., 20.0]
    profile = res_range["profile"]
    assert profile.loc[10.0] > 0
    assert profile.loc[15.0] > 0
    assert profile.loc[20.0] > 0
    assert np.isclose(profile.sum(), 100.0)


def test_volume_profile_all_zero_volume():
    """Verifies that an OHLCV dataset where all volume is zero returns None for POC/VAL/VAH gracefully."""
    df = pd.DataFrame({
        "low": [100.0, 101.0, 102.0],
        "high": [105.0, 106.0, 107.0],
        "close": [103.0, 104.0, 105.0],
        "tick_volume": [0, 0, 0]
    })
    res = VolumeProfileCalculator.calculate_profile(df, price_step=1.0)
    assert res["poc"] is None
    assert res["val"] is None
    assert res["vah"] is None
    assert res["total_volume"] == 0.0


def test_volume_profile_tie_breaking_deterministic():
    """Verifies that when two distinct bins share identical peak volume, tie-breaking is deterministic (lowest index)."""
    df = pd.DataFrame({
        "low": [10.0, 20.0],
        "high": [10.0, 20.0],
        "close": [10.0, 20.0],
        "tick_volume": [50.0, 50.0]
    })
    res = VolumeProfileCalculator.calculate_profile(df, price_step=1.0)
    assert res["total_volume"] == 100.0
    # argmax selects first occurrence (10.0)
    assert res["poc"] == 10.0


def test_volume_profile_very_tight_range():
    """Verifies profile calculation when candle range is smaller than price_step."""
    df = pd.DataFrame({
        "low": [100.01],
        "high": [100.03],
        "close": [100.02],
        "tick_volume": [250.0]
    })
    res = VolumeProfileCalculator.calculate_profile(df, price_step=1.0)
    assert res["total_volume"] == 250.0
    assert res["poc"] == 100.0
    assert res["val"] == 100.0
    assert res["vah"] == 101.0


def test_volume_profile_volume_conservation():
    """Verifies that total volume in calculated profile matches sum of bar volumes exactly."""
    np.random.seed(42)
    n = 25
    df = pd.DataFrame({
        "low": 100.0 + np.random.rand(n) * 5.0,
        "high": 105.0 + np.random.rand(n) * 5.0,
        "tick_volume": np.random.randint(10, 100, n).astype(float)
    })
    df["high"] = np.maximum(df["high"], df["low"] + 0.1)
    res = VolumeProfileCalculator.calculate_profile(df, price_step=0.25)
    expected_sum = df["tick_volume"].sum()
    actual_sum = res["total_volume"]
    assert np.isclose(expected_sum, actual_sum)
    assert np.isclose(res["profile"].sum(), expected_sum)

