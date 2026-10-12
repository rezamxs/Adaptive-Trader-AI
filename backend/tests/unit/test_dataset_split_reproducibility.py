"""Dataset builds must be reproducible and keep purge gaps between partitions."""
import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.services.dataset.pipeline import DatasetPipeline


def _bars(count=600):
    rng = np.random.default_rng(1729)
    close = 2650.0 + np.cumsum(rng.normal(0.0, 0.5, count))
    open_price = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_price, close) + rng.uniform(0.05, 0.35, count)
    low = np.minimum(open_price, close) - rng.uniform(0.05, 0.35, count)
    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": rng.integers(100, 1000, count),
            "spread": np.full(count, 20),
        },
        index=pd.date_range("2026-01-01", periods=count, freq="min", tz="UTC"),
    )


def test_dataset_version_is_deterministic_and_partitions_respect_purge(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_PROCESSED_DIR", tmp_path)
    raw = _bars()
    pipeline = DatasetPipeline()

    first = pipeline.build_dataset(
        raw, symbol="XAUUSD", timeframe="M1", horizon_bars=10,
        spread_points=20, point_value=0.01,
    )
    second = pipeline.build_dataset(
        raw, symbol="XAUUSD", timeframe="M1", horizon_bars=10,
        spread_points=20, point_value=0.01,
    )

    assert first["version_tag"] == second["version_tag"]
    stats = first["target_stats"]
    assert stats["train_samples"] > 0
    assert stats["val_samples"] > 0
    assert stats["test_samples"] > 0
    assert (
        stats["train_samples"]
        + stats["val_samples"]
        + stats["test_samples"]
        + stats["purged_bars"]
        == stats["total_samples"]
    )
    assert first["file_path"] == second["file_path"]
