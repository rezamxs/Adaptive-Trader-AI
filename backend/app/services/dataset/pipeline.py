"""Dataset Pipeline Orchestrator with Temporal Purged/Embargoed Splits."""
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.db.base import utc_now_naive
from backend.app.models.dataset import DatasetVersion
from backend.app.services.dataset.features import FeatureEngineer
from backend.app.services.dataset.labeling import BarrierLabeler


class DatasetPipeline:
    """Manages feature creation, conservative labeling, purged temporal splits, and persistence."""

    def __init__(self, db: Optional[Session] = None):
        self.db = db

    def build_dataset(
        self,
        raw_df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        horizon_bars: int = 24,
        tp_mult: float = 1.5,
        sl_mult: float = 1.0,
        train_ratio: float = 0.70,
        test_ratio: float = 0.15,
        version_prefix: str = "ds",
        spread_points: Optional[int] = None,
        point_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """Builds an ML-ready dataset with strict temporal separation and purging."""
        if len(raw_df) < settings.MIN_DATA_POINTS_REQUIRED:
            raise ValueError(f"Insufficient data: {len(raw_df)} bars provided, minimum required is {settings.MIN_DATA_POINTS_REQUIRED}.")
        if not isinstance(raw_df.index, pd.DatetimeIndex):
            raise ValueError("Raw market data must use a DatetimeIndex for temporal splitting.")
        if raw_df.index.has_duplicates or not raw_df.index.is_monotonic_increasing:
            raise ValueError("Raw market data timestamps must be unique and sorted in ascending order.")
        required_ohlc = ["open", "high", "low", "close"]
        missing_ohlc = [column for column in required_ohlc if column not in raw_df.columns]
        if missing_ohlc:
            raise ValueError(f"Raw market data is missing OHLC columns: {missing_ohlc}.")
        ohlc = raw_df[required_ohlc].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(ohlc).all() or (ohlc <= 0).any():
            raise ValueError("Raw OHLC values must be finite and strictly positive.")
        if (ohlc[:, 1] < ohlc[:, 2]).any():
            raise ValueError("Raw market data has candles where high is below low.")
        if (ohlc[:, 1] < np.maximum(ohlc[:, 0], ohlc[:, 3])).any():
            raise ValueError("Raw market data contains open/close prices above candle high.")
        if (ohlc[:, 2] > np.minimum(ohlc[:, 0], ohlc[:, 3])).any():
            raise ValueError("Raw market data contains open/close prices below candle low.")
        if horizon_bars < 1:
            raise ValueError("horizon_bars must be at least 1.")
        if not (0.0 < train_ratio < 1.0 and 0.0 < test_ratio < 1.0 and train_ratio + test_ratio < 1.0):
            raise ValueError("train_ratio and test_ratio must be positive and sum to less than 1.")

        # Symbol-aware point value and spread defaults
        is_gold = "XAU" in symbol.upper() or "GOLD" in symbol.upper()
        effective_point = point_value if point_value is not None else (0.01 if is_gold else 0.00001)
        effective_spread = spread_points if spread_points is not None else (17 if is_gold else 15)

        # 1. Feature Engineering (purely backward-looking)
        df_feat, feature_names = FeatureEngineer.transform(raw_df)

        # 2. Target Labeling (forward-looking horizon)
        target_series = BarrierLabeler.label_buy_barrier(
            df_feat,
            tp_atr_mult=tp_mult,
            sl_atr_mult=sl_mult,
            max_horizon_bars=horizon_bars,
            spread_points=effective_spread,
            point_value=effective_point
        )
        df_feat["target"] = target_series

        # 3. Clean warmup, invalid labels, and any infinite feature values.
        # dropna alone does not remove +/-inf, which can break scaler/model fitting.
        finite_features = df_feat[feature_names].replace([np.inf, -np.inf], np.nan)
        df_feat.loc[:, feature_names] = finite_features
        clean_df = df_feat.dropna(subset=feature_names + ["target"]).copy()
        clean_df["target"] = clean_df["target"].astype(int)

        n_clean = len(clean_df)
        if n_clean < 50:
            raise ValueError(f"Cleaned dataset has only {n_clean} rows after dropping warmup/horizon NaNs.")

        # 4. Temporal Split with Purge / Embargo
        # Allocate ratios only across usable rows; reserve two separate purge buffers
        # between train/validation and validation/test. Never let slicing shorten a
        # partition into an overlap or silently consume the validation partition.
        purge_window = int(horizon_bars)
        usable_rows = n_clean - (2 * purge_window)
        if usable_rows < 50:
            raise ValueError(
                f"Insufficient data after purge buffers: {n_clean} clean rows with "
                f"{purge_window} rows reserved at each boundary."
            )

        n_train = int(usable_rows * train_ratio)
        n_test = int(usable_rows * test_ratio)
        n_val = usable_rows - n_train - n_test
        if min(n_train, n_val, n_test) < 1:
            raise ValueError("Temporal split produced an empty train, validation, or test partition.")

        train_end_idx = n_train
        val_start_idx = train_end_idx + purge_window
        val_end_idx = val_start_idx + n_val
        test_start_idx = val_end_idx + purge_window

        train_df = clean_df.iloc[:train_end_idx].copy()
        val_df = clean_df.iloc[val_start_idx:val_end_idx].copy()
        test_df = clean_df.iloc[test_start_idx:].copy()
        if min(len(train_df), len(val_df), len(test_df)) < 1:
            raise ValueError("Temporal split produced an empty partition after purging.")

        # 5. Fit Scaler ONLY on Training partition
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_names])
        X_val = scaler.transform(val_df[feature_names]) if len(val_df) > 0 else np.empty((0, len(feature_names)))
        X_test = scaler.transform(test_df[feature_names]) if len(test_df) > 0 else np.empty((0, len(feature_names)))

        # 6. Save Processed Artifacts with a deterministic fingerprint. Rebuilding
        # identical data/settings yields the same dataset version.
        version_payload = (
            pd.util.hash_pandas_object(clean_df, index=True).values.tobytes()
            + (
                f"|{symbol}|{timeframe}|{horizon_bars}|{tp_mult}|{sl_mult}|"
                f"{train_ratio}|{test_ratio}|{effective_spread}|{effective_point}"
            ).encode("utf-8")
        )
        version_hash = hashlib.sha256(version_payload).hexdigest()[:8]
        version_tag = f"{version_prefix}_{symbol.lower()}_{timeframe.lower()}_{version_hash}"
        save_path = settings.DATA_PROCESSED_DIR / f"{version_tag}.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(save_path)

        # 7. Calculate Statistics
        target_counts = clean_df["target"].value_counts().to_dict()
        positive_rate = float(target_counts.get(1, 0)) / n_clean if n_clean > 0 else 0.0

        target_stats = {
            "total_samples": n_clean,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "purged_bars": purge_window * 2,
            "train_ratio_requested": train_ratio,
            "test_ratio_requested": test_ratio,
            "positive_label_count": int(target_counts.get(1, 0)),
            "negative_label_count": int(target_counts.get(0, 0)),
            "positive_rate": round(positive_rate, 4),
            "spread_points": effective_spread,
            "point_value": effective_point,
        }

        # 8. Record in DB if session available
        if self.db:
            start_t = clean_df.index[0] if isinstance(clean_df.index[0], datetime) else utc_now_naive()
            end_t = clean_df.index[-1] if isinstance(clean_df.index[-1], datetime) else utc_now_naive()
            # Naive-UTC DB contract (db/base.py): the MT5 provider emits
            # tz-aware UTC stamps and the SQLite bind silently strips tzinfo,
            # so normalize explicitly — a non-UTC aware index would otherwise
            # be stored with silently shifted wall-clock values.
            if getattr(start_t, "tzinfo", None) is not None:
                start_t = start_t.astimezone(timezone.utc).replace(tzinfo=None)
            if getattr(end_t, "tzinfo", None) is not None:
                end_t = end_t.astimezone(timezone.utc).replace(tzinfo=None)
            
            existing_record = self.db.query(DatasetVersion).filter_by(version_tag=version_tag).first()
            if existing_record is None:
                record = DatasetVersion(
                    version_tag=version_tag,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_t,
                    end_time=end_t,
                    total_bars=n_clean,
                    features_list=feature_names,
                    target_definition=f"BUY TP({tp_mult}*ATR) before SL({sl_mult}*ATR) within {horizon_bars} bars [spread={effective_spread}pts, pt={effective_point}]",
                    target_stats=target_stats,
                    file_path=str(save_path),
                )
                self.db.add(record)
                self.db.commit()
                logger.info(f"Recorded dataset version {version_tag} in database.")
            else:
                # Deterministic rebuilds are idempotent; do not violate the unique key.
                logger.info(f"Reusing existing dataset version {version_tag}.")

        return {
            "version_tag": version_tag,
            "file_path": str(save_path),
            "feature_names": feature_names,
            "target_stats": target_stats,
            "train_data": (X_train, train_df["target"].values),
            "val_data": (X_val, val_df["target"].values),
            "test_data": (X_test, test_df["target"].values),
            "scaler": scaler,
            "clean_df": clean_df,
        }
