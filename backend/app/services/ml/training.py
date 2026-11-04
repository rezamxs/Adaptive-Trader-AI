"""Shared candidate-training flow (step 3.4: used by the HTTP endpoint AND the
nightly scheduler, so the two paths can never drift apart).

Raises typed exceptions; the HTTP endpoint maps them to statuses, the
scheduler logs them. NEVER promotes a model — activation stays a human
endpoint action.
"""
import math
from typing import Any, Dict

from backend.app.core.logging import logger
from backend.app.services.dataset.pipeline import DatasetPipeline
from backend.app.services.dataset.features import FeatureEngineer  # noqa: F401 (dataset contract)
from backend.app.services.market_data.provider_manager import (
    acquire_shared_provider,
    release_shared_provider,
)
from backend.app.services.ml.trainer import ModelTrainer


class TrainingDataError(Exception):
    """Environmental/training-input failure (MT5 unavailable, bad broker data,
    insufficient bars) — maps to 503/400 on the HTTP path."""


class TrainingInternalError(Exception):
    """Unexpected internal failure — maps to a sanitized 500 on the HTTP path."""


def train_candidate_from_market(
    db,
    symbol: str,
    timeframe: str,
    bars_count: int,
    horizon_bars: int,
    algorithm: str,
) -> Dict[str, Any]:
    """Fetches market data via MT5 and trains ONE candidate model.

    Same flow as the /ml/train endpoint (it calls this function) and the
    nightly scheduler. Raises TrainingDataError for environmental failures
    and TrainingInternalError for unexpected internal ones — never promotes.
    """
    provider = acquire_shared_provider()
    if not provider.connect():
        release_shared_provider(provider)
        raise TrainingDataError(
            "MT5 is unavailable; model training with synthetic fallback data is disabled."
        )
    try:
        df_raw = provider.get_historical_dataframe(symbol, timeframe, count=bars_count)
        symbol_info = provider.get_symbol_info(symbol)
    finally:
        release_shared_provider(provider)

    if symbol_info is None:
        raise TrainingDataError(f"MT5 did not return broker specifications for {symbol}.")
    # symbol_info.spread may be None/invalid on broker failures — convert once,
    # guarded, instead of leaking a raw TypeError as a 500 from either use site.
    try:
        broker_spread = int(symbol_info.spread)
    except (TypeError, ValueError, OverflowError):
        raise TrainingDataError(f"MT5 returned invalid point/spread data for {symbol}.")
    if (
        not math.isfinite(float(symbol_info.point))
        or float(symbol_info.point) <= 0
        or broker_spread < 0
    ):
        raise TrainingDataError(f"MT5 returned invalid point/spread data for {symbol}.")
    point_value = float(symbol_info.point)
    if len(df_raw) < 100:
        raise TrainingDataError("Insufficient bars to build training dataset.")

    try:
        ds_res = DatasetPipeline(db=db).build_dataset(
            df_raw,
            symbol=symbol,
            timeframe=timeframe,
            horizon_bars=horizon_bars,
            spread_points=max(0, broker_spread),
            point_value=point_value,
        )
    except Exception as e:
        logger.error(f"Dataset construction failed: {e}")
        raise TrainingInternalError("Dataset construction failed due to an internal error.")

    try:
        train_res = ModelTrainer(db=db).train_candidate(
            dataset_result=ds_res,
            algorithm=algorithm,
        )
    except Exception as outcome:
        logger.error(f"Model training failed: {outcome}")
        raise TrainingInternalError("Model training failed due to an internal error.")

    return train_res
