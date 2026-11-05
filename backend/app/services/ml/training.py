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

    # Buglist item 4: produce + persist + attach per-fold-retrained ML
    # walk-forward evidence for the freshly registered candidate. Best-effort:
    # a short dataset (the API default is only 300 bars) or any evaluation
    # error must never fail the training itself — the outcome is reported
    # explicitly instead, and the promotion gate treats absent evidence the
    # same way it did before (the strategy-level walk-forward gate covers it).
    train_res["ml_walk_forward_evidence"] = produce_ml_walk_forward_evidence(
        db,
        train_res,
        df_raw,
        symbol=symbol,
        timeframe=timeframe,
        algorithm=algorithm,
        horizon_bars=horizon_bars,
        spread_points=max(0, broker_spread),
        point_value=point_value,
    )

    return train_res


def produce_ml_walk_forward_evidence(
    db,
    train_res: Dict[str, Any],
    df_raw,
    *,
    symbol: str,
    timeframe: str,
    algorithm: str,
    horizon_bars: int,
    spread_points: int,
    point_value: float,
) -> Dict[str, Any]:
    """Runs, persists, and attaches ML walk-forward evidence for one candidate.

    The evaluation reuses the EXACT data and configuration the candidate was
    trained with (same raw frame, same algorithm/hyperparameters, same label
    horizon and cost parameters), so the attach-time configuration binding
    (buglist item 5) can never reject its own producer's output. Any failure
    is reported as {"status": "skipped", "reason": ...} — evidence production
    must not be able to break training.
    """
    from backend.app.core.config import settings
    from backend.app.services.ml.walk_forward import run_ml_walk_forward
    from backend.app.services.ml.registry import ModelRegistry

    fold_count = int(getattr(settings, "ML_WALK_FORWARD_FOLDS", 4) or 0)
    if fold_count < 2:
        return {"status": "disabled", "reason": "ML_WALK_FORWARD_FOLDS < 2"}

    try:
        payload = run_ml_walk_forward(
            df_raw,
            symbol=symbol,
            timeframe=timeframe,
            algorithm=algorithm,
            hyperparameters=dict(train_res.get("hyperparameters") or {}),
            fold_count=fold_count,
            horizon_bars=int(horizon_bars),
            spread_points=int(spread_points),
            point_value=float(point_value),
        )
        registry = ModelRegistry(db)
        run_id = registry.persist_ml_walk_forward_evaluation(payload)
        registry.attach_ml_walk_forward_metrics(
            train_res["model_id"], payload, run_id=run_id
        )
        logger.info(
            "ML walk-forward evidence recorded for %s (run_id=%s, folds=%d, "
            "oos_auc=%.4f)",
            train_res["model_id"], run_id, payload["fold_count"],
            payload["aggregate"]["roc_auc"]["mean"],
        )
        return {
            "status": "evidence_recorded",
            "run_id": run_id,
            "fold_count": payload["fold_count"],
            "oos_roc_auc_mean": payload["aggregate"]["roc_auc"]["mean"],
        }
    except Exception as exc:
        logger.warning(
            "ML walk-forward evidence skipped for %s: %s: %s",
            train_res.get("model_id"), type(exc).__name__, exc,
        )
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
