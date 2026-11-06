"""True per-fold ML walk-forward validation (M8).

The strategy walk-forward (backend/app/services/backtest/walk_forward.py)
attests out-of-sample robustness of the APPROVED-RULE STRATEGY with real
costs; it deliberately does not retrain the ML model per fold. This module
produces the model-quality evidence that attestation lacks: a chronological
walk-forward in which a FRESH model (and a fresh baseline) is fitted on every
fold's training partition and evaluated only on the fold's purged, embargoed,
chronologically-later out-of-sample window.

Leakage contract (what makes each fold honest):
- Features (FeatureEngineer) are causal per bar and are computed once over
  the full frame — every fold's bars are still strictly out-of-sample.
- Labels look `horizon_bars` ahead (currently 24). A training row whose label
  horizon overlaps the test window would leak its OUTCOME into training, so
  training ends `purge_window` (= horizon_bars) before the test window.
- `embargo_bars` adds a further information gap after the purge buffer
  (López de Prado style), defaulting to half the horizon.
- The StandardScaler is fitted on each fold's training partition ONLY and
  merely transforms that fold's test partition.
- The decision threshold is the fixed model default (0.5); nothing is tuned
  on the out-of-sample window.
- The evidence payload records fold boundaries (positions + timestamps),
  per-fold metrics for the model AND a dummy baseline, aggregate statistics,
  the dataset fingerprint, feature configuration, model hyperparameters,
  transaction-cost parameters, and the code version.

Stored payloads are re-verified independently by
`verify_ml_walk_forward_payload` (fold geometry, purge/embargo gaps, and
aggregate recomputation) plus an HMAC/SHA-256 digest
(`backend.app.core.metrics_integrity.ml_walk_forward_digest`) — matching
numbers inside one payload alone prove nothing.
"""
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.app.core.logging import logger
from backend.app.core.metrics_integrity import (
    ml_walk_forward_digest,
    verify_ml_walk_forward_digest,
)
from backend.app.services.dataset.features import FeatureEngineer
from backend.app.services.dataset.labeling import BarrierLabeler
from backend.app.services.ml.evaluator import ModelEvaluator
from backend.app.services.ml.models import (
    BaselineModel,
    IStrategyModel,
    LogisticRegressionStrategyModel,
    RandomForestStrategyModel,
)

MLWF_EVALUATOR = "backend/app/services/ml/walk_forward.py::run_ml_walk_forward"
MLWF_EVIDENCE_TYPE = "ml_walk_forward"
MLWF_THRESHOLD = 0.5  # fixed model default; never tuned on OOS


def _default_code_version() -> str:
    """Buglist item 35: evaluations recorded code_version="unrecorded" when
    the caller passed no value, making exact experiment reconstruction
    impossible. The DEFAULT is now a real, resolvable version string (the
    installed package version + the running Python), and only a total
    resolution failure keeps an explicit unrecorded marker."""
    try:
        from importlib.metadata import version as _pkg_version

        pkg = _pkg_version("adaptive-trader-ai")
    except Exception:
        pkg = None
    import sys

    py = f"py{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if pkg:
        return f"adaptive-trader-ai@{pkg};{py}"
    return f"unknown-package;{py}"
DEFAULT_HORIZON_BARS = 24


def _dataset_fingerprint(df: pd.DataFrame) -> Dict[str, Any]:
    """Deterministic fingerprint of the evaluated (cleaned) frame."""
    return {
        "rows": int(len(df)),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "sha256": hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest(),
    }


def _validate_raw_frame(df_raw: pd.DataFrame) -> None:
    if not isinstance(df_raw, pd.DataFrame) or not isinstance(df_raw.index, pd.DatetimeIndex):
        raise ValueError("Raw market data must be a DataFrame with a DatetimeIndex.")
    if df_raw.index.has_duplicates or not df_raw.index.is_monotonic_increasing:
        raise ValueError("Raw market data timestamps must be unique and sorted ascending.")
    missing = [c for c in ("open", "high", "low", "close") if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Raw market data is missing OHLC columns: {missing}.")
    # Same OHLC sanity contract as DatasetPipeline.build_dataset: dirty bars
    # would silently poison barrier labels (NaN comparisons label rows 0).
    ohlc = df_raw[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(ohlc).all() or (ohlc <= 0).any():
        raise ValueError("Raw OHLC values must be finite and strictly positive.")
    if (ohlc[:, 1] < ohlc[:, 2]).any():
        raise ValueError("Raw market data has candles where high is below low.")


def build_ml_walk_forward_folds(
    n_rows: int,
    fold_count: int,
    purge_window: int,
    embargo_bars: int,
    min_train_bars: int,
) -> List[Dict[str, int]]:
    """Chronological contiguous OOS windows with purged expanding-window training.

    The out-of-sample windows are the last `fold_count` contiguous blocks of
    the frame (the last fold absorbs the remainder). Training for fold k is
    the expanding prefix [0, test_start_k - purge_window - embargo_bars).
    Returns position dicts; raises ValueError on impossible geometry so a
    caller can never silently evaluate a degenerate fold.
    """
    if fold_count < 2:
        raise ValueError("ML walk-forward evaluation requires at least 2 folds.")
    if purge_window < 0 or embargo_bars < 0:
        raise ValueError("purge_window and embargo_bars must be non-negative.")
    if purge_window == 0 and embargo_bars == 0:
        raise ValueError("Refusing folds with no purge/embargo gap: label leakage is not detectable afterwards.")
    gap = purge_window + embargo_bars
    if min_train_bars < 10:
        raise ValueError("min_train_bars must be at least 10 rows per fold.")
    usable_tail = n_rows - min_train_bars
    if usable_tail < fold_count:
        raise ValueError(
            f"Not enough rows ({n_rows}) for {fold_count} OOS folds after reserving "
            f"{min_train_bars} training bars; need at least {min_train_bars + fold_count}."
        )

    test_len = usable_tail // fold_count
    folds: List[Dict[str, int]] = []
    for k in range(fold_count):
        test_start = min_train_bars + k * test_len
        test_end = n_rows if k == fold_count - 1 else test_start + test_len
        train_end = test_start - gap
        if train_end < 1:
            raise ValueError(
                f"Fold {k + 1} training partition collapsed (train_end={train_end}); "
                "increase the dataset or reduce purge/embargo windows."
            )
        folds.append({
            "fold_index": k + 1,
            "train_start_pos": 0,
            "train_end_pos": int(train_end),   # exclusive
            "test_start_pos": int(test_start),  # inclusive
            "test_end_pos": int(test_end),      # exclusive
        })
    return folds


def _mean_std(values: List[float]) -> Dict[str, float]:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    mean = float(np.mean(clean))
    std = float(np.std(clean)) if len(clean) > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(std, 4), "n": len(clean)}


def _model_factory(algorithm: str, hyperparameters: Dict[str, Any]) -> IStrategyModel:
    if algorithm == "logistic_regression":
        return LogisticRegressionStrategyModel(**hyperparameters)
    if algorithm == "random_forest":
        return RandomForestStrategyModel(**hyperparameters)
    raise ValueError(
        f"Unsupported algorithm '{algorithm}'. Choose 'logistic_regression' or 'random_forest'."
    )


def run_ml_walk_forward(
    df_raw: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    algorithm: str = "logistic_regression",
    hyperparameters: Optional[Dict[str, Any]] = None,
    fold_count: int = 4,
    horizon_bars: int = DEFAULT_HORIZON_BARS,
    purge_window: Optional[int] = None,
    embargo_bars: Optional[int] = None,
    min_train_bars: int = 200,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    spread_points: int = 17,
    point_value: float = 0.01,
    code_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs the per-fold-retrained ML walk-forward and returns the evidence payload.

    The payload is intentionally plain data (JSON-safe): attach it to a model
    record via ModelRegistry.attach_ml_walk_forward_metrics, which re-verifies
    it and commits a tamper-evident digest.
    """
    _validate_raw_frame(df_raw)
    hyperparameters = hyperparameters or {}
    purge = int(purge_window) if purge_window is not None else int(horizon_bars)
    embargo = int(embargo_bars) if embargo_bars is not None else max(1, int(horizon_bars) // 2)
    if purge < horizon_bars:
        # The label of a training row reads `horizon_bars` bars ahead; a purge
        # window smaller than that lets the tail of the training partition
        # learn from outcomes inside the OOS window (label leakage).
        raise ValueError(
            f"purge_window ({purge}) must be >= horizon_bars ({horizon_bars}) "
            "so no training label can reach into the out-of-sample window."
        )

    # 1. Features (causal) and labels (forward horizon) computed once over the
    #    full frame; cleaning mirrors DatasetPipeline.build_dataset.
    df_feat, feature_names = FeatureEngineer.transform(df_raw)
    target_series = BarrierLabeler.label_buy_barrier(
        df_feat,
        tp_atr_mult=tp_mult,
        sl_atr_mult=sl_mult,
        max_horizon_bars=horizon_bars,
        spread_points=spread_points,
        point_value=point_value,
    )
    df_feat["target"] = target_series
    df_feat.loc[:, feature_names] = df_feat[feature_names].replace([np.inf, -np.inf], np.nan)
    clean_df = df_feat.dropna(subset=feature_names + ["target"]).copy()
    clean_df["target"] = clean_df["target"].astype(int)
    if len(clean_df) < min_train_bars + fold_count:
        raise ValueError(
            f"Cleaned dataset has {len(clean_df)} usable rows; need at least "
            f"{min_train_bars + fold_count} for {fold_count} folds."
        )

    folds = build_ml_walk_forward_folds(
        len(clean_df), fold_count, purge, embargo, min_train_bars
    )

    # 2. Per-fold: fit scaler + FRESH model + FRESH baseline on the training
    #    partition only; evaluate on the fold's OOS window.
    fold_records: List[Dict[str, Any]] = []
    for fold in folds:
        train_slice = clean_df.iloc[fold["train_start_pos"]:fold["train_end_pos"]]
        test_slice = clean_df.iloc[fold["test_start_pos"]:fold["test_end_pos"]]
        if len(train_slice) < 10 or len(test_slice) < 1:
            raise ValueError(f"Fold {fold['fold_index']} has a degenerate partition.")

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_slice[feature_names])
        X_test = scaler.transform(test_slice[feature_names])
        y_train = train_slice["target"].to_numpy()
        y_test = test_slice["target"].to_numpy()

        train_classes = np.unique(y_train)
        if len(train_classes) != 2:
            raise ValueError(
                f"Fold {fold['fold_index']} training partition is single-class "
                f"({train_classes.tolist()}); extend the training window or data."
            )

        model = _model_factory(algorithm, dict(hyperparameters))
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)
        metrics = ModelEvaluator.evaluate(y_test, pred, prob)

        baseline = BaselineModel(strategy="prior")
        baseline.fit(X_train, y_train)
        b_pred = baseline.predict(X_test)
        b_prob = baseline.predict_proba(X_test)
        baseline_metrics = ModelEvaluator.evaluate(y_test, b_pred, b_prob)

        fold_records.append({
            "fold_index": fold["fold_index"],
            "train_start": str(clean_df.index[fold["train_start_pos"]]),
            "train_end": str(clean_df.index[fold["train_end_pos"] - 1]),
            "test_start": str(clean_df.index[fold["test_start_pos"]]),
            "test_end": str(clean_df.index[fold["test_end_pos"] - 1]),
            "train_end_pos": fold["train_end_pos"],
            "test_start_pos": fold["test_start_pos"],
            "test_end_pos": fold["test_end_pos"],
            "train_samples": int(len(y_train)),
            "oos_samples": int(len(y_test)),
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
        })

    # 3. Aggregate statistics (recomputed independently by the verifier).
    aggregate = {
        "roc_auc": _mean_std([f["metrics"].get("roc_auc") for f in fold_records]),
        "brier_score": _mean_std([f["metrics"].get("brier_score") for f in fold_records]),
        "f1": _mean_std([f["metrics"].get("f1") for f in fold_records]),
        "accuracy": _mean_std([f["metrics"].get("accuracy") for f in fold_records]),
        "baseline_roc_auc": _mean_std([f["baseline_metrics"].get("roc_auc") for f in fold_records]),
        "folds_beating_baseline_roc_auc": int(sum(
            1 for f in fold_records
            if (f["metrics"].get("roc_auc") or 0.0) > (f["baseline_metrics"].get("roc_auc") or 0.0)
        )),
        "total_oos_samples": int(sum(f["oos_samples"] for f in fold_records)),
    }

    payload: Dict[str, Any] = {
        "evidence_type": MLWF_EVIDENCE_TYPE,
        "evaluator": MLWF_EVALUATOR,
        "generated_at_epoch_s": int(time.time()),
        "symbol": symbol,
        "timeframe": timeframe,
        "algorithm": algorithm,
        "hyperparameters": dict(hyperparameters),
        "feature_names": list(feature_names),
        "horizon_bars": int(horizon_bars),
        "purge_window": int(purge),
        "embargo_bars": int(embargo),
        "min_train_bars": int(min_train_bars),
        "label_definition": (
            f"BUY TP({tp_mult}*ATR) before SL({sl_mult}*ATR) within {horizon_bars} bars "
            f"[spread={spread_points}pts, pt={point_value}]"
        ),
        "threshold": MLWF_THRESHOLD,
        "threshold_policy": "fixed model default; no tuning on validation or OOS windows",
        "fold_scheme": (
            "chronological contiguous out-of-sample windows (last absorbs remainder); "
            "expanding-window training ending purge_window+embargo_bars before each test start; "
            "scaler fitted on each fold's training partition only; fresh model and fresh "
            "baseline fitted per fold"
        ),
        "fold_count": int(fold_count),
        "folds": fold_records,
        "aggregate": aggregate,
        "dataset_fingerprint": _dataset_fingerprint(clean_df),
        "transaction_costs": {
            "spread_points": int(spread_points),
            "point_value": float(point_value),
            "note": "labels are cost-aware (barrier labeler); backtest P&L evidence is the separate strategy walk-forward",
        },
        # Buglist item 35: the default is a RESOLVED version string, never
        # a bare "unrecorded" — only an explicit caller override replaces it.
        "code_version": code_version or _default_code_version(),
    }
    logger.info(
        f"ML walk-forward: {fold_count} folds, algorithm={algorithm}, "
        f"oos_roc_auc={aggregate['roc_auc']['mean']} vs baseline={aggregate['baseline_roc_auc']['mean']}"
    )
    return payload


def verify_ml_walk_forward_payload(payload: Any) -> Optional[str]:
    """Independently re-verifies a stored ML walk-forward payload.

    Returns None when the payload is structurally valid and internally
    consistent; otherwise a failure reason. This is the anti-forgery layer
    beyond the digest: fold geometry is recomputed from the recorded
    positions, purge/embargo gaps are enforced, and every aggregate is
    recomputed from the per-fold records.
    """
    if not isinstance(payload, dict):
        return "payload is not a dict"
    if payload.get("evidence_type") != MLWF_EVIDENCE_TYPE:
        return f"evidence_type must be '{MLWF_EVIDENCE_TYPE}'"
    if payload.get("evaluator") != MLWF_EVALUATOR:
        return "payload does not name the producing evaluator"

    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) < 2:
        return "at least 2 fold records are required"
    if payload.get("fold_count") != len(folds):
        return "fold_count contradicts the number of fold records"

    purge = payload.get("purge_window")
    embargo = payload.get("embargo_bars")
    horizon = payload.get("horizon_bars")
    if not isinstance(purge, int) or not isinstance(embargo, int) or purge < 0 or embargo < 0:
        return "purge_window/embargo_bars must be non-negative integers"
    if not isinstance(horizon, int) or horizon < 1:
        return "horizon_bars must be a positive integer"
    if purge < horizon:
        return (
            f"purge_window ({purge}) is smaller than the label horizon ({horizon}): "
            "training labels whose outcome window reaches into the OOS bars would "
            "leak their results into the fit — such evidence is not acceptable"
        )
    if purge + embargo < 1:
        return "payload claims zero purge+embargo gap: label leakage cannot be excluded"

    previous_test_end = None
    seen_fold_indices = set()
    for expected_index, fold in enumerate(folds, start=1):
        if not isinstance(fold, dict):
            return "fold record is not a dict"
        if fold.get("fold_index") != expected_index:
            return (
                f"fold_index sequence broken at position {expected_index}: "
                f"got {fold.get('fold_index')!r} (duplicated or reordered records)"
            )
        seen_fold_indices.add(expected_index)
        try:
            train_end = int(fold["train_end_pos"])
            test_start = int(fold["test_start_pos"])
            test_end = int(fold["test_end_pos"])
        except (KeyError, TypeError, ValueError):
            return f"fold {fold.get('fold_index')} lacks integer boundary positions"
        if not (0 < train_end <= test_start < test_end):
            return f"fold {fold.get('fold_index')} boundaries are not chronological"
        if train_end + purge + embargo > test_start:
            return (
                f"fold {fold.get('fold_index')} purge/embargo violated: training ends at "
                f"{train_end} but the test window starts at {test_start} with "
                f"purge={purge}+embargo={embargo}"
            )
        if previous_test_end is not None and test_start < previous_test_end:
            return f"fold {fold.get('fold_index')} overlaps the previous out-of-sample window"
        previous_test_end = test_end
        if not isinstance(fold.get("metrics"), dict) or not isinstance(fold.get("baseline_metrics"), dict):
            return f"fold {fold.get('fold_index')} lacks model/baseline metrics"

    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        return "aggregate block missing"
    for key, fold_key in (
        ("roc_auc", "roc_auc"),
        ("brier_score", "brier_score"),
        ("f1", "f1"),
        ("accuracy", "accuracy"),
        ("baseline_roc_auc", "roc_auc"),
    ):
        fold_values = [
            (f["baseline_metrics"] if key == "baseline_roc_auc" else f["metrics"]).get(fold_key)
            for f in folds
        ]
        recomputed = _mean_std(fold_values)
        claimed = aggregate.get(key)
        if not isinstance(claimed, dict):
            return f"aggregate.{key} missing"
        try:
            claimed_mean = float(claimed.get("mean", 0.0))
            claimed_std = float(claimed.get("std", 0.0))
        except (TypeError, ValueError):
            return f"aggregate.{key}.mean/std must be numeric (forged or corrupt aggregate)"
        if abs(claimed_mean - recomputed["mean"]) > 0.0011:
            return f"aggregate.{key}.mean does not match the per-fold records (forged aggregate)"
        if abs(claimed_std - recomputed["std"]) > 0.0011:
            return f"aggregate.{key}.std does not match the per-fold records (forged aggregate)"

    beating = int(sum(
        1 for f in folds
        if (f["metrics"].get("roc_auc") or 0.0) > (f["baseline_metrics"].get("roc_auc") or 0.0)
    ))
    if aggregate.get("folds_beating_baseline_roc_auc") != beating:
        return "aggregate.folds_beating_baseline_roc_auc does not match the fold records"
    total_oos = int(sum(
        int(f.get("oos_samples", 0))
        if isinstance(f.get("oos_samples"), int) and not isinstance(f.get("oos_samples"), bool)
        else 0
        for f in folds
    ))
    if aggregate.get("total_oos_samples") != total_oos:
        return "aggregate.total_oos_samples does not match the fold records"

    for key in ("feature_names", "hyperparameters", "dataset_fingerprint", "code_version"):
        if key not in payload:
            return f"provenance key '{key}' missing"
    if payload.get("threshold") != MLWF_THRESHOLD:
        return "threshold must be the fixed default; OOS threshold tuning is not acceptable evidence"
    return None


def verify_ml_walk_forward_evidence(
    model_id: str,
    payload: Any,
    digest: Optional[str],
) -> Optional[str]:
    """Full fail-closed verification: structure/consistency + digest."""
    structural_failure = verify_ml_walk_forward_payload(payload)
    if structural_failure:
        return structural_failure
    if not verify_ml_walk_forward_digest(model_id, payload, digest):
        return "ml_walk_forward_metrics digest mismatch (tampered or unattested payload)"
    return None
