"""Buglist 2026-11-05 round: regression tests for items 1, 3, 4, 5.

Item 1 — walk-forward profit factor must be a gross-profit/gross-loss RATIO
         (99.0 sentinel when profitable with no loss), never a raw money amount.
Item 3 — the walk-forward aggregate drawdown is the CONTINUOUS stitched-account
         figure, re-derived by the promotion verifier from the persisted fold
         records; max-of-folds evidence (pre-fix rows) fails closed.
Item 4 — the shared training flow PRODUCES + persists + attaches per-fold
         retrained ML walk-forward evidence for every new candidate.
Item 5 — attached ML walk-forward evidence must match the model's own training
         configuration (algorithm, hyperparameters, feature names).
"""
import copy

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.db.base import Base
from backend.app.services.backtest.walk_forward import (
    _aggregate_profit_factor,
    _continuous_drawdown_from_folds,
)
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.market_data.mock_provider import MockMarketDataProvider
from backend.app.services.dataset.pipeline import DatasetPipeline
from backend.app.core.errors import ModelGovernanceError
from backend.app.core.config import settings


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_dataset(db_session):
    provider = MockMarketDataProvider()
    df = provider.get_historical_dataframe("EURUSD", TimeFrame.M15, count=600)
    pipeline = DatasetPipeline(db=db_session)
    return pipeline.build_dataset(df, "EURUSD", "M15", horizon_bars=12)


# ---------------------------------------------------------------- item 1 (PF)

def test_profit_factor_is_ratio_when_losses_exist():
    assert _aggregate_profit_factor(600.0, -400.0) == 1.5
    assert _aggregate_profit_factor(150.0, -120.0) == pytest.approx(1.25)


def test_profit_factor_no_loss_uses_sentinel_not_the_profit_amount():
    """Buglist item 1: with zero gross loss the old code stored the raw profit
    AMOUNT (e.g. 500.0) as profit_factor — an absolute money figure compared
    against the promotion gate's ratio threshold. It must be the documented
    99.0 sentinel, exactly like BacktestMetrics.calculate."""
    assert _aggregate_profit_factor(500.0, 0.0) == 99.0
    assert _aggregate_profit_factor(0.01, 0.0) == 99.0
    assert _aggregate_profit_factor(0.0, 0.0) == 0.0
    assert _aggregate_profit_factor(0.0, -50.0) == 0.0


# ---------------------------------------------------------------- item 3 (continuous MDD)

def _fold(i, net, peak_abs, trough_abs):
    return {
        "fold_index": i,
        "net_profit": net,
        "total_trades": 5,
        "gross_profit": max(0.0, net),
        "gross_loss": min(0.0, net),
        "max_drawdown_percent": 0.0,
        "equity_peak_abs": peak_abs,
        "equity_trough_abs": trough_abs,
    }


def test_continuous_drawdown_exceeds_max_of_fold_drawdowns():
    """Two consecutive losing folds: each fold's own MDD% is small, but the
    stitched account loses compounding depth the max-of-folds figure misses."""
    folds = [
        _fold(1, -400.0, 10020.0, 9700.0),   # fold-local dd ≈ 3.19%
        _fold(2, -400.0, 9660.0, 9350.0),    # fold-local dd ≈ 3.21%
    ]
    continuous = _continuous_drawdown_from_folds(folds, 10000.0)
    max_of_folds = 3.210726  # the largest fold-local figure, roughly

    # Fold 2's trough, offset by fold 1's loss, reaches 9350-400=8950 against
    # the global peak 10020 — far deeper than any fold-local figure.
    assert continuous["max_drawdown_percent"] > max_of_folds
    expected_pct = (10020.0 - (9350.0 - 400.0)) / 10020.0 * 100.0
    assert continuous["max_drawdown_percent"] == pytest.approx(expected_pct, abs=1e-6)
    assert continuous["max_drawdown_amount"] == pytest.approx(1070.0, abs=1e-6)


def test_continuous_drawdown_recovery_lowens_the_global_peak_requirement():
    """A losing fold followed by a winning one: the second fold's trough is
    measured against the global peak carried from BEFORE the loss, not against
    the fold's own (higher) balance."""
    folds = [
        _fold(1, -500.0, 10000.0, 9500.0),
        _fold(2, 800.0, 10300.0, 9500.0),  # recovers locally, but dips 500 below its own start
    ]
    continuous = _continuous_drawdown_from_folds(folds, 10000.0)
    # Fold 2's trough, offset by fold 1's loss, reaches 9500-500=9000 against
    # the global peak 10000 → 10% (fold-local MDD would claim only 7.8%).
    assert continuous["max_drawdown_percent"] == pytest.approx(10.0, abs=1e-9)


def test_continuous_drawdown_rejects_folds_without_extremes():
    """Pre-fix payloads (fold records without absolute equity extremes) fail
    closed instead of silently falling back to the underestimating aggregate."""
    legacy_fold = {
        "fold_index": 1, "net_profit": 10.0, "total_trades": 5,
        "gross_profit": 10.0, "gross_loss": 0.0, "max_drawdown_percent": 1.0,
    }
    with pytest.raises(ValueError, match="equity_peak_abs"):
        _continuous_drawdown_from_folds([legacy_fold], 10000.0)


# ---------------------------------------------------------------- item 4 (producer)

def test_training_flow_produces_and_attaches_ml_walk_forward_evidence(
    db_session, sample_dataset, monkeypatch, tmp_path
):
    """Buglist item 4: the shared producer must run, persist, and attach the
    evidence chain on the freshly registered candidate — the promotion gate's
    binding requirement becomes reachable through the normal training flow."""
    from backend.app.services.ml.trainer import ModelTrainer
    from backend.app.services.ml.registry import ModelRegistry
    from backend.app.services.ml.training import produce_ml_walk_forward_evidence
    from backend.app.models.ml_model import MLModelRecord
    from backend.app.models.ml_walk_forward_run import MLWalkForwardRun
    from backend.app.core.metrics_integrity import verify_ml_walk_forward_digest

    monkeypatch.setattr(settings, "ARTIFACTS_MODELS_DIR", tmp_path)
    monkeypatch.setattr(settings, "ML_WALK_FORWARD_FOLDS", 3)

    train_res = ModelTrainer(db=db_session).train_candidate(
        sample_dataset, algorithm="logistic_regression")

    df_raw = _random_walk_frame()

    outcome = produce_ml_walk_forward_evidence(
        db_session,
        train_res,
        df_raw,
        symbol="EURUSD",
        timeframe="M15",
        algorithm="logistic_regression",
        horizon_bars=12,
        spread_points=17,
        point_value=0.0001,
    )
    assert outcome["status"] == "evidence_recorded", outcome
    assert outcome["run_id"]

    record = db_session.query(MLModelRecord).filter_by(
        model_id=train_res["model_id"]).first()
    stored = record.ml_walk_forward_metrics
    assert stored and stored.get("run_id") == outcome["run_id"]
    assert "digest" in stored
    assert verify_ml_walk_forward_digest(record.model_id, {
        k: v for k, v in stored.items() if k != "digest"}, stored["digest"])

    row = db_session.query(MLWalkForwardRun).filter_by(run_id=outcome["run_id"]).first()
    assert row is not None and row.model_id == record.model_id

    # The full promotion-gate chain must accept the produced evidence.
    assert ModelRegistry._ml_walk_forward_failure(record.model_id, stored) is None


def test_producer_reports_skip_instead_of_failing_training(
    db_session, sample_dataset, monkeypatch, tmp_path
):
    """A dataset too short for the fold geometry must be reported honestly as
    a skip — evidence production can never break candidate training."""
    from backend.app.services.ml.trainer import ModelTrainer
    from backend.app.services.ml.training import produce_ml_walk_forward_evidence
    import numpy as np

    monkeypatch.setattr(settings, "ARTIFACTS_MODELS_DIR", tmp_path)
    monkeypatch.setattr(settings, "ML_WALK_FORWARD_FOLDS", 3)

    train_res = ModelTrainer(db=db_session).train_candidate(
        sample_dataset, algorithm="logistic_regression")

    tiny = pd.DataFrame({
        "open": np.full(60, 100.0), "high": np.full(60, 101.0),
        "low": np.full(60, 99.0), "close": np.full(60, 100.0),
        "volume": 100.0, "spread": 17,
    }, index=pd.date_range("2026-01-01", periods=60, freq="15min", tz="UTC"))

    outcome = produce_ml_walk_forward_evidence(
        db_session, train_res, tiny, symbol="EURUSD", timeframe="M15",
        algorithm="logistic_regression", horizon_bars=12,
        spread_points=17, point_value=0.0001,
    )
    assert outcome["status"] == "skipped"
    assert outcome["reason"]


def test_producer_disabled_below_two_folds(db_session, sample_dataset, monkeypatch, tmp_path):
    from backend.app.services.ml.trainer import ModelTrainer
    from backend.app.services.ml.training import produce_ml_walk_forward_evidence

    monkeypatch.setattr(settings, "ARTIFACTS_MODELS_DIR", tmp_path)
    monkeypatch.setattr(settings, "ML_WALK_FORWARD_FOLDS", 0)
    train_res = ModelTrainer(db=db_session).train_candidate(
        sample_dataset, algorithm="logistic_regression")
    outcome = produce_ml_walk_forward_evidence(
        db_session, train_res, None, symbol="EURUSD", timeframe="M15",
        algorithm="logistic_regression", horizon_bars=12,
        spread_points=17, point_value=0.0001,
    )
    assert outcome["status"] == "disabled"


# ---------------------------------------------------------------- item 5 (config binding)

def _random_walk_frame(n: int = 1200) -> pd.DataFrame:
    """Deterministic random walk (same construction as the binding tests):
    single-class training partitions never occur on this data."""
    import numpy as np

    rng = np.random.default_rng(11)
    steps = rng.normal(loc=0.01, scale=1.0, size=n)
    close = 2000.0 + np.cumsum(steps)
    open_ = np.roll(close, 1)
    open_[0] = 2000.0
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.35, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.35, size=n))
    idx = pd.date_range("2026-09-01 00:00", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": 100.0, "spread": 20}, index=idx)


def _persisted_mlwf_payload(monkeypatch, tmp_path):
    from backend.app.services.ml.walk_forward import run_ml_walk_forward

    monkeypatch.setattr(settings, "ARTIFACTS_MODELS_DIR", tmp_path)
    return run_ml_walk_forward(
        _random_walk_frame(), symbol="EURUSD", timeframe="M15",
        algorithm="logistic_regression", fold_count=3, min_train_bars=300,
        horizon_bars=12,
    )


def test_attach_rejects_evidence_from_a_different_algorithm(db_session, monkeypatch, tmp_path):
    from backend.app.services.ml.registry import ModelRegistry

    payload = _persisted_mlwf_payload(monkeypatch, tmp_path)
    candidate = _candidate_with_payload_features(db_session, payload, algorithm="random_forest")

    registry = ModelRegistry(db_session)
    run_id = registry.persist_ml_walk_forward_evaluation(copy.deepcopy(payload))
    with pytest.raises(ModelGovernanceError, match="algorithm"):
        registry.attach_ml_walk_forward_metrics(
            candidate.model_id, payload, run_id=run_id)


def test_attach_rejects_evidence_from_a_different_feature_set(db_session, monkeypatch, tmp_path):
    from backend.app.services.ml.registry import ModelRegistry
    from backend.app.services.ml.trainer import ModelTrainer

    payload = _persisted_mlwf_payload(monkeypatch, tmp_path)
    provider = MockMarketDataProvider()
    df = provider.get_historical_dataframe("EURUSD", TimeFrame.M15, count=600)
    ds = DatasetPipeline(db=db_session).build_dataset(df, "EURUSD", "M15", horizon_bars=12)
    train_res = ModelTrainer(db=db_session).train_candidate(
        ds, algorithm="logistic_regression")
    model_id = train_res["model_id"]

    # The evidence carries a feature list that does NOT match the model's own
    # (one feature removed) — same algorithm, same hyperparameters, wrong data.
    forged = copy.deepcopy(payload)
    forged["feature_names"] = forged["feature_names"][:-1]

    registry = ModelRegistry(db_session)
    run_id = registry.persist_ml_walk_forward_evaluation(payload)
    with pytest.raises(ModelGovernanceError, match="feature_names"):
        registry.attach_ml_walk_forward_metrics(model_id, forged, run_id=run_id)


def sample_dataset_for(db_session):
    provider = MockMarketDataProvider()
    df = provider.get_historical_dataframe("EURUSD", TimeFrame.M15, count=600)
    return DatasetPipeline(db=db_session).build_dataset(df, "EURUSD", "M15", horizon_bars=12)


def _candidate_with_payload_features(db_session, payload, algorithm="logistic_regression"):
    """Registers a candidate whose training config matches the payload's
    features/algorithm (so ONLY the tampered dimension can be blamed)."""
    from backend.app.services.ml.trainer import ModelTrainer
    from backend.app.models.ml_model import MLModelRecord

    train_res = ModelTrainer(db=db_session).train_candidate(
        sample_dataset_for(db_session), algorithm=algorithm)
    record = db_session.query(MLModelRecord).filter_by(
        model_id=train_res["model_id"]).first()
    record.features_used = list(payload["feature_names"])
    db_session.commit()
    return record
