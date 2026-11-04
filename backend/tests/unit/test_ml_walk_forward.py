"""M8: true per-fold ML walk-forward validation (regression tests).

The strategy walk-forward proves out-of-sample robustness of the approved
RULES with real costs but never retrains the ML model. These tests pin the
model-quality evidence chain: run_ml_walk_forward must retrain a fresh model
and a fresh baseline per fold on purged/embargoed expanding training windows,
evaluate only on chronologically-later OOS windows, and produce a payload the
registry can independently re-verify (fold geometry, purge/embargo gaps,
recomputed aggregates, HMAC/SHA-256 digest) at attach AND at promotion time.

Synthetic OHLC frames are fixtures: they prove the evidence chain, not
profitability on real market data. No orders are placed.
"""
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from backend.app.core.config import settings
from backend.app.core.errors import ModelGovernanceError
from backend.app.core.metrics_integrity import (
    metric_digest,
    ml_walk_forward_digest,
)
from backend.app.db.base import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.models.ml_model import MLModelRecord
from backend.app.services.ml.registry import ModelRegistry
from backend.app.services.ml.walk_forward import (
    MLWF_EVIDENCE_TYPE,
    MLWF_THRESHOLD,
    build_ml_walk_forward_folds,
    run_ml_walk_forward,
    verify_ml_walk_forward_evidence,
    verify_ml_walk_forward_payload,
)


@pytest.fixture
def db_session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ARTIFACTS_MODELS_DIR", tmp_path)
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture
def synthetic_frame():
    """Deterministic 1200-bar OHLC frame (random walk + mild drift, seeded).

    Long enough for 3 OOS folds after the warmup and the label horizon drop
    the cleaning removes, with min_train_bars=300 per fold."""
    rng = np.random.default_rng(7)
    n = 1200
    steps = rng.normal(loc=0.01, scale=1.0, size=n)
    close = 2000.0 + np.cumsum(steps)
    open_ = np.roll(close, 1)
    open_[0] = 2000.0
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.35, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.35, size=n))
    idx = pd.date_range("2026-09-01 00:00", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": 100.0, "spread": 20}, index=idx)


# ---------------------------------------------------------------- fold geometry

def test_fold_geometry_is_chronological_purged_and_non_overlapping():
    folds = build_ml_walk_forward_folds(n_rows=1000, fold_count=3,
                                        purge_window=24, embargo_bars=12,
                                        min_train_bars=300)
    assert len(folds) == 3
    previous_test_end = None
    for fold in folds:
        # Training ends exactly purge+embargo bars before the test window.
        assert fold["train_end_pos"] == fold["test_start_pos"] - 36
        assert fold["train_end_pos"] >= 300 - 36, "expanding training window must stay usable"
        assert fold["test_start_pos"] < fold["test_end_pos"]
        if previous_test_end is not None:
            assert fold["test_start_pos"] >= previous_test_end, "OOS windows must not overlap"
        previous_test_end = fold["test_end_pos"]
    # The last fold absorbs the remainder up to the end of the frame.
    assert folds[-1]["test_end_pos"] == 1000
    # Windows are contiguous and chronological.
    starts = [f["test_start_pos"] for f in folds]
    assert starts == sorted(starts)


def test_fold_builder_rejects_impossible_geometry():
    with pytest.raises(ValueError, match="at least 2 folds"):
        build_ml_walk_forward_folds(1000, 1, 24, 12, 300)
    with pytest.raises(ValueError, match="no purge/embargo"):
        build_ml_walk_forward_folds(1000, 3, 0, 0, 300)
    with pytest.raises(ValueError, match="Not enough rows"):
        build_ml_walk_forward_folds(150, 3, 24, 12, 300)
    with pytest.raises(ValueError, match="collapsed"):
        # min_train_bars small enough to pass the tail check but smaller than
        # the purge+embargo gap, so fold 1 training collapses under the gap.
        build_ml_walk_forward_folds(100, 2, 24, 12, 20)


# ---------------------------------------------------------------- real run

def test_ml_walk_forward_retrains_per_fold_and_verifies(synthetic_frame):
    """The real run: fresh model + fresh baseline per fold, purged expanding
    training windows, OOS-only evaluation, and a payload that independently
    re-verifies (structure, geometry, aggregates)."""
    payload = run_ml_walk_forward(
        synthetic_frame,
        symbol="XAUUSD",
        timeframe="M15",
        algorithm="logistic_regression",
        fold_count=3,
        horizon_bars=24,
        min_train_bars=300,
        spread_points=20,
        code_version="test-head",
    )
    assert payload["evidence_type"] == MLWF_EVIDENCE_TYPE
    assert payload["fold_count"] == 3
    assert payload["purge_window"] == 24, "purge must match the 24-bar label horizon"
    assert payload["embargo_bars"] == 12
    assert payload["threshold"] == MLWF_THRESHOLD
    assert "no tuning on validation or OOS" in payload["threshold_policy"]

    # Every fold retrained on a strictly purged window.
    for fold in payload["folds"]:
        assert fold["train_end_pos"] + 24 + 12 <= fold["test_start_pos"]
        assert fold["train_samples"] >= 100
        assert fold["oos_samples"] > 0
        assert set(fold["metrics"]) >= {"roc_auc", "brier_score", "accuracy", "f1"}
        assert "roc_auc" in fold["baseline_metrics"]

    assert payload["aggregate"]["roc_auc"]["n"] == 3
    assert payload["aggregate"]["total_oos_samples"] == sum(
        f["oos_samples"] for f in payload["folds"])
    # Independent re-verification passes on the honest payload.
    assert verify_ml_walk_forward_payload(payload) is None
    # Digest layer verifies too (attestation round-trip).
    digest = ml_walk_forward_digest("model_x", payload)
    assert verify_ml_walk_forward_evidence("model_x", payload, digest) is None
    assert verify_ml_walk_forward_evidence("model_x", payload, "0" * 64) is not None


def test_ml_walk_forward_aggregates_recompute_from_folds(synthetic_frame):
    payload = run_ml_walk_forward(
        synthetic_frame, symbol="XAUUSD", timeframe="M15",
        algorithm="logistic_regression", fold_count=3, min_train_bars=300,
    )
    aucs = [f["metrics"]["roc_auc"] for f in payload["folds"]]
    assert payload["aggregate"]["roc_auc"]["mean"] == pytest.approx(
        float(np.mean(aucs)), abs=0.0011)
    beating = sum(
        1 for f in payload["folds"]
        if f["metrics"]["roc_auc"] > f["baseline_metrics"]["roc_auc"]
    )
    assert payload["aggregate"]["folds_beating_baseline_roc_auc"] == beating


def test_ml_walk_forward_rejects_bad_inputs(synthetic_frame):
    with pytest.raises(ValueError, match="DatetimeIndex"):
        run_ml_walk_forward(
            pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1]}),
            symbol="XAUUSD", timeframe="M15", fold_count=3, min_train_bars=50)
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        run_ml_walk_forward(
            synthetic_frame, symbol="XAUUSD", timeframe="M15",
            algorithm="xgboost", fold_count=3, min_train_bars=300)
    with pytest.raises(ValueError, match="Not enough rows|usable rows"):
        run_ml_walk_forward(
            synthetic_frame.iloc[:200], symbol="XAUUSD", timeframe="M15",
            fold_count=3, min_train_bars=300)


# ------------------------------------------------- independent verification (negative)

def _valid_payload(synthetic_frame) -> dict:
    return run_ml_walk_forward(
        synthetic_frame, symbol="XAUUSD", timeframe="M15",
        algorithm="logistic_regression", fold_count=3, min_train_bars=300,
    )


def test_verifier_rejects_forged_aggregate(synthetic_frame):
    payload = _valid_payload(synthetic_frame)
    forged = copy.deepcopy(payload)
    forged["aggregate"]["roc_auc"]["mean"] = 0.99
    failure = verify_ml_walk_forward_payload(forged)
    assert failure is not None and "roc_auc" in failure


def test_verifier_rejects_fold_overlap(synthetic_frame):
    payload = _valid_payload(synthetic_frame)
    forged = copy.deepcopy(payload)
    # Slide fold 2's OOS window back into fold 1's window while keeping its
    # own purge gap intact, so the OVERLAP check (not the purge check) fires.
    forged["folds"][1]["test_start_pos"] = forged["folds"][0]["test_end_pos"] - 5
    forged["folds"][1]["train_end_pos"] = forged["folds"][1]["test_start_pos"] - 36
    failure = verify_ml_walk_forward_payload(forged)
    assert failure is not None and "overlap" in failure


def test_verifier_rejects_purge_violation(synthetic_frame):
    payload = _valid_payload(synthetic_frame)
    forged = copy.deepcopy(payload)
    # Training ends inside the purge/embargo buffer of the test window.
    forged["folds"][0]["train_end_pos"] = forged["folds"][0]["test_start_pos"]
    failure = verify_ml_walk_forward_payload(forged)
    assert failure is not None and "purge/embargo" in failure


def test_verifier_rejects_missing_provenance_and_threshold_tuning(synthetic_frame):
    payload = _valid_payload(synthetic_frame)
    no_version = copy.deepcopy(payload)
    del no_version["code_version"]
    assert "code_version" in verify_ml_walk_forward_payload(no_version)

    tuned = copy.deepcopy(payload)
    tuned["threshold"] = 0.7  # tuned on OOS — not acceptable evidence
    failure = verify_ml_walk_forward_payload(tuned)
    assert failure is not None and "threshold" in failure

    zero_gap = copy.deepcopy(payload)
    zero_gap["purge_window"] = 0
    zero_gap["embargo_bars"] = 0
    failure = verify_ml_walk_forward_payload(zero_gap)
    assert failure is not None and "purge" in failure.lower()


def test_verifier_rejects_purge_smaller_than_label_horizon(synthetic_frame):
    """HIGH-1 regression (2026-10-31 ML audit): a payload whose purge window
    does not cover the label horizon lets training labels read outcomes inside
    the OOS window (label leakage). The verifier must reject it regardless of
    the nominal gap size."""
    payload = _valid_payload(synthetic_frame)
    forged = copy.deepcopy(payload)
    forged["purge_window"] = 0
    forged["embargo_bars"] = 1  # nominal gap >= 1, but purge < horizon_bars=24
    failure = verify_ml_walk_forward_payload(forged)
    assert failure is not None and "horizon" in failure, failure


def test_run_rejects_purge_smaller_than_horizon(synthetic_frame):
    """The runner itself must refuse a leaky configuration up front."""
    with pytest.raises(ValueError, match="purge_window.*horizon_bars"):
        run_ml_walk_forward(
            synthetic_frame, symbol="XAUUSD", timeframe="M15",
            algorithm="logistic_regression", fold_count=3, min_train_bars=300,
            horizon_bars=24, purge_window=0, embargo_bars=1)


def test_verifier_rejects_broken_fold_index_sequence(synthetic_frame):
    payload = _valid_payload(synthetic_frame)
    forged = copy.deepcopy(payload)
    forged["folds"][1]["fold_index"] = forged["folds"][0]["fold_index"]
    failure = verify_ml_walk_forward_payload(forged)
    assert failure is not None and "fold_index" in failure, failure


def test_verifier_rejects_forged_total_oos_samples(synthetic_frame):
    payload = _valid_payload(synthetic_frame)
    forged = copy.deepcopy(payload)
    forged["aggregate"]["total_oos_samples"] = 999999
    failure = verify_ml_walk_forward_payload(forged)
    assert failure is not None and "total_oos_samples" in failure, failure


# ---------------------------------------------------------------- registry integration

def _model(model_id, artifact_path, *, status="candidate", auc=0.56, brier=0.19,
           walk_forward_metrics=None):
    validation_metrics = {"roc_auc": auc, "brier_score": brier, "sample_count": 50}
    test_metrics = {"roc_auc": auc, "brier_score": brier, "sample_count": 50}
    wf = walk_forward_metrics if walk_forward_metrics is not None else {}
    digest = metric_digest(model_id, validation_metrics, test_metrics, wf)
    estimator = DummyClassifier(strategy="prior").fit([[0.0], [1.0]], [0, 1])
    import hashlib
    import joblib

    joblib.dump({
        "model": estimator, "scaler": None, "feature_names": ["rsi_14"],
        "model_id": model_id, "validation_metrics": validation_metrics,
        "test_metrics": test_metrics, "metrics_digest": digest,
    }, artifact_path)
    # Minor-17 contract: every artifact row carries the byte digest the load
    # path verifies BEFORE unpickling.
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return MLModelRecord(
        model_id=model_id, dataset_version_tag="mlwf-dv", algorithm="logistic_regression",
        hyperparameters={}, features_used=["rsi_14"], target_name="target_tp_before_sl",
        train_metrics={}, validation_metrics=validation_metrics, test_metrics=test_metrics,
        calibration_metrics={}, walk_forward_metrics=wf, status=status,
        artifact_path=str(artifact_path), metrics_digest=digest,
        artifact_sha256=artifact_sha256,
    )


def _persist_payload(db_session, payload):
    """Persist a valid evaluator payload and return its run_id (phase 2.4)."""
    return ModelRegistry(db_session).persist_ml_walk_forward_evaluation(payload)


def test_attach_ml_walk_forward_verifies_and_stores(db_session, tmp_path, synthetic_frame):
    payload = _valid_payload(synthetic_frame)
    artifact = tmp_path / "mlwf.joblib"
    candidate = _model("mlwf-candidate", artifact)
    db_session.add(candidate)
    db_session.commit()

    registry = ModelRegistry(db_session)
    run_id = _persist_payload(db_session, payload)
    record = registry.attach_ml_walk_forward_metrics("mlwf-candidate", payload, run_id=run_id)

    stored = record.ml_walk_forward_metrics
    assert stored["evidence_type"] == MLWF_EVIDENCE_TYPE
    assert stored["run_id"] == run_id, "attach must carry the persisted-run binding"
    assert "digest" in stored, "attach must commit a tamper-evident digest"
    assert ModelRegistry._ml_walk_forward_failure(record.model_id, stored) is None

    from backend.app.models.ml_walk_forward_run import MLWalkForwardRun

    row = db_session.query(MLWalkForwardRun).filter_by(run_id=run_id).first()
    assert row is not None and row.model_id == "mlwf-candidate", "run row must record the binding"

    # A tampered aggregate must be caught by the promotion-time re-verification.
    tampered = copy.deepcopy(stored)
    tampered["aggregate"]["roc_auc"]["mean"] = 0.99
    failure = ModelRegistry._ml_walk_forward_failure(record.model_id, tampered)
    assert failure is not None and "roc_auc" in failure

    # Stripping the digest must fail closed too.
    stripped = copy.deepcopy(stored)
    stripped.pop("digest")
    failure = ModelRegistry._ml_walk_forward_failure(record.model_id, stripped)
    assert failure is not None and "digest" in failure


def test_attach_rejects_invalid_ml_walk_forward_payload(db_session, tmp_path, synthetic_frame):
    payload = _valid_payload(synthetic_frame)
    run_id = _persist_payload(db_session, _valid_payload(synthetic_frame))
    payload["folds"][1]["test_start_pos"] = payload["folds"][0]["test_end_pos"] - 5
    payload["folds"][1]["train_end_pos"] = payload["folds"][1]["test_start_pos"] - 36
    artifact = tmp_path / "mlwf_bad.joblib"
    candidate = _model("mlwf-bad", artifact)
    db_session.add(candidate)
    db_session.commit()

    with pytest.raises(ModelGovernanceError, match="overlap"):
        ModelRegistry(db_session).attach_ml_walk_forward_metrics(
            "mlwf-bad", payload, run_id=run_id)

    db_session.refresh(candidate)
    assert candidate.ml_walk_forward_metrics == {} or not candidate.ml_walk_forward_metrics


def test_promotion_blocks_tampered_ml_walk_forward_evidence(db_session, tmp_path, monkeypatch):
    """A candidate that passes every other gate must STILL be refused when its
    attached ML walk-forward payload was tampered with after attach."""
    from backend.tests.unit._wf_evidence import build_walk_forward_payload, seed_walk_forward_run

    monkeypatch.setattr(
        "backend.app.services.ml.registry.log_security_event", lambda *a, **k: None)
    seed_walk_forward_run(db_session, "bt_mlwf_promo")

    artifact = tmp_path / "mlwf_tampered.joblib"
    candidate = _model(
        "mlwf-tampered", artifact, auc=0.56, brier=0.19,
        walk_forward_metrics=build_walk_forward_payload("bt_mlwf_promo"),
    )
    db_session.add(candidate)
    db_session.commit()

    registry = ModelRegistry(db_session)
    payload = run_ml_walk_forward(
        _mlwf_frame(), symbol="XAUUSD", timeframe="M15",
        algorithm="logistic_regression", fold_count=3, min_train_bars=300,
    )
    run_id = registry.persist_ml_walk_forward_evaluation(payload)
    registry.attach_ml_walk_forward_metrics("mlwf-tampered", payload, run_id=run_id)
    db_session.expire_all()

    # Tamper directly in the store after attach: forge the aggregate mean.
    record = db_session.query(MLModelRecord).filter_by(model_id="mlwf-tampered").first()
    stored = dict(record.ml_walk_forward_metrics)
    stored["aggregate"] = dict(stored["aggregate"])
    stored["aggregate"]["roc_auc"] = dict(stored["aggregate"]["roc_auc"])
    stored["aggregate"]["roc_auc"]["mean"] = 0.99
    record.ml_walk_forward_metrics = stored
    db_session.commit()

    with pytest.raises(ModelGovernanceError, match="ml_walk_forward evidence failed"):
        registry.promote_candidate("mlwf-tampered", actor="test")


def _mlwf_frame() -> pd.DataFrame:
    """Standalone frame for the promotion test (mirrors synthetic_frame)."""
    rng = np.random.default_rng(11)
    n = 1200
    steps = rng.normal(loc=0.01, scale=1.0, size=n)
    close = 2000.0 + np.cumsum(steps)
    open_ = np.roll(close, 1)
    open_[0] = 2000.0
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.35, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.35, size=n))
    idx = pd.date_range("2026-09-01 00:00", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": 100.0, "spread": 20}, index=idx)


def test_persisted_run_cannot_be_rebound_to_a_different_model(db_session, tmp_path, synthetic_frame):
    """Binding immutability: a persisted evaluation bound to model A must not
    validate for model B — the row records its owner, so a payload copied
    verbatim (including A's digest) fails closed on B."""
    payload = _valid_payload(synthetic_frame)
    candidate_a = _model("mlwf-model-a", tmp_path / "a.joblib")
    candidate_b = _model("mlwf-model-b", tmp_path / "b.joblib")
    db_session.add_all([candidate_a, candidate_b])
    db_session.commit()

    registry = ModelRegistry(db_session)
    run_id = registry.persist_ml_walk_forward_evaluation(copy.deepcopy(payload))
    attached = registry.attach_ml_walk_forward_metrics("mlwf-model-a", payload, run_id=run_id)
    stored = dict(attached.ml_walk_forward_metrics)  # includes A's digest + run binding

    with pytest.raises(ModelGovernanceError, match="already bound"):
        registry.attach_ml_walk_forward_metrics("mlwf-model-b", payload, run_id=run_id)

    # The promotion-time verifier must reject the replay as well.
    failure = ModelRegistry._ml_walk_forward_failure("mlwf-model-b", stored)
    assert failure is not None, "replayed evidence must fail for another model"


def test_reattach_of_own_payload_stays_idempotent_and_promotable(db_session, tmp_path, synthetic_frame):
    """Agent-C finding (MEDIUM): re-attaching a model's OWN stored payload
    used to re-sign the digest over the payload WITH the stale digest key,
    producing an evidence blob the promotion verifier would later reject as
    'tampered'. Idempotent re-attach must keep the evidence promotable."""
    payload = _valid_payload(synthetic_frame)
    candidate = _model("mlwf-own", tmp_path / "own.joblib")
    db_session.add(candidate)
    db_session.commit()

    registry = ModelRegistry(db_session)
    run_id = registry.persist_ml_walk_forward_evaluation(payload)
    first = registry.attach_ml_walk_forward_metrics("mlwf-own", payload, run_id=run_id)
    stored_first = dict(first.ml_walk_forward_metrics)

    # Re-attach the model's OWN payload verbatim (idempotent path).
    second = registry.attach_ml_walk_forward_metrics("mlwf-own", dict(stored_first), run_id=run_id)
    failure = ModelRegistry._ml_walk_forward_failure("mlwf-own", second.ml_walk_forward_metrics)
    assert failure is None, (
        f"idempotent re-attach must stay promotable, verifier said: {failure}")

    # A non-string digest must produce a clean governance error, not a raw
    # TypeError from hmac.compare_digest.
    bad = dict(second.ml_walk_forward_metrics)
    bad["digest"] = 123
    with pytest.raises(ModelGovernanceError, match="hex string"):
        registry.attach_ml_walk_forward_metrics("mlwf-own", bad, run_id=run_id)


def test_verifier_rejects_non_numeric_aggregate(synthetic_frame):
    payload = _valid_payload(synthetic_frame)
    forged = copy.deepcopy(payload)
    forged["aggregate"]["roc_auc"]["mean"] = "abc"
    failure = verify_ml_walk_forward_payload(forged)
    assert failure is not None and "numeric" in failure, failure
