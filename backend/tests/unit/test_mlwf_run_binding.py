"""M8 phase 2.4: ML walk-forward evidence must be bound to a PERSISTED
evaluation run (ml_walk_forward_runs).

Until now the per-fold-retrained evidence existed only inside the model
record and was re-signed at attach time — self-attesting. These tests pin the
closed chain: the evaluator output is persisted (with its digest) at
generation time; a model attaches ONLY evidence matching a persisted row; a
row is bound to exactly one model; and promotion refuses attached evidence
whose run row is missing or bound elsewhere.
"""
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from backend.app.core.config import settings
from backend.app.core.errors import ModelGovernanceError
from backend.app.core.metrics_integrity import metric_digest, ml_walk_forward_digest
from backend.app.db.base import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.models.ml_model import MLModelRecord
from backend.app.models.ml_walk_forward_run import MLWalkForwardRun
from backend.app.services.ml.registry import ModelRegistry
from backend.app.services.ml.walk_forward import (
    MLWF_EVIDENCE_TYPE,
    run_ml_walk_forward,
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


def _mlwf_frame() -> pd.DataFrame:
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


def _fresh_payload():
    import hashlib

    payload = run_ml_walk_forward(
        _mlwf_frame(), symbol="XAUUSD", timeframe="M15",
        algorithm="logistic_regression", fold_count=3, min_train_bars=300,
    )
    return payload, hashlib


def _model(model_id, artifact_path, *, auc=0.56, brier=0.19, wf=None):
    import hashlib
    import joblib

    validation_metrics = {"roc_auc": auc, "brier_score": brier, "sample_count": 50}
    test_metrics = {"roc_auc": auc, "brier_score": brier, "sample_count": 50}
    wf = wf if wf is not None else {}
    digest = metric_digest(model_id, validation_metrics, test_metrics, wf)
    estimator = DummyClassifier(strategy="prior").fit([[0.0], [1.0]], [0, 1])
    joblib.dump({
        "model": estimator, "scaler": None, "feature_names": ["rsi_14"],
        "model_id": model_id, "validation_metrics": validation_metrics,
        "test_metrics": test_metrics, "metrics_digest": digest,
    }, artifact_path)
    return MLModelRecord(
        model_id=model_id, dataset_version_tag="mlwfr-dv", algorithm="logistic_regression",
        hyperparameters={}, features_used=["rsi_14"], target_name="target_tp_before_sl",
        train_metrics={}, validation_metrics=validation_metrics, test_metrics=test_metrics,
        calibration_metrics={}, walk_forward_metrics=wf, status="candidate",
        artifact_path=str(artifact_path), metrics_digest=digest,
        artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    )


# ---------------------------------------------------------------- persist

def test_persist_verifies_structurally_and_signs(db_session):
    payload, _ = _fresh_payload()
    run_id = ModelRegistry(db_session).persist_ml_walk_forward_evaluation(payload)

    row = db_session.query(MLWalkForwardRun).filter_by(run_id=run_id).first()
    assert row is not None
    assert row.payload["evidence_type"] == MLWF_EVIDENCE_TYPE
    assert "digest" not in row.payload, "the row payload stays digest-stripped"
    assert row.model_id is None, "a fresh run is unbound"
    from backend.app.core.metrics_integrity import verify_ml_walk_forward_digest

    assert verify_ml_walk_forward_digest(
        ModelRegistry.MLWF_RUN_DIGEST_CONTEXT, row.payload, row.digest)


def test_persist_rejects_structurally_invalid_payload(db_session):
    payload, _ = _fresh_payload()
    payload["aggregate"]["roc_auc"]["mean"] = "abc"
    with pytest.raises(ModelGovernanceError, match="verification failed"):
        ModelRegistry(db_session).persist_ml_walk_forward_evaluation(payload)


# ---------------------------------------------------------------- attach negatives

def test_attach_requires_a_run_id(db_session, tmp_path):
    payload, _ = _fresh_payload()
    db_session.add(_model("m1", tmp_path / "m1.joblib"))
    db_session.commit()
    with pytest.raises(ModelGovernanceError, match="requires the run_id"):
        ModelRegistry(db_session).attach_ml_walk_forward_metrics("m1", payload, run_id="")


def test_attach_rejects_a_run_that_was_never_persisted(db_session, tmp_path):
    payload, _ = _fresh_payload()
    db_session.add(_model("m1", tmp_path / "m1.joblib"))
    db_session.commit()
    with pytest.raises(ModelGovernanceError, match="does not exist"):
        ModelRegistry(db_session).attach_ml_walk_forward_metrics(
            "m1", payload, run_id="mlwfr_nonexistent")


def test_attach_rejects_payload_that_differs_from_the_persisted_run(db_session, tmp_path):
    payload, _ = _fresh_payload()
    registry = ModelRegistry(db_session)
    run_id = registry.persist_ml_walk_forward_evaluation(copy.deepcopy(payload))

    db_session.add(_model("m1", tmp_path / "m1.joblib"))
    db_session.commit()
    # Structurally VALID but different from what was persisted (an extra
    # claim the verifier ignores) — the row equality gate must refuse it.
    forged = copy.deepcopy(payload)
    forged["notes"] = "operator-added commentary"
    with pytest.raises(ModelGovernanceError, match="does not match persisted run"):
        registry.attach_ml_walk_forward_metrics("m1", forged, run_id=run_id)


# ---------------------------------------------------------------- promotion gate

def _seed_strategy_wf(db_session):
    from backend.tests.unit._wf_evidence import seed_walk_forward_run

    return seed_walk_forward_run(db_session, "bt_mlwfr_gate")


def _strategy_wf_payload():
    """Strategy-level WF evidence so the EARLIER strategy gate passes and the
    test actually exercises the ML binding gate."""
    from backend.tests.unit._wf_evidence import build_walk_forward_payload

    return build_walk_forward_payload("bt_mlwfr_gate")


def test_promotion_requires_persisted_run_binding(db_session, tmp_path):
    """Hand-signed evidence WITHOUT a persisted run must no longer promote —
    the self-attestation residual is closed."""
    import joblib

    _seed_strategy_wf(db_session)
    payload, hashlib = _fresh_payload()
    payload.pop("digest", None)
    payload.pop("run_id", None)
    artifact = tmp_path / "bound.joblib"
    candidate = _model("m-bound", artifact, wf=_strategy_wf_payload())
    db_session.add(candidate)
    db_session.commit()

    registry = ModelRegistry(db_session)
    payload["digest"] = ml_walk_forward_digest("m-bound", payload)
    record = db_session.query(MLModelRecord).filter_by(model_id="m-bound").first()
    record.ml_walk_forward_metrics = payload
    record.metrics_digest = metric_digest(
        record.model_id, record.validation_metrics, record.test_metrics,
        record.walk_forward_metrics,
    )
    # keep the artifact consistent so earlier gates pass
    art = joblib.load(artifact)
    art["ml_walk_forward_metrics"] = payload
    art["metrics_digest"] = record.metrics_digest
    joblib.dump(art, artifact)
    record.artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    db_session.commit()

    with pytest.raises(ModelGovernanceError, match="not bound to a persisted evaluation run"):
        registry.promote_candidate("m-bound", actor="test")


def test_promotion_succeeds_for_the_full_persisted_chain(db_session, tmp_path):
    _seed_strategy_wf(db_session)
    payload, _ = _fresh_payload()
    db_session.add(_model("m-good", tmp_path / "good.joblib", wf=_strategy_wf_payload()))
    db_session.commit()

    registry = ModelRegistry(db_session)
    run_id = registry.persist_ml_walk_forward_evaluation(copy.deepcopy(payload))
    registry.attach_ml_walk_forward_metrics("m-good", payload, run_id=run_id)

    promoted = registry.promote_candidate("m-good", actor="test")
    assert promoted.status == "active"
    row = db_session.query(MLWalkForwardRun).filter_by(run_id=run_id).first()
    assert row.model_id == "m-good"


def test_promotion_fails_when_the_bound_run_row_is_deleted(db_session, tmp_path):
    _seed_strategy_wf(db_session)
    payload, _ = _fresh_payload()
    db_session.add(_model("m-del", tmp_path / "del.joblib", wf=_strategy_wf_payload()))
    db_session.commit()

    registry = ModelRegistry(db_session)
    run_id = registry.persist_ml_walk_forward_evaluation(copy.deepcopy(payload))
    registry.attach_ml_walk_forward_metrics("m-del", payload, run_id=run_id)

    db_session.query(MLWalkForwardRun).filter_by(run_id=run_id).delete()
    db_session.commit()
    with pytest.raises(ModelGovernanceError, match="not bound to a persisted evaluation run"):
        registry.promote_candidate("m-del", actor="test")


def test_promotion_check_reports_the_binding_gate(db_session, tmp_path):
    """Step 3.5 pre-check summary must expose the binding failure per gate —
    the operator sees WHY before attempting promotion."""
    _seed_strategy_wf(db_session)
    payload, hashlib = _fresh_payload()
    import joblib

    artifact = tmp_path / "chk.joblib"
    candidate = _model("m-chk", artifact, wf=_strategy_wf_payload())
    db_session.add(candidate)
    db_session.commit()
    payload.pop("digest", None)
    payload.pop("run_id", None)
    payload["digest"] = ml_walk_forward_digest("m-chk", payload)
    record = db_session.query(MLModelRecord).filter_by(model_id="m-chk").first()
    record.ml_walk_forward_metrics = payload
    record.metrics_digest = metric_digest(
        record.model_id, record.validation_metrics, record.test_metrics,
        record.walk_forward_metrics,
    )
    art = joblib.load(artifact)
    art["ml_walk_forward_metrics"] = payload
    art["metrics_digest"] = record.metrics_digest
    joblib.dump(art, artifact)
    record.artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    db_session.commit()

    summary = ModelRegistry(db_session).promotion_check("m-chk")
    gate = next(g for g in summary["gates"] if g["gate"] == "ml_walk_forward_evidence")
    assert gate["passed"] is False
    assert "not bound to a persisted evaluation run" in gate["detail"]
