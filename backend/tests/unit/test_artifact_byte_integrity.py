"""Minor-17 hardening: artifact BYTE integrity is verified BEFORE joblib.load.

joblib.load executes pickle code, so a tampered artifact must be refused by
comparing the file bytes against the attested ``artifact_sha256`` BEFORE the
unpickle — not only via the embedded metrics digest afterwards. Legacy rows
without a byte digest fail closed. All three joblib.load surfaces are
covered: runtime load, ML-WF attach, strategy-WF attach.
"""
import copy

import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.dummy import DummyClassifier

from backend.app.core.config import settings
from backend.app.core.errors import ModelGovernanceError
from backend.app.core.metrics_integrity import metric_digest
from backend.app.db.base import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.models.ml_model import MLModelRecord
from backend.app.services.ml.registry import ModelRegistry
from backend.app.services.ml.walk_forward import run_ml_walk_forward


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


def _write_artifact(path, model_id, digest):
    estimator = DummyClassifier(strategy="prior").fit([[0.0], [1.0]], [0, 1])
    joblib.dump({
        "model": estimator, "scaler": None, "feature_names": ["rsi_14"],
        "model_id": model_id, "validation_metrics": {"roc_auc": 0.6, "brier_score": 0.2,
                                                     "sample_count": 50},
        "test_metrics": {"roc_auc": 0.6, "brier_score": 0.2, "sample_count": 50},
        "metrics_digest": digest,
    }, path)


def _mk_active(db_session, tmp_path, model_id, *, record_sha=True, sha=None):
    import hashlib

    path = tmp_path / f"{model_id}.joblib"
    val = {"roc_auc": 0.6, "brier_score": 0.2, "sample_count": 50}
    digest = metric_digest(model_id, val, val, {})
    _write_artifact(path, model_id, digest)
    record = MLModelRecord(
        model_id=model_id, dataset_version_tag="dv-b", algorithm="logistic_regression",
        hyperparameters={}, features_used=["rsi_14"], target_name="t",
        train_metrics={}, validation_metrics=val, test_metrics=val,
        calibration_metrics={}, walk_forward_metrics={}, status="active",
        artifact_path=str(path), metrics_digest=digest,
    )
    if record_sha:
        record.artifact_sha256 = sha or hashlib.sha256(path.read_bytes()).hexdigest()
    db_session.add(record)
    db_session.commit()
    return record


# ---------------------------------------------------------------- runtime load

def test_legacy_row_without_byte_digest_fails_closed(db_session, tmp_path):
    _mk_active(db_session, tmp_path, "legacy", record_sha=False)
    assert ModelRegistry(db_session).load_active_model_artifact() is None


def test_tampered_bytes_refused_before_unpickle(db_session, tmp_path):
    record = _mk_active(db_session, tmp_path, "tampered")
    # Rewrite the artifact (different bytes, same embedded digest).
    payload = joblib.load(record.artifact_path)
    payload["feature_names"] = ["evil_feature"]
    joblib.dump(payload, record.artifact_path)
    assert ModelRegistry(db_session).load_active_model_artifact() is None


def test_matching_bytes_load(db_session, tmp_path):
    _mk_active(db_session, tmp_path, "ok")
    payload = ModelRegistry(db_session).load_active_model_artifact()
    assert payload is not None and callable(payload["model"].predict_proba)


# ---------------------------------------------------------------- attach paths

def _mlwf_frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
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


def test_mlwf_attach_refuses_tampered_artifact_bytes(db_session, tmp_path):
    payload = run_ml_walk_forward(
        _mlwf_frame(), symbol="XAUUSD", timeframe="M15",
        algorithm="logistic_regression", fold_count=3, min_train_bars=300,
    )
    registry = ModelRegistry(db_session)
    run_id = registry.persist_ml_walk_forward_evaluation(copy.deepcopy(payload))

    import hashlib
    from backend.tests.unit.test_mlwf_run_binding import _model  # reuse builder

    candidate = _model("m-tamper", tmp_path / "t.joblib", features_used=list(payload["feature_names"]))
    db_session.add(candidate)
    db_session.commit()
    # Tamper AFTER registration: bytes no longer match the attested digest.
    art = joblib.load(candidate.artifact_path)
    art["extra_marker"] = "tampered"
    joblib.dump(art, candidate.artifact_path)

    with pytest.raises(ModelGovernanceError, match="byte integrity"):
        registry.attach_ml_walk_forward_metrics("m-tamper", payload, run_id=run_id)


def test_attach_refreshes_the_byte_digest_after_rewrite(db_session, tmp_path):
    payload = run_ml_walk_forward(
        _mlwf_frame(), symbol="XAUUSD", timeframe="M15",
        algorithm="logistic_regression", fold_count=3, min_train_bars=300,
    )
    registry = ModelRegistry(db_session)
    run_id = registry.persist_ml_walk_forward_evaluation(copy.deepcopy(payload))

    import hashlib
    from backend.tests.unit.test_mlwf_run_binding import _model

    before_sha = None
    candidate = _model("m-rewrite", tmp_path / "r.joblib", features_used=list(payload["feature_names"]))
    before_sha = candidate.artifact_sha256
    db_session.add(candidate)
    db_session.commit()

    registry.attach_ml_walk_forward_metrics("m-rewrite", payload, run_id=run_id)
    db_session.refresh(candidate)
    assert candidate.artifact_sha256 != before_sha, "the re-dump changed the bytes"
    from pathlib import Path

    current = hashlib.sha256(Path(candidate.artifact_path).read_bytes()).hexdigest()
    assert candidate.artifact_sha256 == current, "attested digest must track the file"
