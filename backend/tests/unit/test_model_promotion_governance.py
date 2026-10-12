"""Model-promotion governance regression tests."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.core.errors import ModelGovernanceError
from backend.app.db.base import Base
from backend.app.models.ml_model import MLModelRecord
from backend.app.services.ml.registry import ModelRegistry


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def _model(model_id, artifact_path, *, status="candidate", auc=0.55, brier=0.20):
    return MLModelRecord(
        model_id=model_id,
        dataset_version_tag="test-dataset-v1",
        algorithm="logistic_regression",
        hyperparameters={},
        features_used=["rsi_14"],
        target_name="target_tp_before_sl",
        train_metrics={},
        test_metrics={"roc_auc": auc, "brier_score": brier},
        calibration_metrics={},
        walk_forward_metrics={},
        status=status,
        artifact_path=str(artifact_path),
    )


def test_promotion_rejects_metrics_at_or_below_baseline(db_session, tmp_path):
    artifact = tmp_path / "weak.joblib"
    artifact.write_bytes(b"test artifact")
    candidate = _model("weak-model", artifact, auc=0.50, brier=0.20)
    db_session.add(candidate)
    db_session.commit()

    with pytest.raises(ModelGovernanceError, match="minimum gate"):
        ModelRegistry(db_session).promote_candidate(candidate.model_id)

    db_session.refresh(candidate)
    assert candidate.status == "candidate"


def test_promotion_rejects_missing_or_invalid_metrics(db_session, tmp_path):
    artifact = tmp_path / "invalid.joblib"
    artifact.write_bytes(b"test artifact")
    candidate = _model("invalid-model", artifact, auc=float("nan"), brier=0.20)
    db_session.add(candidate)
    db_session.commit()

    with pytest.raises(ModelGovernanceError, match="valid test ROC-AUC"):
        ModelRegistry(db_session).promote_candidate(candidate.model_id)


def test_promotion_requires_candidate_status(db_session, tmp_path):
    artifact = tmp_path / "archived.joblib"
    artifact.write_bytes(b"test artifact")
    model = _model("archived-model", artifact, status="archived")
    db_session.add(model)
    db_session.commit()

    with pytest.raises(ModelGovernanceError, match="only candidates"):
        ModelRegistry(db_session).promote_candidate(model.model_id)


def test_promotion_does_not_archive_active_if_candidate_is_weaker(db_session, tmp_path):
    active_artifact = tmp_path / "active.joblib"
    active_artifact.write_bytes(b"active artifact")
    candidate_artifact = tmp_path / "candidate.joblib"
    candidate_artifact.write_bytes(b"candidate artifact")
    active = _model("active-model", active_artifact, status="active", auc=0.56, brier=0.19)
    candidate = _model("weaker-model", candidate_artifact, auc=0.55, brier=0.20)
    db_session.add_all([active, candidate])
    db_session.commit()

    with pytest.raises(ModelGovernanceError, match="does not exceed active model"):
        ModelRegistry(db_session).promote_candidate(candidate.model_id)

    db_session.refresh(active)
    db_session.refresh(candidate)
    assert active.status == "active"
    assert candidate.status == "candidate"


def test_promotion_archives_previous_active_after_all_gates_pass(db_session, tmp_path, monkeypatch):
    active_artifact = tmp_path / "active2.joblib"
    active_artifact.write_bytes(b"active artifact")
    candidate_artifact = tmp_path / "candidate2.joblib"
    candidate_artifact.write_bytes(b"candidate artifact")
    active = _model("active-model-2", active_artifact, status="active", auc=0.53, brier=0.20)
    candidate = _model("better-model", candidate_artifact, auc=0.56, brier=0.19)
    db_session.add_all([active, candidate])
    db_session.commit()
    monkeypatch.setattr("backend.app.services.ml.registry.log_security_event", lambda *args, **kwargs: None)

    promoted = ModelRegistry(db_session).promote_candidate(candidate.model_id, actor="test")

    assert promoted.status == "active"
    assert active.status == "archived"
