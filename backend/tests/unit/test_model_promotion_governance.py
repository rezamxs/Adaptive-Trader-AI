"""Model-promotion governance regression tests."""
import hashlib
import pytest
import joblib
from sklearn.dummy import DummyClassifier
from backend.app.core.config import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.core.errors import ModelGovernanceError
from backend.app.core.metrics_integrity import metric_digest
from backend.app.db.base import Base
from backend.app.models.ml_model import MLModelRecord
from backend.app.services.ml.registry import ModelRegistry


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


def _write_artifact(path, model_id=None, validation_metrics=None, test_metrics=None, metrics_digest=None):
    """Write an artifact; when attestation values are supplied the payload
    carries them so promotion's H1 digest gate can verify it."""
    estimator = DummyClassifier(strategy="prior").fit([[0.0], [1.0]], [0, 1])
    payload = {"model": estimator, "scaler": None, "feature_names": ["rsi_14"]}
    if metrics_digest is not None:
        payload["model_id"] = model_id
        payload["validation_metrics"] = validation_metrics
        payload["test_metrics"] = test_metrics
        payload["metrics_digest"] = metrics_digest
    joblib.dump(payload, path)


def _passing_walk_forward_metrics():
    """M8 round 2: mirrors a real walk-forward evaluation's payload."""
    from backend.tests.unit._wf_evidence import build_walk_forward_payload
    return build_walk_forward_payload("bt_promo_gov")


def _seed_backtest_run(session):
    """M8 round 2: seeds the walk_forward run the payload claims as origin."""
    from backend.tests.unit._wf_evidence import seed_walk_forward_run
    return seed_walk_forward_run(session, "bt_promo_gov")


def _model(
    model_id, artifact_path, *, status="candidate", auc=0.55, brier=0.20,
    walk_forward_metrics=None,
):
    validation_metrics = {"roc_auc": auc, "brier_score": brier, "sample_count": 50}
    test_metrics = {"roc_auc": auc, "brier_score": brier, "sample_count": 50}
    wf = walk_forward_metrics if walk_forward_metrics is not None else {}
    digest = metric_digest(model_id, validation_metrics, test_metrics, wf)
    _write_artifact(
        artifact_path, model_id=model_id, validation_metrics=validation_metrics,
        test_metrics=test_metrics, metrics_digest=digest,
    )
    # Minor-17 contract: rows carry the byte digest verified pre-load.
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return MLModelRecord(
        model_id=model_id,
        dataset_version_tag="test-dataset-v1",
        algorithm="logistic_regression",
        hyperparameters={},
        features_used=["rsi_14"],
        target_name="target_tp_before_sl",
        train_metrics={},
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        calibration_metrics={},
        walk_forward_metrics=wf,
        status=status,
        artifact_path=str(artifact_path),
        metrics_digest=digest,
        artifact_sha256=artifact_sha256,
    )


def test_promotion_rejects_metrics_at_or_below_baseline(db_session, tmp_path):
    artifact = tmp_path / "weak.joblib"
    _write_artifact(artifact)
    candidate = _model("weak-model", artifact, auc=0.50, brier=0.20)
    db_session.add(candidate)
    db_session.commit()

    with pytest.raises(ModelGovernanceError, match="metrics do not pass"):
        ModelRegistry(db_session).promote_candidate(candidate.model_id)

    db_session.refresh(candidate)
    assert candidate.status == "candidate"


def test_promotion_rejects_missing_or_invalid_metrics(db_session, tmp_path):
    artifact = tmp_path / "invalid.joblib"
    _write_artifact(artifact)
    candidate = _model("invalid-model", artifact, auc=float("nan"), brier=0.20)
    db_session.add(candidate)
    db_session.commit()

    with pytest.raises(ModelGovernanceError, match="validation and test partitions"):
        ModelRegistry(db_session).promote_candidate(candidate.model_id)


def test_promotion_rejects_missing_walk_forward_backtest(db_session, tmp_path):
    artifact = tmp_path / "no-walk-forward.joblib"
    _write_artifact(artifact)
    candidate = _model("no-walk-forward", artifact)
    db_session.add(candidate)
    db_session.commit()

    with pytest.raises(ModelGovernanceError, match="Walk-forward backtest evidence is missing"):
        ModelRegistry(db_session).promote_candidate(candidate.model_id)

    db_session.refresh(candidate)
    assert candidate.status == "candidate"


def test_promotion_rejects_walk_forward_without_costs(db_session, tmp_path):
    artifact = tmp_path / "no-costs.joblib"
    _write_artifact(artifact)
    metrics = _passing_walk_forward_metrics()
    metrics["slippage_costs_included"] = False
    candidate = _model("no-costs", artifact, walk_forward_metrics=metrics)
    db_session.add(candidate)
    db_session.commit()

    with pytest.raises(ModelGovernanceError, match="must include spread, slippage, and commission"):
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
    _write_artifact(active_artifact)
    candidate_artifact = tmp_path / "candidate.joblib"
    _write_artifact(candidate_artifact)
    active = _model("active-model", active_artifact, status="active", auc=0.56, brier=0.19)
    candidate = _model(
        "weaker-model", candidate_artifact, auc=0.55, brier=0.20,
        walk_forward_metrics=_passing_walk_forward_metrics(),
    )
    _seed_backtest_run(db_session)
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
    _write_artifact(active_artifact)
    candidate_artifact = tmp_path / "candidate2.joblib"
    _write_artifact(candidate_artifact)
    active = _model("active-model-2", active_artifact, status="active", auc=0.53, brier=0.20)
    candidate = _model(
        "better-model", candidate_artifact, auc=0.56, brier=0.19,
        walk_forward_metrics=_passing_walk_forward_metrics(),
    )
    _seed_backtest_run(db_session)
    db_session.add_all([active, candidate])
    db_session.commit()
    monkeypatch.setattr("backend.app.services.ml.registry.log_security_event", lambda *args, **kwargs: None)

    promoted = ModelRegistry(db_session).promote_candidate(candidate.model_id, actor="test")

    assert promoted.status == "active"
    assert active.status == "archived"
