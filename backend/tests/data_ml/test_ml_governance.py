"""Tests for Model Training, Evaluation Metrics, and Controlled Governance Workflows."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.db.base import Base
from backend.app.services.market_data.mock_provider import MockMarketDataProvider
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.dataset.pipeline import DatasetPipeline
from backend.app.services.ml.trainer import ModelTrainer
from backend.app.services.ml.registry import ModelRegistry
from backend.app.core.errors import ModelGovernanceError


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


def test_model_training_registers_as_candidate_only(db_session, sample_dataset):
    """CRITICAL SAFETY TEST:
    A newly trained model MUST start in 'candidate' status and never become active automatically.
    """
    trainer = ModelTrainer(db=db_session)
    result = trainer.train_candidate(sample_dataset, algorithm="logistic_regression")

    assert result["status"] == "candidate"
    assert "roc_auc" in result["test_metrics"]
    assert "brier_score" in result["test_metrics"]

    registry = ModelRegistry(db_session)
    # Active model must be NONE initially
    assert registry.get_active_model_record() is None

    # Models list must show the candidate
    candidates = registry.list_models(status="candidate")
    assert len(candidates) == 1
    assert candidates[0].model_id == result["model_id"]


def test_controlled_model_promotion_and_archiving(db_session, sample_dataset):
    """Verify manual model promotion, archiving of previous active, and audit trail."""
    from backend.app.models.ml_model import MLModelRecord
    trainer = ModelTrainer(db=db_session)
    registry = ModelRegistry(db_session)

    # Train model 1
    m1 = trainer.train_candidate(sample_dataset, algorithm="logistic_regression")
    # Train model 2
    m2 = trainer.train_candidate(sample_dataset, algorithm="random_forest")

    passing_wf = {
        "backtest_run_id": "bt_wf_governance",
        # M8 round 2: the remaining numbers come from the shared builder so
        # the seeded run IS a walk-forward evaluation with re-aggregatable
        # fold records (the builder keeps the historical 150.0/50/1.25/4.0).
    }
    from backend.tests.unit._wf_evidence import build_walk_forward_payload, seed_walk_forward_run
    passing_wf = build_walk_forward_payload("bt_wf_governance")
    seed_walk_forward_run(db_session, "bt_wf_governance")

    # Attach required walk-forward metrics and passing validation/test metrics
    rec1 = db_session.query(MLModelRecord).filter_by(model_id=m1["model_id"]).first()
    val1 = dict(rec1.validation_metrics or {})
    val1.update({"roc_auc": 0.55, "brier_score": 0.20, "sample_count": 50})
    test1 = dict(rec1.test_metrics or {})
    test1.update({"roc_auc": 0.55, "brier_score": 0.20, "sample_count": 50})

    rec2 = db_session.query(MLModelRecord).filter_by(model_id=m2["model_id"]).first()
    val2 = dict(rec2.validation_metrics or {})
    val2.update({"roc_auc": 0.65, "brier_score": 0.18, "sample_count": 50})
    test2 = dict(rec2.test_metrics or {})
    test2.update({"roc_auc": 0.65, "brier_score": 0.18, "sample_count": 50})

    # Finding 5: walk-forward evidence flows through the registry's supported
    # path, which re-attests the digest and syncs the artifact in one write.
    registry.attach_walk_forward_metrics(m1["model_id"], passing_wf, validation_metrics=val1, test_metrics=test1)
    registry.attach_walk_forward_metrics(m2["model_id"], passing_wf, validation_metrics=val2, test_metrics=test2)

    # Promote Model 1
    promoted_1 = registry.promote_candidate(m1["model_id"], actor="QUANT_LEAD")
    assert promoted_1.status == "active"
    assert registry.get_active_model_record().model_id == m1["model_id"]

    # Now promote Model 2 (beats Model 1 on validation ROC-AUC)
    promoted_2 = registry.promote_candidate(m2["model_id"], actor="QUANT_LEAD")
    assert promoted_2.status == "active"
    assert registry.get_active_model_record().model_id == m2["model_id"]

    # Verify Model 1 was safely moved to ARCHIVED
    db_session.refresh(promoted_1)
    assert promoted_1.status == "archived"


def test_cannot_promote_rejected_model(db_session, sample_dataset):
    """Verify that a rejected model cannot be promoted."""
    trainer = ModelTrainer(db=db_session)
    registry = ModelRegistry(db_session)

    m = trainer.train_candidate(sample_dataset, algorithm="baseline")
    cand = registry.list_models()[0]
    cand.status = "rejected"
    db_session.commit()

    with pytest.raises(ModelGovernanceError):
        registry.promote_candidate(cand.model_id)
