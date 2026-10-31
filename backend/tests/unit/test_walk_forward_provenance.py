"""M8 round 2: real walk-forward provenance (regression tests).

The registry must accept walk-forward evidence ONLY from a real evaluation
produced by backend/app/services/backtest/walk_forward.py (per-fold engine
runs persisted with real costs + a provenance block the registry can
re-aggregate). A hand-made plain BacktestRun — even with top-level numbers
matching the payload EXACTLY — must never attest walk-forward evidence or
promote a model.

The synthetic OHLC frame and rule are test fixtures: they prove the evidence
CHAIN (evaluate → persist → attach → verify), not profitability of a real
account (no real market data, no orders).
"""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.core.config import settings
from backend.app.core.errors import ModelGovernanceError
from backend.app.core.metrics_integrity import metric_digest
from backend.app.db.base import Base
from backend.app.models.backtest import BacktestRun
from backend.app.models.ml_model import MLModelRecord
from backend.app.models.rule import TradingRule
from backend.app.services.backtest.walk_forward import run_walk_forward_backtest
from backend.app.services.ml.registry import ModelRegistry

SYMBOL = "XAUUSD"
TIMEFRAME = "M15"


@pytest.fixture
def db_session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ARTIFACTS_MODELS_DIR", tmp_path)
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture
def synthetic_frame():
    """Deterministic 300-bar OHLC frame (random walk + mild drift, seeded)."""
    rng = np.random.default_rng(42)
    n = 300
    steps = rng.normal(loc=0.02, scale=1.2, size=n)
    close = 2000.0 + np.cumsum(steps)
    open_ = np.roll(close, 1)
    open_[0] = 2000.0
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.4, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.4, size=n))
    idx = pd.date_range("2026-09-01 00:00", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": 100.0, "spread": 20}, index=idx)


@pytest.fixture
def approved_rule(db_session):
    rule = TradingRule(
        rule_id="rule_wf_rsi",
        name="Buy on RSI below 70",
        category="entry",
        description="WF provenance test rule",
        conditions=[{"indicator": "RSI", "operator": "<", "threshold": 70.0}],
        parameters={},
        symbol_scope=[SYMBOL],
        timeframe_scope=[TIMEFRAME],
        invalidation_conditions=[],
        source_pages=[],
        assumptions=[],
        ambiguities=[],
        approval_status="approved",
    )
    db_session.add(rule)
    db_session.commit()
    return rule


def _candidate(model_id, artifact_path, wf, auc=0.56, brier=0.19):
    """A candidate model whose attested payload already embeds `wf` (as if the
    payload arrived through attach — digest + artifact kept consistent)."""
    val = {"roc_auc": auc, "brier_score": brier, "sample_count": 50}
    digest = metric_digest(model_id, val, val, wf)
    estimator = DummyClassifier(strategy="prior").fit([[0.0], [1.0]], [0, 1])
    joblib.dump({
        "model": estimator, "scaler": None, "feature_names": ["rsi_14"],
        "model_id": model_id, "validation_metrics": val, "test_metrics": val,
        "metrics_digest": digest,
    }, artifact_path)
    return MLModelRecord(
        model_id=model_id, dataset_version_tag="wf-dv", algorithm="logistic_regression",
        hyperparameters={}, features_used=["rsi_14"], target_name="target_tp_before_sl",
        train_metrics={}, validation_metrics=val, test_metrics=val,
        calibration_metrics={}, walk_forward_metrics=wf, status="candidate",
        artifact_path=str(artifact_path), metrics_digest=digest,
    )


def _run_evaluation(db_session, synthetic_frame, approved_rule, fold_count=5):
    return run_walk_forward_backtest(
        db_session, synthetic_frame, symbol=SYMBOL, timeframe=TIMEFRAME,
        approved_rules=[approved_rule], fold_count=fold_count,
        spread_points=20, slippage_points=2, commission_per_lot=7.0,
    )


def test_evaluator_produces_real_provenance_run(db_session, synthetic_frame, approved_rule):
    """The evaluation persists ONE aggregate walk_forward run + one real
    engine run per fold, with a re-aggregatable provenance block."""
    result = _run_evaluation(db_session, synthetic_frame, approved_rule, fold_count=5)
    wf = result["walk_forward_metrics"]

    agg = db_session.query(BacktestRun).filter_by(run_id=result["run_id"]).first()
    assert agg is not None
    assert agg.strategy_name == "walk_forward"
    fold_rows = db_session.query(BacktestRun).filter(
        BacktestRun.strategy_name == "walk_forward_fold").all()
    assert len(fold_rows) == 5, "each fold must be a real persisted engine run"
    assert {r.run_id for r in fold_rows} == set(result["fold_run_ids"])

    prov = agg.assumptions["walk_forward"]
    assert len(prov["folds"]) == 5
    starts = [f["test_start"] for f in prov["folds"]]
    assert starts == sorted(starts), "fold windows must be chronological"
    assert prov["dataset_fingerprint"]["rows"] == len(synthetic_frame)
    assert len(prov["dataset_fingerprint"]["sha256"]) == 64

    # The stored aggregates re-aggregate from the persisted fold records.
    assert agg.total_trades == sum(f["total_trades"] for f in prov["folds"])
    assert agg.net_profit == pytest.approx(sum(f["net_profit"] for f in prov["folds"]))
    # Costs really flowed through the engine (commission flag consistent).
    assert prov["engine_config"]["commission_per_lot"] == 7.0
    # The payload mirrors the persisted provenance exactly.
    assert wf["folds"] == prov["folds"]
    assert wf["dataset_fingerprint"] == prov["dataset_fingerprint"]


def test_real_evaluation_payload_attaches_and_verifies(db_session, synthetic_frame, approved_rule, tmp_path):
    """Positive chain: a payload produced by the real evaluator passes the
    registry's provenance verification at attach time."""
    result = _run_evaluation(db_session, synthetic_frame, approved_rule)
    wf = result["walk_forward_metrics"]

    registry = ModelRegistry(db_session)
    run = db_session.query(BacktestRun).filter_by(run_id=result["run_id"]).first()
    assert registry._verify_backtest_provenance(wf, run) is None, (
        "a real evaluation payload must pass provenance verification")

    candidate = _candidate("wf-real-model", tmp_path / "wf-real.joblib", wf)
    db_session.add(candidate)
    db_session.commit()
    record = registry.attach_walk_forward_metrics(candidate.model_id, wf)
    assert record.walk_forward_metrics["backtest_run_id"] == result["run_id"]
    assert record.metrics_digest == metric_digest(
        candidate.model_id, record.validation_metrics, record.test_metrics, wf)


def test_handmade_plain_run_with_matching_numbers_is_rejected(db_session, synthetic_frame, approved_rule, tmp_path):
    """THE negative test: a hand-made ordinary BacktestRun — even when its
    top-level numbers match the wf payload EXACTLY — must never attest
    walk-forward evidence (it is not a walk-forward evaluation)."""
    result = _run_evaluation(db_session, synthetic_frame, approved_rule)
    wf = dict(result["walk_forward_metrics"])
    real_run = db_session.query(BacktestRun).filter_by(run_id=result["run_id"]).first()

    fake = BacktestRun(
        run_id="bt_fake_plain_run", strategy_name="Quantitative Adaptive Strategy",
        symbol=SYMBOL, timeframe=TIMEFRAME,
        start_date=real_run.start_date, end_date=real_run.end_date,
        initial_balance=10000.0, final_balance=10000.0 + real_run.net_profit,
        net_profit=real_run.net_profit, gross_profit=real_run.gross_profit,
        gross_loss=real_run.gross_loss, max_drawdown_percent=real_run.max_drawdown_percent,
        total_trades=real_run.total_trades, profit_factor=real_run.profit_factor,
        assumptions={},  # no walk_forward provenance block — plain hand-made run
    )
    db_session.add(fake)
    db_session.commit()

    wf["backtest_run_id"] = "bt_fake_plain_run"
    registry = ModelRegistry(db_session)
    failure = registry._verify_backtest_provenance(wf, fake)
    assert failure is not None and "not a walk-forward evaluation" in failure

    candidate = _candidate("wf-fake-model", tmp_path / "wf-fake.joblib", wf)
    db_session.add(candidate)
    db_session.commit()
    with pytest.raises(ModelGovernanceError, match="not a walk-forward evaluation"):
        registry.attach_walk_forward_metrics(candidate.model_id, wf)


def test_wf_strategy_name_without_provenance_block_is_rejected(db_session, synthetic_frame, approved_rule, tmp_path):
    """A run merely LABELLED walk_forward but carrying no provenance block is
    still not evidence."""
    wf = {
        "backtest_run_id": "bt_wf_labeled_only", "fold_count": 5,
        "trade_count": 40, "net_profit": 150.0, "profit_factor": 1.25,
        "max_drawdown_percent": 4.0, "spread_costs_included": True,
        "slippage_costs_included": True, "commission_costs_included": True,
        "folds": [], "dataset_fingerprint": {"sha256": "x" * 64},
    }
    db_session.add(BacktestRun(
        run_id="bt_wf_labeled_only", strategy_name="walk_forward",
        symbol=SYMBOL, timeframe=TIMEFRAME,
        start_date=pd.Timestamp("2026-01-01").to_pydatetime(),
        end_date=pd.Timestamp("2026-02-01").to_pydatetime(),
        initial_balance=10000.0, final_balance=10150.0, net_profit=150.0,
        gross_profit=200.0, gross_loss=-50.0, max_drawdown_percent=4.0,
        total_trades=40, profit_factor=1.25, assumptions={},
    ))
    db_session.commit()

    registry = ModelRegistry(db_session)
    run = db_session.query(BacktestRun).filter_by(run_id="bt_wf_labeled_only").first()
    failure = registry._verify_backtest_provenance(wf, run)
    assert failure is not None and "provenance block" in failure

    candidate = _candidate("wf-label-model", tmp_path / "wf-label.joblib", wf)
    db_session.add(candidate)
    db_session.commit()
    with pytest.raises(ModelGovernanceError, match="provenance block"):
        registry.attach_walk_forward_metrics(candidate.model_id, wf)


def test_tampered_or_inconsistent_fold_records_are_rejected(db_session, synthetic_frame, approved_rule):
    """Three forgery paths: (a) payload folds differ from the persisted ones,
    (b) persisted folds do not re-aggregate to the run's own top-level fields,
    (c) fold windows not chronological."""
    result = _run_evaluation(db_session, synthetic_frame, approved_rule)
    registry = ModelRegistry(db_session)
    run = db_session.query(BacktestRun).filter_by(run_id=result["run_id"]).first()
    wf = dict(result["walk_forward_metrics"])

    # (a) payload fold edited
    tampered = dict(wf)
    tampered["folds"] = [dict(f) for f in wf["folds"]]
    tampered["folds"][0]["net_profit"] = tampered["folds"][0]["net_profit"] + 100.0
    assert registry._verify_backtest_provenance(tampered, run) is not None

    # (b) persisted folds no longer re-aggregate to the run's fields
    import copy
    run2 = db_session.query(BacktestRun).filter_by(run_id=result["run_id"]).first()
    prov = copy.deepcopy(run2.assumptions)
    prov["walk_forward"]["folds"][0]["net_profit"] += 100.0
    run2.assumptions = prov
    db_session.commit()
    db_session.refresh(run2)
    assert registry._verify_backtest_provenance(wf, run2) is not None, (
        "persisted folds must re-aggregate to the run's own top-level numbers")

    # (c) chronological violation — tamper BOTH sides identically so the
    # equality check passes and the chronological gate is what rejects.
    run3 = db_session.query(BacktestRun).filter_by(run_id=result["run_id"]).first()
    prov3 = copy.deepcopy(run3.assumptions)
    prov3["walk_forward"]["folds"][0]["test_start"] = "2099-01-01 00:00:00+00:00"
    run3.assumptions = prov3
    db_session.commit()
    db_session.refresh(run3)
    wf3 = dict(wf)
    wf3["folds"] = copy.deepcopy(prov3["walk_forward"]["folds"])
    failure = registry._verify_backtest_provenance(wf3, run3)
    assert failure is not None and "chronological" in failure


def test_non_integer_trade_count_is_rejected(db_session, synthetic_frame, approved_rule):
    """float trade_count (e.g. 120.7) must not be silently truncated to 120 —
    the claim must be a real integer."""
    result = _run_evaluation(db_session, synthetic_frame, approved_rule)
    registry = ModelRegistry(db_session)
    run = db_session.query(BacktestRun).filter_by(run_id=result["run_id"]).first()
    wf = dict(result["walk_forward_metrics"])
    wf["trade_count"] = wf["trade_count"] + 0.7  # fractional claim
    failure = registry._verify_backtest_provenance(wf, run)
    assert failure is not None and "must be an integer" in failure


def test_backtest_endpoint_walk_forward_mode_is_wired(monkeypatch):
    """M8 production wiring: POST /backtest/run with walk_forward=true runs
    the real evaluator through the API and returns the provenance payload."""
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.core.config import settings
    from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider
    from backend.app.models.rule import TradingRule as RuleModel
    from backend.app.db.session import get_db as real_get_db
    from sqlalchemy.orm import sessionmaker as _sm

    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    TestSession = _sm(bind=engine)
    with TestSession() as session:
        session.add(RuleModel(
            rule_id="rule_wf_endpoint", name="WF endpoint rule", category="entry",
            description="d",
            conditions=[{"indicator": "RSI", "operator": "<", "threshold": 70.0}],
            parameters={}, symbol_scope=[SYMBOL], timeframe_scope=[TIMEFRAME],
            invalidation_conditions=[], source_pages=[], assumptions=[],
            ambiguities=[], approval_status="approved",
        ))
        session.commit()

        def test_get_db():
            db = TestSession()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[real_get_db] = test_get_db
        try:
            idx = pd.date_range("2026-09-01 00:00", periods=300, freq="15min", tz="UTC")
            rng = np.random.default_rng(7)
            close = 2000.0 + np.cumsum(rng.normal(0.02, 1.2, size=300))
            frame = pd.DataFrame({
                "open": np.roll(close, 1), "high": close + 0.5,
                "low": close - 0.5, "close": close,
                "volume": 100.0, "spread": 20}, index=idx)
            # pandas Copy-on-Write: chained `frame["open"].iloc[0] = ...` never
            # writes through (ChainedAssignmentError); .loc targets the frame.
            frame.loc[frame.index[0], "open"] = 2000.0

            class _SI:
                point = 0.01; spread = 20; tick_size = 0.01; tick_value = 1.0
                trade_contract_size = 100.0
                volume_min = 0.01; volume_step = 0.01; volume_max = 10.0

            monkeypatch.setattr(MT5MarketDataProvider, "connect", lambda self: True)
            monkeypatch.setattr(
                MT5MarketDataProvider, "get_historical_dataframe",
                lambda self, symbol, timeframe, count=100: frame)
            monkeypatch.setattr(MT5MarketDataProvider, "get_symbol_info", lambda self, s: _SI())
            monkeypatch.setattr(MT5MarketDataProvider, "disconnect", lambda self: None)

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/backtest/run",
                json={"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars_count": 300,
                      "spread_points": 20, "walk_forward": True, "wf_fold_count": 4},
                headers={"X-API-TOKEN": settings.API_TOKEN},
            )
        finally:
            app.dependency_overrides.pop(real_get_db, None)

    assert resp.status_code == 200, resp.text[:300]
    body = resp.json()
    assert body["strategy_name"] == "walk_forward"
    wf = body["walk_forward_metrics"]
    assert wf["fold_count"] == 4 and len(wf["folds"]) == 4
    assert wf["dataset_fingerprint"]["sha256"]
    run = TestSession().query(BacktestRun).filter_by(run_id=body["run_id"]).first()
    assert run is not None and run.strategy_name == "walk_forward"
    assert run.assumptions["walk_forward"]["folds"] == wf["folds"]
