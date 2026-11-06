"""Regression tests for the 2026-10-30 full-audit round-3 fixes.

Covers the defects confirmed during the post-M15 repository audit:
- demo_executor: TRADE_RETCODE_PLACED must resolve via reconciliation
  (EXECUTION_UNKNOWN), never REJECTED; the stale-order sweep must isolate a
  failing order instead of aborting the whole sweep; the simulated (non-MT5)
  dispatch branch must apply the same duplicate-prevention contract.
- risk_engine: a None account context fails closed (REJECTED, not AttributeError).
- market_worker: a failed bar reservation skips the cycle instead of running
  the analysis unreserved and colliding with the unique bar_key.
- evaluator: measurement-failure sentinels fail the promotion Brier gate.
- registry: load_active_model_artifact verifies the attested digest; attach
  refuses artifact paths outside the configured artifact root; walk-forward
  cost flags must match the producing run's persisted engine_config.
- API: list endpoints bound the `limit` query parameter.

All MT5 interactions use stubs; no order is dispatched, no real terminal is
touched, EXECUTION_MODE stays SIGNAL_ONLY outside the explicitly monkeypatched
demo-gate tests.
"""
from datetime import datetime, timezone
import unittest.mock

import joblib
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.app.models  # noqa: F401 - register all ORM tables
from backend.app.core.config import settings, ExecutionMode
from backend.app.core.errors import ModelGovernanceError
from backend.app.db.base import Base, utc_now_naive
from backend.app.main import app
from backend.app.models.backtest import BacktestRun
from backend.app.models.ml_model import MLModelRecord
from backend.app.models.order import DemoOrder
from backend.app.models.signal import TradeSignal
from backend.app.services.market_data.base import AccountDetails, Candle, Quote, SymbolInfo
from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider
from backend.app.services.execution import demo_executor
from backend.app.services.execution.demo_executor import DemoTradingExecutor, EXEC_MSG_DUPLICATE
from backend.app.services.ml.evaluator import ModelEvaluator
from backend.app.services.ml.registry import ModelRegistry
from backend.app.services.risk.risk_engine import RiskEngine
from backend.app.core.metrics_integrity import metric_digest
from backend.tests.unit._wf_evidence import build_walk_forward_payload, seed_walk_forward_run

try:
    import MetaTrader5 as _real_mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not MT5_AVAILABLE,
    reason="the PLACED-retcode test builds retcodes from the Windows-only MetaTrader5 package",
)


# ---------------------------------------------------------------------------
# Shared stubs (mirrors test_audit_fixes_h6.py conventions)
# ---------------------------------------------------------------------------

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


def _approved_signal(db_session, signal_id="sig_r3_001", idem_key="XAUUSD_M15_202610300900_BUY"):
    sig = TradeSignal(
        signal_id=signal_id,
        idempotency_key=idem_key,
        symbol="XAUUSD",
        timeframe="M15",
        direction="BUY",
        strength=0.85,
        entry_price=1900.0,
        stop_loss=1895.0,
        take_profit=1910.0,
        rules_triggered=["rule_r3"],
        features_snapshot={},
        risk_status="APPROVED",
        calculated_volume=0.10,
        execution_mode="DEMO_TRADING",
        status="GENERATED",
        bar_time=utc_now_naive(),
    )
    db_session.add(sig)
    db_session.commit()
    return sig


class _StubProvider:
    """Deterministic non-MT5 provider: the executor takes the SIMULATED
    dispatch branch for it (not an MT5MarketDataProvider instance)."""

    def __init__(self):
        self._connected = True

    def connect(self):
        return True

    def is_connected(self):
        return True

    def disconnect(self):
        pass
    def get_account_info(self):
        return AccountDetails(
            login_masked="***", server="Demo-Server", currency="USD", leverage=100,
            balance=10000.0, equity=10000.0, margin=0.0, free_margin=10000.0,
            trade_mode=0, is_demo=True,
        )

    def get_current_quote(self, symbol, max_staleness_seconds=None):
        return Quote(symbol=symbol, time=datetime.now(timezone.utc),
                     bid=1899.9, ask=1900.1, spread_points=20)

    def get_historical_candles(self, symbol, timeframe, count=100, start_pos=0):
        """count synthetic CLOSED M1 candles ending 'now' (tz-aware UTC)."""
        base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        candles = []
        for i in range(count, 0, -1):
            t = base - __import__("datetime").timedelta(minutes=i + start_pos)
            candles.append(Candle(time=t, open=2000.0, high=2001.0, low=1999.0,
                                  close=2000.5, tick_volume=10, spread=20))
        return candles

    def get_symbol_info(self, symbol):
        return SymbolInfo(
            name=symbol, digits=2, point=0.1, spread=20, tick_size=0.1, tick_value=1.0,
            volume_min=0.01, volume_max=10.0, volume_step=0.01,
            trade_contract_size=100.0, currency_base="XAU", currency_profit="USD",
            currency_margin="USD",
        )

    def get_open_positions_count(self):
        return 0

    def get_open_positions(self):
        return []

    def get_daily_loss_percent(self):
        return 0.0

    def calculate_order_margin(self, *args, **kwargs):
        return 0.0


def _enable_demo(monkeypatch):
    monkeypatch.setattr(settings, "EXECUTION_MODE", ExecutionMode.DEMO_TRADING)
    monkeypatch.setattr(settings, "DEMO_TRADING_ENABLED", True)
    monkeypatch.setattr(settings, "ENFORCE_SESSION_FILTER", False)


class _FakeMT5Provider(MT5MarketDataProvider):
    """Looks like the real MT5 provider to the executor's isinstance gate so
    the dispatch takes the MT5 branch, but every data method is a stub and no
    terminal is touched (same convention as test_audit_fixes_h6.py)."""

    def __init__(self):
        self._connected = True

    def connect(self):
        return True

    def is_connected(self):
        return True

    def get_account_info(self):
        return AccountDetails(
            login_masked="***", server="Demo-Server", currency="USD", leverage=100,
            balance=10000.0, equity=10000.0, margin=0.0, free_margin=10000.0,
            trade_mode=0, is_demo=True,
        )

    def get_current_quote(self, symbol, max_staleness_seconds=None):
        return Quote(symbol=symbol, time=datetime.now(timezone.utc),
                     bid=1899.9, ask=1900.1, spread_points=20)

    def get_symbol_info(self, symbol):
        return SymbolInfo(
            name=symbol, digits=2, point=0.1, spread=20, tick_size=0.1, tick_value=1.0,
            volume_min=0.01, volume_max=10.0, volume_step=0.01,
            trade_contract_size=100.0, currency_base="XAU", currency_profit="USD",
            currency_margin="USD",
        )

    def get_open_positions_count(self):
        return 0

    def get_open_positions(self):
        return []

    def get_daily_loss_percent(self):
        return 0.0

    def calculate_order_margin(self, *args, **kwargs):
        return 0.0


# ---------------------------------------------------------------------------
# demo_executor
# ---------------------------------------------------------------------------

def test_placed_retcode_resolves_via_reconciliation_not_rejected(db_session, monkeypatch):
    """10008 TRADE_RETCODE_PLACED means the broker ACCEPTED the order: the row
    must become EXECUTION_UNKNOWN (idempotency reserved, reconcilable), never
    REJECTED — a REJECTED row would strand a possibly-executing position."""
    _enable_demo(monkeypatch)
    _approved_signal(db_session)

    class _StubMT5:
        TRADE_ACTION_DEAL = 1
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1
        ORDER_TIME_GTC = 0
        ORDER_FILLING_IOC = 1
        TRADE_RETCODE_DONE = 10009

        def terminal_info(self):
            class _Term:
                trade_allowed = True
            return _Term()

        def order_send(self, request):
            class _Res:
                retcode = _real_mt5.TRADE_RETCODE_PLACED  # 10008
                comment = "order placed"
            return _Res()

        def history_orders_get(self, *args, **kwargs):
            return None  # no broker history available in this stub

    monkeypatch.setattr(demo_executor, "mt5", _StubMT5())
    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)

    executor = DemoTradingExecutor(db_session, _FakeMT5Provider())
    sig = db_session.query(TradeSignal).first()
    success, order, msg = executor.execute_demo_signal(sig)

    assert success is False
    assert order is not None
    assert order.status == "EXECUTION_UNKNOWN", (
        "PLACED must be treated as outcome-unknown, not a definitive refusal")
    # Retry on the unresolved signal must be blocked (idempotency reserved).
    sig = db_session.query(TradeSignal).first()
    success2, order2, msg2 = executor.execute_demo_signal(sig)
    assert success2 is False
    assert "UNKNOWN" in msg2 or "UNKNOWN" in (order2.error_message or "")


def test_reconciliation_sweep_isolates_a_failing_order(db_session):
    """One order whose reconciliation raises (e.g. SQLite lock) must not abort
    the sweep: the session is rolled back and later stale orders still resolve."""
    now = utc_now_naive()
    order_common = dict(symbol="XAUUSD", order_type="BUY", volume=0.01,
                        requested_price=1900.0, stop_loss=1895.0, take_profit=1910.0)
    o1 = DemoOrder(order_id="ord_bad", signal_id="s1", idempotency_key="k1",
                   status="EXECUTION_UNKNOWN", created_at=now, **order_common)
    o2 = DemoOrder(order_id="ord_good", signal_id="s2", idempotency_key="k2",
                   status="PENDING", created_at=now, **order_common)
    db_session.add_all([o1, o2])
    db_session.commit()

    executor = DemoTradingExecutor(db_session, _StubProvider())
    calls = {"n": 0}

    def flaky_reconcile(order_rec):
        calls["n"] += 1
        if order_rec.order_id == "ord_bad":
            raise IntegrityError("simulated lock", {}, Exception("database is locked"))
        order_rec.status = "FILLED"
        db_session.commit()

    with unittest.mock.patch.object(DemoTradingExecutor, "_reconcile_unknown_order", side_effect=flaky_reconcile):
        reconciled = executor.reconcile_stale_pending_orders(max_age_minutes=0)

    assert reconciled == 1, "the healthy order must still be resolved after the failure"
    assert calls["n"] == 2
    assert db_session.query(DemoOrder).filter_by(order_id="ord_good").first().status == "FILLED"


def test_simulated_dispatch_race_returns_duplicate_message(db_session, monkeypatch):
    """A concurrent duplicate that slips past the idempotency pre-check hits
    the unique key at commit time in the SIMULATED branch too — the loser must
    get the duplicate-prevention contract, not a raw IntegrityError (500)."""
    _enable_demo(monkeypatch)
    sig = _approved_signal(db_session)

    real_commit = db_session.commit

    def integrity_error_once():
        raise IntegrityError("simulated concurrent insert", {}, Exception("UNIQUE constraint failed"))

    # First commit attempt inside the simulated branch collides.
    monkeypatch.setattr(db_session, "commit", integrity_error_once)
    executor = DemoTradingExecutor(db_session, _StubProvider())
    success, order, msg = executor.execute_demo_signal(sig)

    assert success is False
    assert EXEC_MSG_DUPLICATE in msg
    monkeypatch.setattr(db_session, "commit", real_commit)
    # The duplicate must NOT have created a second row.
    assert db_session.query(DemoOrder).count() == 0


# ---------------------------------------------------------------------------
# risk_engine
# ---------------------------------------------------------------------------

def test_risk_engine_rejects_missing_account_fail_closed(db_session):
    """A None account must produce the REJECTED contract, not AttributeError."""
    engine = RiskEngine(db_session)
    provider = _StubProvider()
    ok, lots, reason = engine.assess_signal(
        symbol="XAUUSD", direction="BUY", entry_price=1900.0,
        stop_loss=1895.0, take_profit=1910.0,
        account=None,  # missing context
        symbol_info=provider.get_symbol_info("XAUUSD"),
        quote=provider.get_current_quote("XAUUSD"),
        open_trades_count=0, daily_loss_percent=0.0,
        current_time_utc=datetime.now(timezone.utc),
        active_martingale_drawdown=0.0,
        enforce_session_filter=False,
    )
    assert ok is False
    assert "REJECTED_INVALID_ACCOUNT" in reason


# ---------------------------------------------------------------------------
# market_worker: reservation failure skips the cycle
# ---------------------------------------------------------------------------

def test_reservation_failure_skips_analysis_instead_of_unreserved_run(monkeypatch):
    """A non-IntegrityError reservation failure must end the cycle (no
    analysis, no unreserved pipeline, no guaranteed bar_key collision)."""
    from backend.app.services.worker.market_worker import MarketMonitoringWorker

    class _FailingFactory:
        """Session whose WorkerCycleRecord reservation commit fails hard."""
        def __init__(self):
            from sqlalchemy import create_engine as ce
            eng = ce("sqlite:///:memory:")
            Base.metadata.create_all(bind=eng)
            self.Session = sessionmaker(bind=eng)

        def __call__(self):
            from backend.app.models.worker_cycle import WorkerCycleRecord
            session = self.Session()
            real_commit = session.commit

            def flaky_commit():
                # Fail ONLY the bar-reservation insert (a WorkerCycleRecord is
                # pending); any earlier housekeeping commit goes through.
                if any(isinstance(obj, WorkerCycleRecord) for obj in session.new):
                    raise ConnectionError("simulated DB outage at reservation")
                return real_commit()

            session.commit = flaky_commit
            return session

    worker = MarketMonitoringWorker(
        data_provider=_StubProvider(), db_factory=_FailingFactory(),
        symbol="XAUUSD", timeframe="M1", poll_interval=1.0,
    )
    # Emergency-stop check runs before reservation; keep it inactive.
    result = worker.run_single_cycle()

    assert result["analysis_executed"] is False
    assert result["status_code"] == "RESERVATION_FAILED"


# ---------------------------------------------------------------------------
# evaluator sentinel fails the Brier gate
# ---------------------------------------------------------------------------

def test_evaluator_measurement_failure_sentinel_fails_brier_gate():
    """A NaN probability vector makes brier_score_loss raise: the sentinel must
    be a gate-FAILING value (1.0), not 0.25 — which exactly satisfied the
    promotion gate and fabricated passing evidence."""
    y_true = __import__("numpy").array([0, 1, 0, 1])
    y_pred = __import__("numpy").array([0, 1, 0, 1])
    y_prob = __import__("numpy").array([[0.5, 0.5], [float("nan"), float("nan")],
                                        [0.2, 0.8], [0.7, 0.3]])
    metrics = ModelEvaluator.evaluate(y_true, y_pred, y_prob)
    assert metrics["brier_score"] == 1.0
    assert metrics["brier_score"] > ModelRegistry.MAX_PROMOTION_BRIER_SCORE


def test_single_class_labels_sentinel_fails_brier_gate():
    import numpy as np
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.95, 0.99])
    metrics = ModelEvaluator.evaluate(y_true, y_pred, y_prob)
    assert metrics["brier_score"] == 1.0
    assert metrics["brier_score"] > ModelRegistry.MAX_PROMOTION_BRIER_SCORE


# ---------------------------------------------------------------------------
# registry: load digest verification + attach containment + cost-flag provenance
# ---------------------------------------------------------------------------

def _mk_candidate(db_session, tmp_path, model_id="m_r3", active=True, with_digest=True):
    from sklearn.dummy import DummyClassifier
    val = {"roc_auc": 0.60, "brier_score": 0.20, "sample_count": 50}
    digest = metric_digest(model_id, val, val, {}) if with_digest else None
    path = tmp_path / f"{model_id}.joblib"
    payload = {
        "model": DummyClassifier(strategy="prior").fit([[0.0], [1.0]], [0, 1]),
        "feature_names": ["rsi_14"],
        "validation_metrics": val, "test_metrics": val,
    }
    if with_digest:
        payload["metrics_digest"] = digest
    joblib.dump(payload, path)
    import hashlib
    artifact_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    rec = MLModelRecord(
        model_id=model_id, dataset_version_tag="dv-r3", algorithm="logistic_regression",
        hyperparameters={}, features_used=["rsi_14"], target_name="t",
        train_metrics={}, validation_metrics=val, test_metrics=val,
        calibration_metrics={}, walk_forward_metrics={},
        status="active" if active else "candidate",
        artifact_path=str(path), metrics_digest=digest,
        artifact_sha256=artifact_sha256,
    )
    db_session.add(rec)
    db_session.commit()
    return rec


def test_load_active_artifact_verifies_attested_digest(db_session, tmp_path, monkeypatch):
    """A tampered/digest-less artifact must NOT be served to runtime inference."""
    monkeypatch.setattr(settings, "ARTIFACTS_MODELS_DIR", tmp_path)
    registry = ModelRegistry(db_session)

    # Digest-less artifact (legacy H1 rows): fail closed.
    _mk_candidate(db_session, tmp_path, model_id="m_no_digest", with_digest=False)
    assert registry.load_active_model_artifact() is None
    db_session.query(MLModelRecord).filter_by(model_id="m_no_digest").delete()
    db_session.commit()

    # Artifact whose digest does not match the store: fail closed.
    _mk_candidate(db_session, tmp_path, model_id="m_mismatch", with_digest=True)
    rec = db_session.query(MLModelRecord).filter_by(model_id="m_mismatch").first()
    rec.metrics_digest = "a" * 64  # tampered store side
    db_session.commit()
    assert registry.load_active_model_artifact() is None
    db_session.query(MLModelRecord).filter_by(model_id="m_mismatch").delete()
    db_session.commit()

    # Matching digest: loads.
    _mk_candidate(db_session, tmp_path, model_id="m_ok", with_digest=True)
    payload = registry.load_active_model_artifact()
    assert payload is not None and callable(payload["model"].predict_proba)


def test_attach_rejects_artifact_path_outside_artifact_root(db_session, tmp_path, monkeypatch):
    """attach rewrites the artifact file: its path must obey the same
    containment contract as promote/load."""
    monkeypatch.setattr(settings, "ARTIFACTS_MODELS_DIR", tmp_path / "allowed")
    (tmp_path / "allowed").mkdir()

    run_id = "bt_wf_attach_containment"
    seed_walk_forward_run(db_session, run_id)
    wf = build_walk_forward_payload(run_id)
    wf["backtest_run_id"] = run_id

    # _mk_candidate writes the artifact at tmp_path/<id>.joblib — OUTSIDE the
    # configured root (tmp_path/allowed), which is exactly the violation.
    candidate = _mk_candidate(db_session, tmp_path, model_id="m_attach", active=False, with_digest=True)
    db_session.commit()

    registry = ModelRegistry(db_session)
    with pytest.raises(ModelGovernanceError, match="outside the configured artifact directory"):
        registry.attach_walk_forward_metrics(candidate.model_id, wf)


def test_zero_cost_walk_forward_cost_flags_are_rejected():
    """A payload claiming costs on a run whose persisted engine_config shows
    zero costs must fail provenance (fabricated cost evidence)."""
    from copy import deepcopy
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        run_id = "bt_wf_zero_cost"
        seed_walk_forward_run(session, run_id)
        wf = build_walk_forward_payload(run_id)  # flags all True, engine costs all > 0

        registry = ModelRegistry(session)
        run = session.query(BacktestRun).filter_by(run_id=run_id).first()
        assert registry._verify_backtest_provenance(wf, run) is None

        # Now fabricate the contradiction: claim costs, run with zero costs.
        # IMPORTANT: deepcopy FIRST, then mutate the copy — SQLAlchemy skips
        # the UPDATE when the new JSON value compares equal to the (already
        # in-place-mutated) old value.
        mutated = deepcopy(run.assumptions)
        mutated["walk_forward"]["engine_config"] = dict(
            mutated["walk_forward"]["engine_config"])
        mutated["walk_forward"]["engine_config"].update({
            "spread_points": 0, "slippage_points": 0, "commission_per_lot": 0.0})
        mutated["walk_forward"]["costs_included"] = {
            "spread": False, "slippage": False, "commission": False}
        run.assumptions = mutated
        session.commit()
        session.refresh(run)
        failure = registry._verify_backtest_provenance(wf, run)
        assert failure is not None and "contradicts the producing run" in failure

        # Honest flags on the zero-cost run pass provenance but MUST fail the
        # promotion costs gate.
        wf_honest = dict(wf)
        wf_honest["spread_costs_included"] = False
        wf_honest["slippage_costs_included"] = False
        wf_honest["commission_costs_included"] = False
        assert registry._verify_backtest_provenance(wf_honest, run) is None
        gate_failure = registry._walk_forward_failure(wf_honest)
        assert gate_failure is not None and "costs" in gate_failure
    finally:
        session.close()


# ---------------------------------------------------------------------------
# API: bounded limit parameters
# ---------------------------------------------------------------------------

def test_list_endpoints_reject_out_of_range_limit(isolated_app_db):
    client = TestClient(app)
    headers = {"X-API-TOKEN": settings.API_TOKEN}
    for url in ("/api/v1/signals", "/api/v1/signals/orders",
                "/api/v1/system/worker/cycles", "/api/v1/system/audit-logs"):
        assert client.get(f"{url}?limit=-1", headers=headers).status_code == 422, url
        assert client.get(f"{url}?limit=501", headers=headers).status_code == 422, url
        assert client.get(f"{url}?limit=50", headers=headers).status_code == 200, url
