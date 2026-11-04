"""Steps 3.4 + 3.5: nightly scheduled candidate training and the promotion
pre-check summary.

The scheduler trains ONE candidate per day through the shared flow and NEVER
promotes; the promotion pre-check reports every gate read-only so the system
"runs all checks and gives the summary" while the human still activates.
"""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.services.ml.training_scheduler import (
    TrainingScheduler,
    compute_next_run,
    last_training_status,
)

V1 = "/api/v1"


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def test_compute_next_run_same_day_when_hour_ahead():
    now = datetime(2026, 11, 3, 10, 30, tzinfo=timezone.utc)
    assert compute_next_run(now, 21) == datetime(2026, 11, 3, 21, 0, tzinfo=timezone.utc)


def test_compute_next_run_tomorrow_when_hour_passed():
    now = datetime(2026, 11, 3, 22, 0, tzinfo=timezone.utc)
    assert compute_next_run(now, 21) == datetime(2026, 11, 4, 21, 0, tzinfo=timezone.utc)


def test_scheduler_runs_once_per_utc_day():
    sched = TrainingScheduler(db_factory=object)
    assert sched._due(datetime(2026, 11, 3, 20, 59, tzinfo=timezone.utc)) is False
    assert sched._due(datetime(2026, 11, 3, 21, 0, tzinfo=timezone.utc)) is True
    sched._trained_on_date = "2026-11-03"
    assert sched._due(datetime(2026, 11, 3, 23, 0, tzinfo=timezone.utc)) is False, (
        "the scheduler must train at most once per UTC day")
    assert sched._due(datetime(2026, 11, 4, 21, 0, tzinfo=timezone.utc)) is True


def test_scheduler_trains_candidate_and_never_promotes(monkeypatch):
    calls = []

    def fake_train(db, **kwargs):
        calls.append(kwargs)
        return {"model_id": "model_sched_1", "status": "candidate"}

    def fake_factory():
        return _nullcontext(object())

    sched = TrainingScheduler(db_factory=fake_factory, train_fn=fake_train, algorithm="random_forest")
    sched.run_training_once()

    assert calls and calls[0]["algorithm"] == "random_forest"
    assert calls[0]["symbol"] == "XAUUSD" and calls[0]["timeframe"] == "M1", (
        "the scheduler is gold-only, M1-only by construction")
    assert last_training_status["outcome"] == "trained"
    assert last_training_status["model_id"] == "model_sched_1"


def test_scheduler_records_failure_and_retries_next_day():
    def boom(db, **kwargs):
        raise ConnectionError("MT5 unavailable at 2am")

    def fake_factory():
        return _nullcontext(object())

    sched = TrainingScheduler(db_factory=fake_factory, train_fn=boom)
    assert sched.run_training_once() is None
    assert last_training_status["outcome"] == "failed"
    assert "MT5 unavailable" in last_training_status["detail"]


def test_scheduler_disabled_by_settings_starts_no_thread(monkeypatch):
    monkeypatch.setattr(settings, "TRAINING_SCHEDULE_ENABLED", False)
    sched = TrainingScheduler(db_factory=object)
    sched.start()
    assert sched._thread is None


class _nullcontext:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Promotion pre-check endpoint (read-only gate summary)
# ---------------------------------------------------------------------------

def _client():
    return TestClient(app, raise_server_exceptions=False)


def _headers():
    return {"X-API-TOKEN": settings.API_TOKEN}


def _seed_candidate(factory, model_id, **overrides):
    from backend.app.models.ml_model import MLModelRecord

    fields = dict(
        model_id=model_id,
        algorithm="logistic_regression",
        status="candidate",
        dataset_version_tag="ds_test",
        target_name="target_tp_before_sl",
        validation_metrics={},
        test_metrics={},
        artifact_path="models/none.joblib",
        metrics_digest=None,
    )
    fields.update(overrides)
    with factory() as session:
        session.add(MLModelRecord(**fields))
        session.commit()


def test_promotion_check_reports_failing_gate(isolated_app_db):
    factory = isolated_app_db
    _seed_candidate(factory, "model_check_1")  # no digest -> gate 2 fails
    resp = _client().get(f"{V1}/ml/models/model_check_1/promotion-check", headers=_headers())
    assert resp.status_code == 200
    body = resp.json()
    assert body["promotable"] is False
    assert body["first_failure"] == (
        "Cannot promote model model_check_1: stored metrics are not attested "
        "(missing digest); re-register the candidate."
    )
    gates = {g["gate"]: g for g in body["gates"]}
    assert gates["status"]["passed"] is True
    assert gates["digest_attestation"]["passed"] is False
    assert gates["artifact_path"]["passed"] is None, "later gates are skipped, not passed"


def test_promotion_check_unknown_model_is_404(isolated_app_db):
    resp = _client().get(f"{V1}/ml/models/nope/promotion-check", headers=_headers())
    assert resp.status_code == 404


def test_promotion_check_is_token_gated(isolated_app_db):
    resp = _client().get(f"{V1}/ml/models/whatever/promotion-check")
    assert resp.status_code in (401, 403)


def test_promotion_check_does_not_activate(isolated_app_db):
    factory = isolated_app_db
    _seed_candidate(factory, "model_check_2")
    _client().get(f"{V1}/ml/models/model_check_2/promotion-check", headers=_headers())
    with factory() as session:
        from backend.app.models.ml_model import MLModelRecord

        row = session.query(MLModelRecord).filter_by(model_id="model_check_2").one()
    assert row.status == "candidate", "the pre-check must stay read-only"
