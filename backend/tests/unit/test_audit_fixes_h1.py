"""TDD tests for audit fix H1: promotion must read attested metrics only.

Metrics used by promote_candidate must be bound to a digest committed at
registration time (HMAC-SHA256 when METRIC_SIGNING_KEY is set, else SHA-256).
Tampering with the DB row, the joblib artifact, or dropping the digest must
all be rejected before any gate mutates model state.
"""
import hashlib
import json
import joblib
import pytest
from datetime import datetime
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.core.config import settings
from backend.app.core.errors import ModelGovernanceError
from backend.app.core.metrics_integrity import metric_digest, verify_metric_digest
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


def _passing_walk_forward_metrics():
    """M8 round 2: the wf payload mirrors what a REAL walk-forward evaluation
    persists (fold records + dataset fingerprint) — see _wf_evidence.py."""
    from backend.tests.unit._wf_evidence import build_walk_forward_payload
    return build_walk_forward_payload("bt_wf_h1_fixture")


def _seed_backtest_run(session, run_id: str = "bt_wf_h1_fixture"):
    """M8 round 2: provenance requires the referenced run to BE a walk-forward
    evaluation whose per-fold records re-aggregate to its own numbers — this
    seeds exactly what walk_forward.py would persist."""
    from backend.tests.unit._wf_evidence import seed_walk_forward_run
    return seed_walk_forward_run(session, run_id)


def _write_artifact(path, model_id=None, validation_metrics=None, test_metrics=None, metrics_digest=None):
    estimator = DummyClassifier(strategy="prior").fit([[0.0], [1.0]], [0, 1])
    payload = {
        "model": estimator,
        "scaler": None,
        "feature_names": ["rsi_14"],
    }
    if model_id is not None:
        payload["model_id"] = model_id
    if validation_metrics is not None:
        payload["validation_metrics"] = validation_metrics
    if test_metrics is not None:
        payload["test_metrics"] = test_metrics
    if metrics_digest is not None:
        payload["metrics_digest"] = metrics_digest
    joblib.dump(payload, path)


def _candidate(
    model_id, artifact_path, *, auc=0.56, brier=0.19, wf=None, digest=None, with_artifact_metrics=True
):
    val = {"roc_auc": auc, "brier_score": brier, "sample_count": 50}
    test = {"roc_auc": auc, "brier_score": brier, "sample_count": 50}
    walk_forward = wf if wf is not None else _passing_walk_forward_metrics()
    if digest is None:
        digest = metric_digest(model_id, val, test, walk_forward)
    if with_artifact_metrics:
        _write_artifact(
            artifact_path,
            model_id=model_id,
            validation_metrics=val,
            test_metrics=test,
            metrics_digest=digest,
        )
    else:
        _write_artifact(artifact_path)
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
        validation_metrics=val,
        test_metrics=test,
        calibration_metrics={},
        walk_forward_metrics=walk_forward,
        status="candidate",
        artifact_path=str(artifact_path),
        metrics_digest=digest,
        artifact_sha256=artifact_sha256,
    )


def test_metric_digest_is_tamper_evident():
    base = {"roc_auc": 0.6, "brier_score": 0.2, "sample_count": 50}
    d1 = metric_digest("m1", base, base, {})
    d2 = metric_digest("m1", base, base, {})
    assert d1 == d2 and len(d1) == 64
    tampered = dict(base)
    tampered["roc_auc"] = 0.99
    assert metric_digest("m1", tampered, base, {}) != d1
    assert metric_digest("m2", base, base, {}) != d1
    assert verify_metric_digest("m1", base, base, {}, d1)
    assert not verify_metric_digest("m1", tampered, base, {}, d1)


def test_promotion_rejects_unattested_candidate(db_session, tmp_path):
    artifact = tmp_path / "unattested.joblib"
    _write_artifact(artifact)
    candidate = _candidate("unattested", artifact, digest="")
    db_session.add(candidate)
    db_session.commit()

    with pytest.raises(ModelGovernanceError, match="not attested"):
        ModelRegistry(db_session).promote_candidate(candidate.model_id)
    db_session.refresh(candidate)
    assert candidate.status == "candidate"


def test_promotion_rejects_tampered_db_metrics(db_session, tmp_path):
    artifact = tmp_path / "tampered-db.joblib"
    candidate = _candidate("tampered-db", artifact)
    db_session.add(candidate)
    db_session.commit()

    # Simulate direct store tampering after registration. The metric blob is
    # reassigned wholesale because SQLAlchemy does not persist in-place JSON
    # mutations — mirroring a real UPDATE against the store.
    candidate.validation_metrics = dict(candidate.validation_metrics, roc_auc=0.99)
    db_session.commit()

    with pytest.raises(ModelGovernanceError, match="integrity"):
        ModelRegistry(db_session).promote_candidate(candidate.model_id)
    db_session.refresh(candidate)
    assert candidate.status == "candidate"


def test_promotion_rejects_artifact_metrics_mismatch(db_session, tmp_path):
    """Deliberate two-layer contract (Minor-17): rewriting the artifact bytes
    is caught FIRST by the pre-load byte-integrity gate; an attacker who ALSO
    re-attests the DB-side byte digest (DB write access) then trips the
    embedded-metrics cross-check ("artifact tamper")."""
    artifact = tmp_path / "mismatch.joblib"
    candidate = _candidate("mismatch-model", artifact)
    db_session.add(candidate)
    db_session.commit()
    _seed_backtest_run(db_session)  # wf provenance (M8) must pass to reach the artifact gate

    # Rewrite the artifact with inflated metrics while keeping the digest —
    # the bytes no longer match the attested artifact_sha256.
    payload = joblib.load(artifact)
    payload["validation_metrics"] = dict(payload["validation_metrics"], roc_auc=0.99)
    joblib.dump(payload, artifact)

    with pytest.raises(ModelGovernanceError, match="byte integrity"):
        ModelRegistry(db_session).promote_candidate(candidate.model_id)
    db_session.refresh(candidate)
    assert candidate.status == "candidate"


def test_promotion_rejects_artifact_tamper_with_refreshed_byte_digest(db_session, tmp_path):
    """The deeper tamper: file bytes rewritten AND the DB-side byte digest
    refreshed to match (attacker with DB write access). The post-load embedded
    metrics cross-check must still refuse promotion."""
    import hashlib as _hashlib

    artifact = tmp_path / "deep-mismatch.joblib"
    candidate = _candidate("deep-mismatch-model", artifact)
    db_session.add(candidate)
    db_session.commit()
    _seed_backtest_run(db_session)

    payload = joblib.load(artifact)
    payload["validation_metrics"] = dict(payload["validation_metrics"], roc_auc=0.99)
    joblib.dump(payload, artifact)
    candidate.artifact_sha256 = _hashlib.sha256(artifact.read_bytes()).hexdigest()
    db_session.commit()

    with pytest.raises(ModelGovernanceError, match="artifact tamper"):
        ModelRegistry(db_session).promote_candidate(candidate.model_id)
    db_session.refresh(candidate)
    assert candidate.status == "candidate"


def test_promotion_succeeds_for_intact_candidate(db_session, tmp_path, monkeypatch):
    active_artifact = tmp_path / "active.joblib"
    active = _candidate("weaker-active", active_artifact, auc=0.53, brier=0.20)
    active.status = "active"
    db_session.add(active)

    artifact = tmp_path / "good.joblib"
    candidate = _candidate("good-model", artifact, wf={})
    db_session.add(candidate)
    db_session.commit()
    monkeypatch.setattr("backend.app.services.ml.registry.log_security_event", lambda *a, **k: None)

    # Finding 5: the supported path for adding walk-forward evidence after
    # registration — it must re-attest the digest and update the artifact so
    # promotion sees one consistent attested payload. Provenance: the WF
    # payload points at a REAL persisted backtest run.
    _seed_backtest_run(db_session)
    registry = ModelRegistry(db_session)
    record = registry.attach_walk_forward_metrics(
        candidate.model_id, _passing_walk_forward_metrics()
    )
    assert record.walk_forward_metrics["trade_count"] == 50
    payload = joblib.load(candidate.artifact_path)
    assert payload["walk_forward_metrics"]["trade_count"] == 50
    assert payload["metrics_digest"] == record.metrics_digest

    promoted = registry.promote_candidate(candidate.model_id, actor="test")
    assert promoted.status == "active"
    db_session.refresh(active)
    assert active.status == "archived"


def test_attach_walk_forward_rejects_empty_and_missing_model(db_session, tmp_path):
    artifact = tmp_path / "wf-target.joblib"
    candidate = _candidate("wf-target", artifact)
    db_session.add(candidate)
    db_session.commit()
    registry = ModelRegistry(db_session)

    with pytest.raises(ValueError, match="non-empty dict"):
        registry.attach_walk_forward_metrics(candidate.model_id, {})
    with pytest.raises(ValueError, match="not found"):
        registry.attach_walk_forward_metrics("model_missing", {"trade_count": 1})


# ---------------------------------------------------------------------------
# Item 9: walk-forward provenance — no self-declared evidence, no digest-less
# artifact promotion, no post-promotion re-attestation.
# ---------------------------------------------------------------------------

def test_attach_rejects_wf_without_backtest_run_id(db_session, tmp_path):
    artifact = tmp_path / "wf-norun.joblib"
    candidate = _candidate("wf-norun", artifact)
    db_session.add(candidate)
    db_session.commit()
    registry = ModelRegistry(db_session)

    wf = _passing_walk_forward_metrics()
    wf.pop("backtest_run_id")
    with pytest.raises(ModelGovernanceError, match="backtest_run_id"):
        registry.attach_walk_forward_metrics(candidate.model_id, wf)


def test_attach_rejects_wf_with_nonexistent_run(db_session, tmp_path):
    artifact = tmp_path / "wf-badrun.joblib"
    candidate = _candidate("wf-badrun", artifact)
    db_session.add(candidate)
    db_session.commit()
    registry = ModelRegistry(db_session)

    wf = _passing_walk_forward_metrics()
    wf["backtest_run_id"] = "bt_does_not_exist_anywhere"
    with pytest.raises(ModelGovernanceError, match="does not exist in backtest_runs"):
        registry.attach_walk_forward_metrics(candidate.model_id, wf)


def test_attach_rejects_wf_numbers_that_dont_match_stored_run(db_session, tmp_path):
    """M8: the run existing is not provenance — the walk-forward numbers must
    BE the stored run's numbers, otherwise self-attested evidence passes the
    gate by pointing at an arbitrary real run."""
    artifact = tmp_path / "wf-mismatch.joblib"
    candidate = _candidate("wf-mismatch", artifact)
    db_session.add(candidate)
    db_session.commit()
    _seed_backtest_run(db_session)  # stored run: net_profit=150.0
    registry = ModelRegistry(db_session)

    wf = _passing_walk_forward_metrics()
    wf["net_profit"] = 999.0  # claimed, but not what the stored run produced
    with pytest.raises(ModelGovernanceError, match="does not match the stored"):
        registry.attach_walk_forward_metrics(candidate.model_id, wf)


def test_promotion_rejects_when_backtest_run_deleted_after_attach(db_session, tmp_path, monkeypatch):
    """M8: provenance is re-checked at promotion — a run deleted between
    attach and promote must block activation (fail closed)."""
    monkeypatch.setattr(
        "backend.app.services.ml.registry.log_security_event", lambda *a, **k: None)
    artifact = tmp_path / "wf-deleted-run.joblib"
    candidate = _candidate("wf-deleted-run", artifact)
    db_session.add(candidate)
    db_session.commit()
    _seed_backtest_run(db_session)
    registry = ModelRegistry(db_session)
    registry.attach_walk_forward_metrics(candidate.model_id, _passing_walk_forward_metrics())

    from backend.app.models.backtest import BacktestRun
    db_session.query(BacktestRun).filter(
        BacktestRun.run_id == "bt_wf_h1_fixture").delete()
    db_session.commit()

    with pytest.raises(ModelGovernanceError, match="no longer exists"):
        registry.promote_candidate(candidate.model_id)
    db_session.refresh(candidate)
    assert candidate.status == "candidate"


def test_promotion_rejects_wf_without_backtest_run_id_provenance(db_session, tmp_path):
    """M8: a walk-forward payload that bypassed attach (registered directly on
    the record) and carries no backtest_run_id must fail promotion even when
    its own numbers pass every other gate."""
    artifact = tmp_path / "wf-noprovenance.joblib"
    wf = _passing_walk_forward_metrics()
    wf.pop("backtest_run_id")
    candidate = _candidate("wf-noprovenance", artifact, wf=wf)
    db_session.add(candidate)
    db_session.commit()

    with pytest.raises(ModelGovernanceError, match="backtest_run_id"):
        ModelRegistry(db_session).promote_candidate(candidate.model_id)
    db_session.refresh(candidate)
    assert candidate.status == "candidate"


def test_attach_rejects_non_candidate_model(db_session, tmp_path):
    artifact = tmp_path / "wf-active.joblib"
    candidate = _candidate("wf-active", artifact)
    candidate.status = "active"
    db_session.add(candidate)
    db_session.commit()
    registry = ModelRegistry(db_session)

    _seed_backtest_run(db_session)
    with pytest.raises(ModelGovernanceError, match="status 'active'"):
        registry.attach_walk_forward_metrics(candidate.model_id, _passing_walk_forward_metrics())


def test_promotion_rejects_digest_less_artifact(db_session, tmp_path):
    """An artifact WITHOUT an attested digest is a provenance failure, not a
    pass: the payload could not be verified against the store."""
    artifact = tmp_path / "digestless.joblib"
    candidate = _candidate("digestless", artifact, with_artifact_metrics=False)
    candidate.walk_forward_metrics = _passing_walk_forward_metrics()
    candidate.metrics_digest = metric_digest(
        candidate.model_id, candidate.validation_metrics,
        candidate.test_metrics, candidate.walk_forward_metrics,
    )
    db_session.add(candidate)
    db_session.commit()
    _seed_backtest_run(db_session)  # wf provenance (M8) must pass to reach the digest gate
    registry = ModelRegistry(db_session)

    with pytest.raises(ModelGovernanceError, match="no attested metrics digest"):
        registry.promote_candidate(candidate.model_id)


# ---------------------------------------------------------------------------
# Item 24: HMAC mode — the production signing path (METRIC_SIGNING_KEY set)
# must actually run and detect tampering, not exist untested.
# ---------------------------------------------------------------------------

def test_metric_digest_hmac_mode_differs_from_sha256_and_detects_tamper(monkeypatch):
    from backend.app.core import metrics_integrity as mi

    monkeypatch.setenv("METRIC_SIGNING_KEY", "stage4-hmac-test-key")
    val = {"roc_auc": 0.6, "brier_score": 0.18, "sample_count": 40}
    test = {"roc_auc": 0.61, "brier_score": 0.17, "sample_count": 40}
    wf = {"fold_count": 3, "trade_count": 30, "net_profit": 10.0,
          "profit_factor": 1.1, "spread_costs_included": True,
          "slippage_costs_included": True, "commission_costs_included": True,
          "backtest_run_id": "bt_hmac"}

    hmac_digest = mi.metric_digest("model-hmac", val, test, wf)

    # Deterministic under the same key...
    assert mi.metric_digest("model-hmac", val, test, wf) == hmac_digest
    # ...differs from the unsigned SHA-256 mode...
    monkeypatch.delenv("METRIC_SIGNING_KEY", raising=False)
    unsigned_digest = mi.metric_digest("model-hmac", val, test, wf)
    assert hmac_digest != unsigned_digest, (
        "HMAC mode must produce a different digest than unsigned SHA-256")
    # ...key rotation changes the digest (key is a real input)...
    monkeypatch.setenv("METRIC_SIGNING_KEY", "stage4-hmac-test-key-rotated")
    assert mi.metric_digest("model-hmac", val, test, wf) != hmac_digest, (
        "a different signing key must produce a different digest")
    # ...and tampering with any metric fails HMAC verification.
    monkeypatch.setenv("METRIC_SIGNING_KEY", "stage4-hmac-test-key")
    tampered_val = dict(val, roc_auc=0.99)
    tampered = mi.metric_digest("model-hmac", tampered_val, test, wf)
    assert tampered != hmac_digest, "tampered metrics must fail HMAC verification"


def test_signing_key_is_honored_from_settings_not_only_os_environ(monkeypatch):
    """2026-10-31 audit regression: pydantic-settings loads .env into Settings
    but never injects it into os.environ, so a .env-only METRIC_SIGNING_KEY
    used to be silently ignored and the digests silently degraded to plain
    SHA-256. The settings field must now act as the fallback key source."""
    from backend.app.core import metrics_integrity as mi
    from backend.app.core.config import settings

    monkeypatch.delenv("METRIC_SIGNING_KEY", raising=False)
    val = {"roc_auc": 0.6, "brier_score": 0.18, "sample_count": 40}

    # Baseline: no key anywhere -> plain SHA-256 mode.
    unsigned_digest = mi.metric_digest("model-env", val, val, {})

    # Key provided ONLY through the Settings field (the .env path).
    monkeypatch.setattr(settings, "METRIC_SIGNING_KEY", "settings-only-key", raising=False)
    signed_digest = mi.metric_digest("model-env", val, val, {})
    assert signed_digest != unsigned_digest, (
        "a Settings-carried METRIC_SIGNING_KEY must activate HMAC mode, "
        "not be silently ignored")
    assert mi.verify_metric_digest("model-env", val, val, {}, signed_digest), (
        "verification must succeed under the Settings-carried key")

    # Tampering is detected in Settings-key mode too.
    tampered = mi.metric_digest("model-env", dict(val, roc_auc=0.99), val, {})
    assert tampered != signed_digest

    monkeypatch.setattr(settings, "METRIC_SIGNING_KEY", "", raising=False)
