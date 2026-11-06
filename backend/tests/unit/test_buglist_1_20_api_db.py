"""Buglist 2026-11-05 round: regression tests for items 13, 16, 17, 18, 19.

Item 13 — the database enforces the inter-record relations (FK constraints),
          not just code conventions.
Item 16 — an acquired shared provider is ALWAYS released, including when
          connect() raises or the /status endpoint takes the acquire path.
Item 17 — market-data endpoints are token-gated; the /health governance probe
          is TTL-cached instead of deserializing the model artifact per poll.
Item 18 — the heavy CPU endpoints (backtest run, ML train) enforce a
          per-minute call cap (429 beyond it).
Item 19 — free-text API inputs are length-bounded before they reach the
          unbounded DB Text columns.
"""
import types
from typing import Optional

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.event import listens_for
from sqlalchemy.pool import StaticPool
from fastapi import HTTPException

import backend.app.models  # noqa: F401 - register all ORM tables on Base
from backend.app.core.config import settings
from backend.app.db.base import Base


@pytest.fixture
def fk_session():
    """In-memory session with SQLite foreign-key enforcement ON (the runtime
    engine sets this PRAGMA; bare test engines must mirror it)."""
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})

    @listens_for(engine, "connect")
    def _fk_on(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session
    session.close()
    engine.dispose()


# ---------------------------------------------------------------- item 13 (FK)

def test_fk_rejects_chunk_without_document(fk_session):
    from backend.app.models.knowledge_chunk import KnowledgeChunk

    chunk = KnowledgeChunk(
        chunk_id="chunk_fk1", document_id="doc_missing", source_file="f.txt",
        location_type="page", location_reference="Page 1", title="t",
        content="x", key_concepts=[], parameters={}, exceptions_and_limits=[],
    )
    fk_session.add(chunk)
    with pytest.raises(Exception):
        fk_session.commit()


def test_fk_cascades_chunk_deletion_with_document(fk_session):
    from backend.app.models.document import Document
    from backend.app.models.knowledge_chunk import KnowledgeChunk

    doc = Document(document_id="doc_fk1", file_name="f.txt", file_type="txt",
                   file_size_bytes=1, processing_status="completed", pages_count=1,
                   content_sha256="1" * 64)
    fk_session.add(doc)
    fk_session.add(KnowledgeChunk(
        chunk_id="chunk_fk2", document_id="doc_fk1", source_file="f.txt",
        location_type="page", location_reference="Page 1", title="t",
        content="x", key_concepts=[], parameters={}, exceptions_and_limits=[]))
    fk_session.commit()

    fk_session.delete(doc)
    fk_session.commit()
    assert fk_session.query(KnowledgeChunk).count() == 0, "chunks die with their document"


def test_fk_set_null_keeps_rule_alive_when_document_dies(fk_session):
    from backend.app.models.document import Document
    from backend.app.models.rule import TradingRule

    doc = Document(document_id="doc_fk2", file_name="f.txt", file_type="txt",
                   file_size_bytes=1, processing_status="completed", pages_count=1,
                   content_sha256="2" * 64)
    fk_session.add(doc)
    fk_session.add(TradingRule(
        rule_id="rule_fk1", name="n", category="risk", description="d",
        conditions=[], parameters={}, symbol_scope=[], timeframe_scope=[],
        source_document="doc_fk2", approval_status="approved", version=1))
    fk_session.commit()

    fk_session.delete(doc)
    fk_session.commit()
    rule = fk_session.query(TradingRule).filter_by(rule_id="rule_fk1").one()
    assert rule.source_document is None, "SET NULL keeps the approved rule"


def test_fk_set_null_on_ml_walk_forward_run_when_model_dies(fk_session, tmp_path, monkeypatch):
    import hashlib
    import joblib
    from backend.app.models.ml_model import MLModelRecord
    from backend.app.models.ml_walk_forward_run import MLWalkForwardRun

    monkeypatch.setattr(settings, "ARTIFACTS_MODELS_DIR", tmp_path)
    artifact = tmp_path / "fk.joblib"
    joblib.dump({"marker": True}, artifact)
    model = MLModelRecord(
        model_id="m_fk1", dataset_version_tag="dv", algorithm="logistic_regression",
        hyperparameters={}, features_used=["f"], target_name="t",
        train_metrics={}, validation_metrics={}, test_metrics={},
        calibration_metrics={}, walk_forward_metrics={}, status="archived",
        artifact_path=str(artifact),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    fk_session.add(model)
    fk_session.add(MLWalkForwardRun(
        run_id="mlwfr_fk1", payload={"evidence_type": "ml_walk_forward"},
        digest="d" * 64, dataset_fingerprint=None, fold_count=3, model_id="m_fk1"))
    fk_session.commit()

    fk_session.delete(model)
    fk_session.commit()
    row = fk_session.query(MLWalkForwardRun).filter_by(run_id="mlwfr_fk1").one()
    assert row.model_id is None, "evidence survives its model row with the binding cleared"


# ---------------------------------------------------------------- item 16 (provider leak)

@pytest.fixture
def api_client(fk_session):
    from fastapi.testclient import TestClient
    from backend.app.main import app
    from backend.app.db.session import get_db

    def _override_db():
        yield fk_session

    app.dependency_overrides[get_db] = _override_db
    # raise_server_exceptions=False: item-16 tests EXPECT a 500 from a
    # connect() explosion instead of the test client re-raising it.
    client = TestClient(app, raise_server_exceptions=False)
    yield client, app
    app.dependency_overrides.pop(get_db, None)


_AUTH = lambda: {"X-API-TOKEN": settings.API_TOKEN}


def test_market_connect_exception_releases_the_provider(api_client, monkeypatch):
    """Buglist item 16: an exception from connect() used to strand one
    usage-counter unit per call on the market endpoints."""
    import backend.app.api.v1.endpoints.market as market_module

    counts = {"acquire": 0, "release": 0}

    class _Boom:
        def connect(self):
            raise RuntimeError("connect exploded")

    monkeypatch.setattr(market_module, "acquire_shared_provider",
                        lambda: counts.__setitem__("acquire", counts["acquire"] + 1) or _Boom())
    monkeypatch.setattr(market_module, "release_shared_provider",
                        lambda p: counts.__setitem__("release", counts["release"] + 1))

    client, _ = api_client
    resp = client.get("/api/v1/market-data/symbols", headers=_AUTH())
    assert resp.status_code == 500  # the connection error propagates as 500
    assert counts["acquire"] == 1 and counts["release"] == 1, (
        "the acquired provider must be released even when connect() raises")


def test_market_connect_false_releases_and_returns_503(api_client, monkeypatch):
    import backend.app.api.v1.endpoints.market as market_module

    counts = {"acquire": 0, "release": 0}

    class _Down:
        def connect(self):
            return False

    monkeypatch.setattr(market_module, "acquire_shared_provider",
                        lambda: counts.__setitem__("acquire", counts["acquire"] + 1) or _Down())
    monkeypatch.setattr(market_module, "release_shared_provider",
                        lambda p: counts.__setitem__("release", counts["release"] + 1))

    client, _ = api_client
    resp = client.get("/api/v1/market-data/symbols", headers=_AUTH())
    assert resp.status_code == 503
    assert counts["acquire"] == 1 and counts["release"] == 1


def test_system_status_acquire_path_releases_the_provider(api_client, monkeypatch):
    """Buglist item 16: the /status endpoint leaked one shared-provider usage
    counter unit on EVERY call that took its acquire path (no release)."""
    import backend.app.api.v1.endpoints.system as system_module

    counts = {"acquire": 0, "release": 0}

    class _Down:
        def connect(self):
            return False

        def is_connected(self):
            return False

    class _StubWorker:
        data_provider = types.SimpleNamespace(is_connected=lambda: False)

        def get_diagnostics(self):
            return {
                "liveness": "LIVE", "readiness": "DEGRADED",
                "service_ready": False, "service_ready_reason": "stub",
                "market_data_ready": False, "market_data_reason": "stub",
                "trading_ready": False, "trading_readiness_reason": "stub",
                "worker": {},
            }

    monkeypatch.setattr(system_module, "get_market_worker", lambda: _StubWorker())
    monkeypatch.setattr(system_module, "acquire_shared_provider",
                        lambda: counts.__setitem__("acquire", counts["acquire"] + 1) or _Down())
    monkeypatch.setattr(system_module, "release_shared_provider",
                        lambda p: counts.__setitem__("release", counts["release"] + 1))

    client, _ = api_client
    resp = client.get("/api/v1/system/status", headers=_AUTH())
    assert resp.status_code == 200, resp.text[:200]
    assert counts["acquire"] == 1 and counts["release"] == 1, (
        "/status must release the provider it acquired (was a leak)")


# ---------------------------------------------------------------- item 17 (gating + probe cache)

def test_market_endpoints_reject_missing_token(api_client):
    client, _ = api_client
    for path in ("/api/v1/market-data/symbols", "/api/v1/market-data/quote/XAUUSD",
                 "/api/v1/market-data/candles/XAUUSD"):
        resp = client.get(path)
        assert resp.status_code == 401, f"{path} must be token-gated (item 17)"


def test_governance_probe_is_ttl_cached(api_client, monkeypatch, tmp_path):
    """Buglist item 17: repeated /health polls must not deserialize the model
    artifact per poll — the probe result is cached for the configured TTL."""
    from backend.app.services.worker.market_worker import MarketMonitoringWorker

    monkeypatch.setattr(settings, "HEALTH_PROBE_CACHE_SECONDS", 60.0)
    worker = MarketMonitoringWorker(
        data_provider=types.SimpleNamespace(is_connected=lambda: False),
        db_factory=lambda: None,
    )
    probe_calls = {"n": 0}

    def _stub():
        probe_calls["n"] += 1
        return True, "READY_FOR_SIGNAL_GENERATION"

    worker._trading_governance_probe_uncached = _stub

    assert worker.trading_governance_probe() == (True, "READY_FOR_SIGNAL_GENERATION")
    assert worker.trading_governance_probe() == (True, "READY_FOR_SIGNAL_GENERATION")
    assert probe_calls["n"] == 1, "the second poll within the TTL must hit the cache"

    # TTL disabled: every poll re-evaluates.
    monkeypatch.setattr(settings, "HEALTH_PROBE_CACHE_SECONDS", 0.0)
    worker._gov_probe_cache = None
    worker.trading_governance_probe()
    assert probe_calls["n"] == 2


# ---------------------------------------------------------------- item 18 (rate limit)

def test_heavy_rate_limit_caps_backtest_runs(api_client, monkeypatch):
    from backend.app.core import rate_limit

    monkeypatch.setattr(settings, "HEAVY_RATE_LIMIT_PER_MINUTE", 2)
    rate_limit.reset_rate_limiter_for_tests()
    try:
        client, _ = api_client
        # The limiter runs BEFORE the handler: allowed calls reach the (MT5-
        # unavailable) handler and fail 503, but they still consume the quota.
        client.post("/api/v1/backtest/run", json={"symbol": "EURUSD"}, headers=_AUTH())
        client.post("/api/v1/backtest/run", json={"symbol": "EURUSD"}, headers=_AUTH())
        resp = client.post("/api/v1/backtest/run", json={"symbol": "EURUSD"}, headers=_AUTH())
        assert resp.status_code == 429, "the 3rd heavy call within a minute must be capped"
        assert "Retry-After" in resp.headers
        # The limiter is scoped to the heavy endpoints only.
        resp_other = client.get("/api/v1/backtest/runs", headers=_AUTH())
        assert resp_other.status_code != 429
    finally:
        rate_limit.reset_rate_limiter_for_tests()


def test_heavy_rate_limit_disabled_with_zero(api_client, monkeypatch):
    from backend.app.core import rate_limit

    monkeypatch.setattr(settings, "HEAVY_RATE_LIMIT_PER_MINUTE", 0)
    rate_limit.reset_rate_limiter_for_tests()
    client, _ = api_client
    for _ in range(4):
        resp = client.post("/api/v1/ml/train", json={"symbol": "EURUSD"}, headers=_AUTH())
        assert resp.status_code != 429


def test_heavy_rate_limit_keys_by_token_unit(fk_session, monkeypatch):
    """Two different tokens get independent windows (the counter is keyed by
    the hashed token, not global)."""
    from fastapi import Request
    from backend.app.core.rate_limit import check_heavy_rate_limit

    monkeypatch.setattr(settings, "HEAVY_RATE_LIMIT_PER_MINUTE", 1)
    scope = {"client": types.SimpleNamespace(host="127.0.0.1")}
    request = types.SimpleNamespace(client=scope["client"])

    def _rq():
        return request

    check_heavy_rate_limit("token-a", _rq())
    with pytest.raises(HTTPException) as exc:
        check_heavy_rate_limit("token-a", _rq())
    assert exc.value.status_code == 429
    # token-b still has its own full window.
    check_heavy_rate_limit("token-b", _rq())


# ---------------------------------------------------------------- item 19 (length bounds)

def test_batch_review_reason_is_length_bounded():
    from pydantic import ValidationError
    from backend.app.schemas.rule_schema import BatchReviewItem

    with pytest.raises(ValidationError, match="at most 2000"):
        BatchReviewItem(rule_id="r1", action="reject", reason="x" * 2001)
    item = BatchReviewItem(rule_id="r1", action="reject", reason="x" * 2000)
    assert len(item.reason) == 2000


def test_reject_rule_api_rejects_overlong_reason(api_client):
    client, _ = api_client
    resp = client.post(
        "/api/v1/rules/nonexistent/reject",
        json={"reason": "x" * 2001},
        headers=_AUTH(),
    )
    assert resp.status_code == 422, "over-long reasons must be refused at the schema"
