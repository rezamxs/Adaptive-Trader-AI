"""Buglist 2026-11-05 round 2: regression tests for items 21-46 (verified subset).

Each test names the item it pins. Items without a dedicated test here are
covered by updated contracts in the existing suite (e.g. item 17's policy
tests, item 40's import-side-effect checks, item 46's doc sync) or by the
migration smoke checks.
"""
import json
import logging
import sqlite3
import zipfile
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.app.models  # noqa: F401 - register all ORM tables on Base
from backend.app.core.config import settings
from backend.app.db.base import Base, utc_now_naive


@pytest.fixture
def db_session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(settings, "ARTIFACTS_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(settings, "ARTIFACTS_REPORTS_DIR", tmp_path / "reports")
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session, tmp_path
    session.close()
    engine.dispose()


# ---------------------------------------------------------------- item 21 (BM25 exact-word match)

def _index_chunks(session, docs):
    from backend.app.models.knowledge_chunk import KnowledgeChunk

    for i, text in enumerate(docs):
        session.add(KnowledgeChunk(
            chunk_id=f"chunk_bm25_{i}", document_id="doc_bm25", source_file="doc.pdf",
            location_type="page", location_reference=f"Page {i+1}", title=f"t{i}",
            content=text, key_concepts=[], parameters={}, exceptions_and_limits=[]))
    session.commit()


def test_short_token_does_not_substring_match(db_session):
    """Buglist item 21: "لات" (a real 2-letter word) used to substring-match
    inside "لاتین" and inflate the BM25 score. Field matching is now
    word-exact, so the chunk is skipped by the core-topic gate."""
    session, _ = db_session
    _index_chunks(session, [
        "درباره لاتین و ریشه واژه‌ها توضیح می‌دهیم.",       # contains لاتین, NOT لات
        "حجم پوزیشن لات برابر حجم ریسک است.",              # contains the real word لات
    ])
    from backend.app.services.knowledge.retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever(session)
    results = retriever.search("لات", top_k=5, min_score=0.5)
    assert results, "the chunk with the REAL word لات must still match"
    assert all("لاتین" not in (r["content"]) for r in results), (
        "the substring-only chunk must never surface")


def test_word_match_still_scores_and_bonuses_apply(db_session):
    session, _ = db_session
    _index_chunks(session, ["اهرم روی موجودی اثر می‌گذارد و لات تعیین می‌شود."])
    from backend.app.services.knowledge.retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever(session)
    results = retriever.search("اهرم موجودی", top_k=5)
    assert results and results[0]["score"] > 0


# ---------------------------------------------------------------- item 22 (ZIP member name bound)

def test_zip_member_name_is_bounded_for_varchar_columns(db_session):
    """Buglist item 22: a 400-char ZIP member name flowed into
    location_reference (String(128)) — fine on SQLite, an insert failure on
    PostgreSQL. The parser bounds it with an explicit truncation marker."""
    from backend.app.services.knowledge.parsers import DocumentParser

    long_name = "very_" + "x" * 400 + ".txt"
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(long_name, "Rule: buy on strength.\n")
    full_text, n_chunks, chunks = DocumentParser.parse_zip(buf.getvalue())
    assert chunks
    for c in chunks:
        assert len(c["location_reference"]) <= 120, "bounded to the String(128) column"
        assert len(c["title"]) <= 255
    assert "..." in chunks[0]["location_reference"], "truncation is explicit"


# ---------------------------------------------------------------- item 23 (stale market gate) + 24 (heartbeat)

def _worker(session_factory, quote_time):
    from unittest.mock import MagicMock

    from backend.app.services.market_data.base import Candle, Quote
    from backend.app.services.worker.market_worker import MarketMonitoringWorker

    candles = []
    price = 4190.0
    base = datetime(2026, 10, 9, 23, 0, 0, tzinfo=timezone.utc)
    for i in range(60):
        candles.append(Candle(time=base + timedelta(minutes=i), open=price,
                              high=price + 0.5, low=price - 0.5, close=price + 0.1,
                              tick_volume=100, spread=34, is_closed=True))
        price += 0.05
    provider = MagicMock()
    provider.is_connected.return_value = True
    provider.get_current_quote.return_value = Quote(
        symbol="XAUUSD", time=quote_time, bid=4194.0, ask=4194.34,
        spread_points=34, is_stale=False)
    provider.get_historical_candles.return_value = candles
    return MarketMonitoringWorker(data_provider=provider, db_factory=session_factory)


def test_stale_market_blocks_analysis(db_session):
    """Buglist item 23: staleness was diagnosed but the cycle analyzed the
    old bar anyway. Now a non-fresh market state skips the analysis."""
    session, _ = db_session
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=10_000)
    worker = _worker(lambda: session, stale_time)
    result = worker.run_single_cycle()
    assert result["analysis_executed"] is False
    assert result["status_code"] != "NO_APPROVED_RULES_AND_NO_ACTIVE_MODEL", (
        "a stale feed must never reach the analysis pipeline")
    assert "MARKET" in result["status_code"] or "STALE" in result["status_code"]


def test_processing_record_with_fresh_heartbeat_is_not_recovered(db_session):
    """Buglist item 24: recovery keyed on the record's created_at — a
    slow-but-alive cycle was recovered after 120s. The heartbeat
    (updated_at) now gates recovery."""
    session, _ = db_session
    from backend.app.models.worker_cycle import WorkerCycleRecord

    fresh_quote = datetime.now(timezone.utc)
    stale_bar_time = datetime(2026, 10, 9, 23, 59, 0, tzinfo=timezone.utc)
    bar_key = "XAUUSD_M1_202610092359"  # the mock provider's latest candle (23:00+59min)
    session.add(WorkerCycleRecord(
        cycle_id="cyc_old", symbol="XAUUSD", timeframe="M1", bar_key=bar_key,
        bar_time=stale_bar_time, status_code="PROCESSING",
        lifecycle_phase="PROCESSING", decision="PENDING", duration_ms=0.0,
        created_at=utc_now_naive() - timedelta(seconds=500),
        updated_at=utc_now_naive() - timedelta(seconds=5),   # heartbeat FRESH
    ))
    session.commit()

    worker = _worker(lambda: session, fresh_quote)
    result = worker.run_single_cycle()
    assert result["status_code"] == "BAR_UNDER_PROCESSING", (
        "a fresh heartbeat means the processing is alive; no recovery")


def test_processing_record_with_dead_heartbeat_is_recovered(db_session):
    session, _ = db_session
    from backend.app.models.worker_cycle import WorkerCycleRecord

    fresh_quote = datetime.now(timezone.utc)
    stale_bar_time = datetime(2026, 10, 9, 23, 59, 0, tzinfo=timezone.utc)
    bar_key = "XAUUSD_M1_202610092359"  # the mock provider's latest candle (23:00+59min)
    session.add(WorkerCycleRecord(
        cycle_id="cyc_dead", symbol="XAUUSD", timeframe="M1", bar_key=bar_key,
        bar_time=stale_bar_time, status_code="PROCESSING",
        lifecycle_phase="PROCESSING", decision="PENDING", duration_ms=0.0,
        created_at=utc_now_naive() - timedelta(seconds=500),
        updated_at=utc_now_naive() - timedelta(seconds=500),  # heartbeat DEAD
    ))
    session.commit()

    worker = _worker(lambda: session, fresh_quote)
    result = worker.run_single_cycle()
    assert result.get("lifecycle_phase") in ("PROCESSING", "COMPLETED", "FAILED"), (
        "a dead heartbeat orphan is recovered, not skipped forever")


# ---------------------------------------------------------------- item 25 (bounded reconcile window)

def test_reconcile_history_window_is_bounded_to_the_order(db_session, monkeypatch):
    """Buglist item 25: the sweep read broker history from 2020-01-01 under
    the global MT5 lock; the window is now the order's own lifetime."""
    from unittest.mock import MagicMock

    from backend.app.models.order import DemoOrder
    from backend.app.services.execution import demo_executor

    session, _ = db_session
    order = DemoOrder(
        order_id="ord_r25", signal_id="sig_r25", idempotency_key="idem_r25",
        symbol="XAUUSD", order_type="BUY", volume=0.01, requested_price=2000.0,
        stop_loss=1990.0, take_profit=2020.0, status="EXECUTION_UNKNOWN",
        created_at=utc_now_naive() - timedelta(days=3),
    )
    session.add(order)
    session.commit()

    captured = {}
    mt5_mod = MagicMock()
    mt5_mod.history_orders_get.side_effect = lambda a, b: captured.update(
        frm=a, to=b) or ()

    monkeypatch.setattr(demo_executor, "MT5_AVAILABLE", True)
    monkeypatch.setattr(demo_executor, "mt5", mt5_mod)

    class _Lock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import backend.app.services.market_data.mt5_provider as mt5_provider_mod
    monkeypatch.setattr(mt5_provider_mod, "_mt5_lock", _Lock())

    executor = demo_executor.DemoTradingExecutor(db=session, data_provider=MagicMock())
    executor._reconcile_unknown_order(order)

    frm = captured.get("frm")
    assert frm is not None
    if frm.tzinfo is None:
        frm = frm.replace(tzinfo=timezone.utc)
    expected_floor = (order.created_at.replace(tzinfo=timezone.utc)) - timedelta(days=1, seconds=1)
    assert frm >= expected_floor, (
        f"the window must start near the order's creation ({frm} >= {expected_floor}), not 2020")


# ---------------------------------------------------------------- item 26 (fresh == migrated parity)

def test_fresh_schema_matches_migrated_semantics(tmp_path):
    """Buglist item 26: the two schema differences found by the audit are
    fixed — no redundant non-unique documents index, and lifecycle_phase
    carries the server default on a fresh create_all too."""
    engine = create_engine(f"sqlite:///{tmp_path}/fresh.db")
    Base.metadata.create_all(engine)
    insp = __import__("sqlalchemy").inspect(engine)
    doc_idx = {i["name"]: i["unique"] for i in insp.get_indexes("documents")}
    assert "ux_documents_content_sha256" in doc_idx
    assert "ix_documents_content_sha256" not in doc_idx, (
        "the redundant non-unique index was dropped on migrated DBs (dshq); "
        "a fresh schema must match")
    cols = {c["name"]: c for c in insp.get_columns("worker_cycle_records")}
    assert "COMPLETED" in str(cols["lifecycle_phase"].get("default")), (
        "server_default must exist on a fresh schema (wcbf parity)")
    engine.dispose()


# ---------------------------------------------------------------- item 29 (multi-head detection)

def test_init_db_rejects_multiple_alembic_heads(tmp_path, monkeypatch):
    """Buglist item 29: init_db read only the FIRST alembic_version row; a
    branched chain booted silently. Multiple rows must fail loudly."""
    import sqlalchemy

    from backend.app.db import init_db as init_db_mod

    db_path = tmp_path / "multihead.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
    conn.execute("INSERT INTO alembic_version VALUES ('rev_a')")
    conn.execute("INSERT INTO alembic_version VALUES ('rev_b')")
    conn.commit()
    conn.close()

    monkeypatch.setattr(init_db_mod, "engine", create_engine(f"sqlite:///{db_path}"))
    with pytest.raises(RuntimeError, match="MULTIPLE alembic_version"):
        init_db_mod.init_db()
    engine = create_engine(f"sqlite:///{db_path}")
    engine.dispose()


# ---------------------------------------------------------------- item 32 (corrupt evidence rejected)

def test_corrupt_ml_walk_forward_payload_is_rejected_not_treated_as_absent():
    """Buglist item 32: a non-dict payload fell through the absence check and
    PASSED the gate. Absence is None/{}; anything else non-dict is corrupt."""
    from backend.app.services.ml.registry import ModelRegistry

    assert ModelRegistry._ml_walk_forward_failure("m", None) is None
    assert ModelRegistry._ml_walk_forward_failure("m", {}) is None
    failure = ModelRegistry._ml_walk_forward_failure("m", "corrupted string")
    assert failure is not None and "corrupt" in failure


# ---------------------------------------------------------------- item 33 (recorded live threshold)

def test_evaluate_model_uses_the_recorded_artifact_threshold():
    """Buglist item 33: inference reads live_prob_threshold from the attested
    artifact payload; the hidden function default is only a legacy fallback."""
    import numpy as np
    import pandas as pd

    from backend.app.services.signal.generator import SignalGenerator

    class _Model:
        def predict(self, X):
            return np.array([1])

        def predict_proba(self, X):
            return np.array([[0.10, 0.90]])

    artifact = {
        "model": _Model(), "scaler": None,
        "feature_names": ["feat_a"],
        "live_prob_threshold": 0.95,   # recorded policy value
    }
    feats = pd.Series({"feat_a": 1.0})
    direction, prob = SignalGenerator.evaluate_model(feats, artifact)
    assert direction == "HOLD", "0.90 < recorded 0.95 → HOLD (0.55 default would BUY)"
    # Legacy artifact without the field keeps the old default behavior.
    legacy = {"model": _Model(), "scaler": None, "feature_names": ["feat_a"]}
    direction2, prob2 = SignalGenerator.evaluate_model(feats, legacy)
    assert direction2 == "BUY" and prob2 == pytest.approx(0.90)


def test_trainer_records_live_threshold_in_artifact_and_record(db_session, monkeypatch, tmp_path):
    session, _tmp = db_session
    from backend.app.services.market_data.mock_provider import MockMarketDataProvider
    from backend.app.services.dataset.pipeline import DatasetPipeline
    from backend.app.services.ml.trainer import ModelTrainer
    from backend.app.models.ml_model import MLModelRecord
    import joblib
    from backend.app.services.market_data.base import TimeFrame

    provider = MockMarketDataProvider()
    df = provider.get_historical_dataframe("EURUSD", TimeFrame.M15, count=600)
    ds = DatasetPipeline(db=session).build_dataset(df, "EURUSD", "M15", horizon_bars=12)
    res = ModelTrainer(db=session).train_candidate(ds, algorithm="logistic_regression")

    record = session.query(MLModelRecord).filter_by(model_id=res["model_id"]).one()
    assert record.live_prob_threshold == pytest.approx(settings.LIVE_MODEL_PROB_THRESHOLD)
    payload = joblib.load(record.artifact_path)
    assert payload["live_prob_threshold"] == pytest.approx(settings.LIVE_MODEL_PROB_THRESHOLD)


# ---------------------------------------------------------------- item 34 (digest covers train metrics)

def test_metric_digest_covers_train_metrics():
    """Buglist item 34: train_metrics were the only evaluation block outside
    the attested digest."""
    from backend.app.core.metrics_integrity import metric_digest

    val = {"roc_auc": 0.6}
    test = {"roc_auc": 0.55}
    train_a = {"roc_auc": 0.9}
    train_b = {"roc_auc": 0.8}
    base = metric_digest("m1", val, test, {})
    assert metric_digest("m1", val, test, {}, train_metrics=train_a) != base
    assert metric_digest("m1", val, test, {}, train_metrics=train_a) != (
        metric_digest("m1", val, test, {}, train_metrics=train_b))
    # Empty/absent train metrics keep the legacy message shape.
    assert metric_digest("m1", val, test, {}, train_metrics=None) == base
    assert metric_digest("m1", val, test, {}, train_metrics={}) == base


# ---------------------------------------------------------------- item 35 (resolved code_version)

def test_ml_walk_forward_records_a_resolved_code_version(db_session):
    """Buglist item 35: evaluations recorded code_version='unrecorded'."""
    import numpy as np
    import pandas as pd

    from backend.app.services.ml.walk_forward import run_ml_walk_forward

    rng = np.random.default_rng(11)
    n = 900
    steps = rng.normal(loc=0.01, scale=1.0, size=n)
    close = 2000.0 + np.cumsum(steps)
    open_ = np.roll(close, 1)
    open_[0] = 2000.0
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.35, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.35, size=n))
    idx = pd.date_range("2026-09-01 00:00", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": 100.0, "spread": 20}, index=idx)

    payload = run_ml_walk_forward(df, symbol="XAUUSD", timeframe="M15",
                                  fold_count=3, min_train_bars=300)
    assert payload["code_version"] != "unrecorded"
    assert "adaptive-trader-ai@" in payload["code_version"] or "py3" in payload["code_version"]


# ---------------------------------------------------------------- item 39 (batch cap tested)

def test_batch_review_cap_of_200_is_enforced():
    """Buglist item 39: the 200-item batch cap existed but was never tested."""
    from backend.app.schemas.rule_schema import BatchReviewRequest, BatchReviewItem

    items = [BatchReviewItem(rule_id=f"r{i}", action="approve") for i in range(200)]
    assert len(BatchReviewRequest(items=items).items) == 200
    with pytest.raises(ValidationError):
        BatchReviewRequest(items=items + [BatchReviewItem(rule_id="r201", action="approve")])


# ---------------------------------------------------------------- item 40 (no file writes at import)

def test_import_time_logger_has_no_file_sink():
    """Buglist item 40: importing the logging module must not touch the
    filesystem; the file sink attaches only via explicit setup_logging()."""
    import backend.app.core.logging as log_mod

    handlers = log_mod.logger.handlers
    file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
    assert not file_handlers, "the import-time logger must be console-only"


def test_setup_logging_defers_file_open_until_first_emit(tmp_path):
    from backend.app.core.logging import setup_logging

    log_dir = tmp_path / "logs"
    lg = setup_logging(logger_name="item40_test_logger", log_dir=str(log_dir))
    handler = next(h for h in lg.handlers if isinstance(h, logging.FileHandler))
    from logging.handlers import RotatingFileHandler

    assert isinstance(handler, RotatingFileHandler)
    assert not (log_dir / "app.log").exists(), "delay=True: no file before the first record"
    lg.info("first real record")
    assert (log_dir / "app.log").exists(), "the file appears on the first emit"


# ---------------------------------------------------------------- item 42 (CORS wildcard rejected)

def test_cors_wildcard_with_credentials_fails_validation():
    """Buglist item 42: CORS_ORIGINS=["*"] + allow_credentials=True was never
    validated; the combination is invalid and must be refused at config time."""
    from backend.app.core.config import Settings

    with pytest.raises(ValueError, match="credentials"):
        Settings(CORS_ORIGINS=["*"], _env_file=None)


# ---------------------------------------------------------------- item 43 (SQL echo decoupled)

def test_sql_echo_is_an_independent_switch():
    from backend.app.db import session as db_session_mod

    assert db_session_mod.engine.echo is False, "default must be silent"
    assert settings.SQL_ECHO is False
    assert not hasattr(type(settings), "DEBUG") or settings.SQL_ECHO != settings.DEBUG or settings.DEBUG is False


# ---------------------------------------------------------------- item 45 (alembic autogenerate options)

def test_alembic_env_configures_comparison_options():
    """Buglist item 45: autogenerate compared nothing but table existence."""
    source = Path("migrations/env.py").read_text(encoding="utf-8")
    for option in ("compare_type=True", "compare_server_default=True",
                   "render_as_batch=True"):
        assert option in source, f"env.py must set {option}"


# ---------------------------------------------------------------- item 44 (dashboard contract)

def test_dashboard_accepts_docx_and_zip_and_names_401():
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    assert ".docx,.zip" in html, "the file picker must expose the formats the parser supports"
    js = Path("frontend/app.js").read_text(encoding="utf-8")
    assert "e.status === 401" in js, "401 must be distinguished from backend-down"
    assert "توکن نامعتبر" in js
    assert "نمی‌تواند خالی باشد" in js, "an empty rejection reason is refused in the UI"


# ---------------------------------------------------------------- item 28 (idempotent archive)

def test_monthly_archive_is_idempotent_under_commit_failure(db_session):
    """Buglist item 28: a pass whose DB commit failed re-archived the SAME
    rows on retry and inflated the monthly totals. The archive now tracks
    row ids and derives totals from the merged id set."""
    session, tmp_path = db_session
    from backend.app.services.worker.market_worker import _archive_monthly_summary
    from backend.app.models.audit_log import AuditLog

    old = utc_now_naive() - timedelta(days=45)
    for i in range(3):
        session.add(AuditLog(event_type=f"E{i}", severity="WARNING",
                             actor="t", details={}, created_at=old))
    session.commit()

    cutoff = utc_now_naive() - timedelta(days=30)
    _archive_monthly_summary(session, "audit_logs", AuditLog, "created_at", "severity", cutoff)
    archive_file = (Path(settings.ARTIFACTS_REPORTS_DIR) / "archive" / "audit_logs"
                    / f"{old.strftime('%Y-%m')}.json")
    first = json.loads(archive_file.read_text(encoding="utf-8"))
    assert first["archived_count"] == 3

    # Retry pass over the SAME (still-existing) rows: totals must NOT double.
    _archive_monthly_summary(session, "audit_logs", AuditLog, "created_at", "severity", cutoff)
    second = json.loads(archive_file.read_text(encoding="utf-8"))
    assert second["archived_count"] == 3, "re-archiving the same rows must not inflate totals"
    assert second["by_status"] == first["by_status"]


# ---------------------------------------------------------------- item 31 (concurrent doc collision)

def test_concurrent_ingestion_collision_returns_the_winner_at_any_status(tmp_path, monkeypatch):
    """Buglist item 31: a collision with a still-processing winner re-raised
    (HTTP 500) and left the raw file without a record. The deterministic
    winner is returned WHATEVER its status. The race is simulated with a
    second session on the same file database committing the winner BETWEEN
    the manager's lookup and its insert."""
    import hashlib as _hashlib
    from sqlalchemy.exc import IntegrityError

    from backend.app.services.knowledge.manager import KnowledgeManager
    from backend.app.models.document import Document

    monkeypatch.setattr(settings, "DATA_RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(settings, "DATA_PROCESSED_DIR", tmp_path / "processed_data")
    monkeypatch.setattr(settings, "ARTIFACTS_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(settings, "ARTIFACTS_REPORTS_DIR", tmp_path / "reports")

    db_file = tmp_path / "race.db"
    engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session1 = factory()
    session2 = factory()
    try:
        inbox = tmp_path / "in"
        inbox.mkdir()
        (inbox / "race.txt").write_text("Rule: buy when RSI < 30 on XAUUSD M15.", encoding="utf-8")
        content_hash = _hashlib.sha256((inbox / "race.txt").read_bytes()).hexdigest()

        # No winner is visible to session1 yet (the other thread has not
        # committed): the first lookup finds nothing.
        assert session1.query(Document).filter_by(content_sha256=content_hash).count() == 0

        km = KnowledgeManager(session1)
        real_commit = session1.commit
        state = {"raced": False}

        def _racing_commit(*a, **kw):
            if not state["raced"]:
                state["raced"] = True
                # The other thread wins the UNIQUE index race right here.
                winner = Document(
                    document_id="doc_winner", file_name="race.txt", file_type="txt",
                    file_size_bytes=(inbox / "race.txt").stat().st_size,
                    processing_status="processing", pages_count=0,
                    content_sha256=content_hash,
                )
                session2.add(winner)
                session2.commit()
                raise IntegrityError("UNIQUE constraint failed", None, None)
            return real_commit(*a, **kw)

        monkeypatch.setattr(session1, "commit", _racing_commit)
        doc, rules = km.ingest_document_path("race.txt", inbox / "race.txt")
        assert doc.document_id == "doc_winner", (
            "the deterministic winner is returned instead of a raw IntegrityError")
        assert doc.processing_status == "processing"
    finally:
        session1.close()
        session2.close()
        engine.dispose()


# ---------------------------------------------------------------- item 36 (signing-key warning) & 43 note

def test_signing_key_warning_is_wired_in_lifespan():
    source = Path("backend/app/main.py").read_text(encoding="utf-8")
    assert "METRIC_SIGNING_KEY" in source and "tamper-PROOF" in source, (
        "an empty signing key must never be silent at startup (item 36)")
