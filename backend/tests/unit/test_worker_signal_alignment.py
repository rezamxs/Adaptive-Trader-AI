"""Item 40 regressions: (a) the worker passes its precomputed feature frame so
a mid-cycle bar rollover cannot record bar N+1's signal under cycle N; (b) a
configurable, safe retention mechanism for operational tables.
"""
from datetime import datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.core.config import settings
from backend.app.db.base import Base
import backend.app.models  # noqa: F401 - register all tables
from backend.app.models.rule import TradingRule
from backend.app.models.worker_cycle import WorkerCycleRecord
from backend.app.models.audit_log import AuditLog
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.market_data.mock_provider import MockMarketDataProvider
from backend.app.services.signal.manager import SignalManager
from backend.app.services.worker.market_worker import prune_operational_records


# ---------------------------------------------------------------------------
# (a) signal pinned to the reserved bar
# ---------------------------------------------------------------------------

class _RolloverProvider(MockMarketDataProvider):
    """Returns a frame ending at bar N on the FIRST call and bar N+1 on every
    LATER call — simulating an M1 close between the worker's fetch and the
    manager's (previously unconditional) refetch."""

    def __init__(self):
        super().__init__(is_demo_account=True)
        self.calls = 0
        base = super().get_historical_dataframe("EURUSD", TimeFrame.M15, count=120)
        self._frame_n = base
        # Frame N+1 = one EXTRA bar closing after N (a real M1 rollover adds a
        # row, not just a price change) — otherwise the bar TIME never shifts.
        next_time = base.index[-1] + pd.Timedelta(minutes=15)
        prev_close = base["close"].iloc[-1]
        extra = pd.DataFrame({
            "open": [prev_close],
            "high": [prev_close + 0.0008],
            "low": [prev_close - 0.0008],
            "close": [prev_close + 0.0004],
            "tick_volume": [50],
            "spread": [20],
        }, index=[next_time])
        self._frame_n1 = pd.concat([base, extra])
        self._frame_n1.index.name = base.index.name

    def get_historical_dataframe(self, symbol, timeframe, count=500, start_pos=0):
        self.calls += 1
        return self._frame_n1 if self.calls >= 2 else self._frame_n


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def approved_rule(db_session):
    rule = TradingRule(
        rule_id="rule_rsi_oversold_align",
        name="Buy on RSI Oversold",
        category="entry",
        description="Buy when RSI is below 70 (alignment test)",
        conditions=[{"indicator": "RSI", "operator": "<", "threshold": 70.0}],
        parameters={},
        symbol_scope=["EURUSD"],
        timeframe_scope=["M15"],
        invalidation_conditions=[],
        source_pages=[],
        assumptions=[],
        ambiguities=[],
        approval_status="approved",
    )
    db_session.add(rule)
    db_session.commit()
    return rule


def test_precomputed_features_pin_signal_to_reserved_bar(db_session, approved_rule, monkeypatch):
    monkeypatch.setattr(settings, "ENFORCE_SESSION_FILTER", False)
    provider = _RolloverProvider()
    manager = SignalManager(db_session, provider)

    # The "worker" computes features from the FIRST fetch (bar N reserved).
    worker_frame = provider.get_historical_dataframe("EURUSD", TimeFrame.M15, count=120)
    from backend.app.services.dataset.features import FeatureEngineer
    df_feats, _ = FeatureEngineer.transform(worker_frame)
    reserved_bar_time = df_feats.index[-1]

    sig = manager.process_latest_bar(
        "EURUSD", TimeFrame.M15, precomputed_features=df_feats)

    assert sig is not None
    sig_bar = sig.bar_time.replace(tzinfo=None) if sig.bar_time.tzinfo else sig.bar_time
    reserved = reserved_bar_time.to_pydatetime() if hasattr(reserved_bar_time, "to_pydatetime") else reserved_bar_time
    reserved = reserved.replace(tzinfo=None)
    assert sig_bar == reserved, (
        f"signal must be pinned to the reserved bar N, recorded {sig_bar} vs reserved {reserved}")


def test_refetch_without_precomputed_shifts_bar_then_flagged(db_session, approved_rule):
    """Documents the race: WITHOUT precomputed features the manager's refetch
    sees bar N+1. The worker path (previous test) is the fix; this test pins
    the old behavior so the difference is explicit."""
    provider = _RolloverProvider()
    manager = SignalManager(db_session, provider)
    worker_frame = provider.get_historical_dataframe("EURUSD", TimeFrame.M15, count=120)
    from backend.app.services.dataset.features import FeatureEngineer
    df_feats, _ = FeatureEngineer.transform(worker_frame)
    reserved_bar_time = df_feats.index[-1]

    sig = manager.process_latest_bar("EURUSD", TimeFrame.M15)

    sig_bar = sig.bar_time.replace(tzinfo=None) if sig.bar_time.tzinfo else sig.bar_time
    reserved = reserved_bar_time.to_pydatetime() if hasattr(reserved_bar_time, "to_pydatetime") else reserved_bar_time
    reserved = reserved.replace(tzinfo=None)
    assert sig_bar != reserved, (
        "without precomputed features the refetch race must be observable")


def test_worker_passes_precomputed_features_to_manager(db_session, approved_rule, monkeypatch):
    """Wiring proof: the worker cycle calls process_latest_bar WITH its
    precomputed feature frame (item 40 race cannot reappear silently)."""
    # The worker runs XAUUSD M1 — the rule scope must match the worker symbol.
    rule = TradingRule(
        rule_id="rule_rsi_oversold_worker",
        name="Buy on RSI Oversold",
        category="entry",
        description="Buy when RSI is below 70 (worker alignment test)",
        conditions=[{"indicator": "RSI", "operator": "<", "threshold": 99.0}],
        parameters={},
        symbol_scope=["XAUUSD"],
        timeframe_scope=["M1"],
        invalidation_conditions=[],
        source_pages=[],
        assumptions=[],
        ambiguities=[],
        approval_status="approved",
    )
    db_session.add(rule)
    db_session.commit()

    captured = {}

    class _SpyManager(SignalManager):
        def process_latest_bar(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            captured["args"] = args
            return None  # short-circuit; we only verify the wiring

    monkeypatch.setattr(
        "backend.app.services.worker.market_worker.SignalManager", _SpyManager)

    from backend.app.services.worker.market_worker import MarketMonitoringWorker
    provider = _RolloverProvider()
    worker = MarketMonitoringWorker(data_provider=provider, db_factory=lambda: db_session)
    result = worker.run_single_cycle()
    print("cycle result:", result)

    assert "kwargs" in captured, (
        f"worker must invoke SignalManager.process_latest_bar; cycle={result}")
    assert captured["kwargs"].get("precomputed_features") is not None, (
        "worker must pass its precomputed feature frame to the signal manager")


# ---------------------------------------------------------------------------
# (b) retention: configurable, explicit, default-off; nothing is deleted
# without the operator setting a retention window.
# ---------------------------------------------------------------------------

def test_retention_disabled_by_default_deletes_nothing(db_session):
    for i in range(5):
        db_session.add(WorkerCycleRecord(
            cycle_id=f"cyc_ret_{i}", symbol="XAUUSD", timeframe="M1",
            status_code="PROCESSED",
            created_at=datetime.utcnow() - timedelta(days=400),
        ))
        db_session.add(AuditLog(
            event_type="RETENTION_TEST", severity="INFO", actor="TEST",
            details={}, created_at=datetime.utcnow() - timedelta(days=400),
        ))
    db_session.commit()

    deleted = prune_operational_records(db_session, worker_cycle_days=0, audit_log_days=0)

    assert deleted == {"worker_cycle_records": 0, "audit_logs": 0}
    assert db_session.query(WorkerCycleRecord).count() == 5
    assert db_session.query(AuditLog).count() == 5


def test_retention_prunes_only_records_older_than_window(db_session):
    now = datetime.utcnow()
    for i in range(3):
        db_session.add(WorkerCycleRecord(
            cycle_id=f"cyc_old_{i}", symbol="XAUUSD", timeframe="M1",
            status_code="PROCESSED", created_at=now - timedelta(days=40),
        ))
        db_session.add(AuditLog(
            event_type="OLD_AUDIT", severity="INFO", actor="TEST",
            details={}, created_at=now - timedelta(days=40),
        ))
    for i in range(2):
        db_session.add(WorkerCycleRecord(
            cycle_id=f"cyc_new_{i}", symbol="XAUUSD", timeframe="M1",
            status_code="PROCESSED", created_at=now - timedelta(days=1),
        ))
        db_session.add(AuditLog(
            event_type="NEW_AUDIT", severity="INFO", actor="TEST",
            details={}, created_at=now - timedelta(days=1),
        ))
    db_session.commit()

    deleted = prune_operational_records(db_session, worker_cycle_days=30, audit_log_days=30)

    assert deleted == {"worker_cycle_records": 3, "audit_logs": 3}
    assert db_session.query(WorkerCycleRecord).count() == 2
    assert db_session.query(AuditLog).count() == 2
    remaining = [r.cycle_id for r in db_session.query(WorkerCycleRecord).all()]
    assert all(cid.startswith("cyc_new_") for cid in remaining)


# ---------------------------------------------------------------------------
# Item 16 wiring: the configured retention actually runs at app startup.
# ---------------------------------------------------------------------------

def test_lifespan_applies_configured_retention(monkeypatch):
    """The prune function existed with zero production callers (unbounded
    growth). The lifespan must apply the operator-configured windows at
    startup; with both windows at 0 nothing may be called."""
    import asyncio
    import contextlib

    import backend.app.main as main_module
    from backend.app.db import session as db_session_module
    from backend.app.services.worker import market_worker

    calls = []

    def fake_prune(db, worker_cycle_days=0, audit_log_days=0):
        calls.append((worker_cycle_days, audit_log_days))
        return {"worker_cycle_records": 0, "audit_logs": 0}

    @contextlib.contextmanager
    def fake_session():
        yield object()

    async def _run_lifespan():
        async with main_module.lifespan(object()):
            pass

    # Configured windows -> prune called once with exactly those windows.
    monkeypatch.setattr(settings, "WORKER_CYCLE_RETENTION_DAYS", 30)
    monkeypatch.setattr(settings, "AUDIT_LOG_RETENTION_DAYS", 90)
    monkeypatch.setattr(settings, "WORKER_ENABLED", False)
    monkeypatch.setattr(main_module, "init_db", lambda: None)
    monkeypatch.setattr(db_session_module, "SessionLocal", fake_session)
    monkeypatch.setattr(market_worker, "prune_operational_records", fake_prune)
    asyncio.run(_run_lifespan())
    assert calls == [(30, 90)], "startup must apply the configured retention windows"

    # Default (0/0) -> no prune call at all: audit history is never pruned
    # without an explicit operator window.
    calls.clear()
    monkeypatch.setattr(settings, "WORKER_CYCLE_RETENTION_DAYS", 0)
    monkeypatch.setattr(settings, "AUDIT_LOG_RETENTION_DAYS", 0)
    asyncio.run(_run_lifespan())
    assert calls == [], "default retention windows must disable pruning entirely"


# ---------------------------------------------------------------------------
# Item 8: the corpus index must not rebuild per request / readiness probe.
# ---------------------------------------------------------------------------

def test_retriever_index_is_lazy_and_amortized(db_session):
    """Item 8 measurement: constructing a per-request KnowledgeManager must
    query knowledge_chunks ZERO times (the old __init__ eagerly rebuilt the
    whole BM25 index), and repeated searches must reuse the built index."""
    from sqlalchemy import event

    from backend.app.services.knowledge.retriever import KnowledgeRetriever
    from backend.app.models.knowledge_chunk import KnowledgeChunk

    db_session.add(KnowledgeChunk(
        chunk_id="chk_item8", document_id="doc_item8",
        title="RSI rule", content="Rule: Buy when RSI < 30 on M15",
        key_concepts=["RSI"], source_file="item8.txt",
        location_type="page", location_reference="p.1",
        parameters={}, exceptions_and_limits=[],
    ))
    db_session.commit()

    queries = {"n": 0}

    def _count(conn, cursor, statement, parameters, context, executemany):
        if "knowledge_chunks" in statement:
            queries["n"] += 1

    event.listen(db_session.bind, "before_cursor_execute", _count)
    try:
        retriever = KnowledgeRetriever(db_session)
        assert queries["n"] == 0, (
            "constructing a retriever must not rebuild the corpus index (lazy build)")
        retriever.search("RSI")
        first = queries["n"]
        assert first >= 1, "the first search must build the index once"
        retriever.search("RSI buy")
        assert queries["n"] == first, "subsequent searches must reuse the cached index"
    finally:
        event.remove(db_session.bind, "before_cursor_execute", _count)


def test_trading_ready_probe_does_not_build_knowledge_index(db_session):
    """Item 8: the readiness probe (evaluated around every worker cycle) must
    query TradingRule directly — constructing KnowledgeManager there paid a
    full BM25 rebuild twice per cycle. The discriminator is behavioral: with
    an approved rule seeded and no active model, the probe must reach the
    rules query and report NO_ACTIVE_MODEL, not a KnowledgeManager failure."""
    from unittest.mock import MagicMock

    from backend.app.services.knowledge.manager import KnowledgeManager
    from backend.app.services.worker.market_worker import MarketMonitoringWorker

    def _boom(self, *a, **k):
        raise AssertionError("KnowledgeManager must not be constructed by the readiness probe")

    db_session.add(TradingRule(
        rule_id="rule_probe_ready", name="Probe rule", category="entry",
        description="d", conditions=[{"indicator": "RSI", "operator": "<", "threshold": 70.0}],
        parameters={}, approval_status="approved",
    ))
    db_session.commit()

    class _ReadyWorker(MarketMonitoringWorker):
        @property
        def service_ready(self):
            return True

        @property
        def market_data_ready(self):
            return True

    original_init = KnowledgeManager.__init__
    KnowledgeManager.__init__ = _boom
    try:
        worker = _ReadyWorker(data_provider=MagicMock(), db_factory=lambda: db_session)
        reason = worker.trading_readiness_reason
    finally:
        KnowledgeManager.__init__ = original_init

    assert reason == "NO_ACTIVE_MODEL", (
        f"probe must reach the direct rules query, got {reason!r}")
