"""Step 3.7: a monthly summary is archived BEFORE retention pruning deletes rows.

Owner decision 2026-11-01: 30-day retention windows; step 3.7 requires the
monthly summary to survive before older data is removed. The pruning pass is
fail-safe: if the archive write fails, pruning for that table is skipped this
pass instead of deleting unarchived rows.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.app.models  # noqa: F401 - register all ORM tables on Base
from backend.app.core.config import settings
from backend.app.db.base import Base, utc_now_naive
from backend.app.models.audit_log import AuditLog
from backend.app.models.worker_cycle import WorkerCycleRecord
from backend.app.services.worker import market_worker
from backend.app.services.worker.market_worker import prune_operational_records


@pytest.fixture
def db_session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ARTIFACTS_REPORTS_DIR", tmp_path)
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session, tmp_path
    session.close()
    engine.dispose()


def _add_cycle(session, cycle_id: str, age_days: int, status: str = "COMPLETED"):
    session.add(WorkerCycleRecord(
        cycle_id=cycle_id, symbol="XAUUSD", timeframe="M1",
        status_code=status,
        created_at=utc_now_naive() - timedelta(days=age_days),
    ))


def test_archive_written_before_rows_deleted(db_session):
    session, reports_dir = db_session
    # Two old months + one fresh row that must survive.
    _add_cycle(session, "cyc_sep", 60)   # ~2 months back
    _add_cycle(session, "cyc_sep2", 55)
    _add_cycle(session, "cyc_fresh", 1)
    session.commit()

    deleted = prune_operational_records(session, worker_cycle_days=30)

    assert deleted["worker_cycle_records"] == 2
    assert session.query(WorkerCycleRecord).count() == 1

    archive_dir = Path(reports_dir) / "archive" / "worker_cycle_records"
    files = sorted(p.name for p in archive_dir.glob("*.json"))
    assert len(files) >= 1, "at least one monthly summary must exist after pruning"
    # The fresh row's month must NOT appear (nothing about it was deleted).
    fresh_month = (utc_now_naive() - timedelta(days=1)).strftime("%Y-%m")
    assert f"{fresh_month}.json" not in files
    # Counts and statuses recorded.
    total_archived = sum(json.loads((archive_dir / f).read_text(encoding="utf-8"))["archived_count"] for f in files)
    assert total_archived == 2


def test_archive_merge_accumulates_across_prune_passes(db_session):
    session, reports_dir = db_session
    _add_cycle(session, "cyc_a", 40)
    session.commit()
    prune_operational_records(session, worker_cycle_days=30)
    prune_operational_records(session, worker_cycle_days=30)  # second pass: nothing old left
    archive_dir = Path(reports_dir) / "archive" / "worker_cycle_records"
    files = list(archive_dir.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["archived_count"] == 1


def test_archive_failure_skips_pruning_fail_safe(db_session, monkeypatch):
    session, reports_dir = db_session
    _add_cycle(session, "cyc_keep", 60)
    session.commit()

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(market_worker, "_archive_monthly_summary", _boom)
    deleted = prune_operational_records(session, worker_cycle_days=30)
    assert deleted["worker_cycle_records"] == 0, (
        "rows must never be pruned when their monthly summary could not be written")
    assert session.query(WorkerCycleRecord).count() == 1


def test_disabled_window_archives_and_deletes_nothing(db_session):
    session, reports_dir = db_session
    _add_cycle(session, "cyc_stay", 400)
    session.commit()
    deleted = prune_operational_records(session, worker_cycle_days=0)
    assert deleted == {"worker_cycle_records": 0, "audit_logs": 0, "trade_signals": 0, "demo_orders": 0}
    assert session.query(WorkerCycleRecord).count() == 1
    assert not (Path(reports_dir) / "archive").exists()


def test_by_status_counts_recorded(db_session):
    session, reports_dir = db_session
    _add_cycle(session, "cyc_1", 60, status="COMPLETED")
    _add_cycle(session, "cyc_2", 61, status="RESERVATION_FAILED")
    session.commit()
    prune_operational_records(session, worker_cycle_days=30)
    archive_dir = Path(reports_dir) / "archive" / "worker_cycle_records"
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in archive_dir.glob("*.json")]
    by_status = {}
    for p in payloads:
        for k, v in p["by_status"].items():
            by_status[k] = by_status.get(k, 0) + v
    assert by_status == {"COMPLETED": 1, "RESERVATION_FAILED": 1}
