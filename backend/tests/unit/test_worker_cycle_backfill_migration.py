"""Regression for the wcbf revision: it backfills legacy NULL updated_at
values from created_at and enforces the model's NOT NULL contract on
worker_cycle_records. Runs ONLY on an isolated tmp database through Alembic
with ALEMBIC_DATABASE_URL — the production DB is never touched, connected to,
or consulted. Checking real-database state is a manual operator step
(recorded in docs/PROJECT_MEMORY.md), never a condition for this suite to
pass or skip.
"""
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def alembic_cfg(tmp_path, monkeypatch):
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    db_path = tmp_path / "backfill_cycle.db"
    url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("ALEMBIC_DATABASE_URL", url)
    return cfg, url, db_path


def _connect(url):
    return create_engine(url)


LEGACY_DDL = """
CREATE TABLE worker_cycle_records (
    id INTEGER NOT NULL,
    cycle_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    bar_key VARCHAR(80),
    bar_time DATETIME,
    status_code VARCHAR(50) NOT NULL,
    decision VARCHAR(20),
    signal_id VARCHAR(50),
    duration_ms FLOAT NOT NULL,
    details TEXT,
    error_message TEXT,
    created_at DATETIME NOT NULL,
    lifecycle_phase VARCHAR(30) DEFAULT 'COMPLETED',
    updated_at DATETIME,
    PRIMARY KEY (id)
)
"""


def test_wcbf_backfills_and_enforces_not_null(alembic_cfg):
    cfg, url, db_path = alembic_cfg

    # 1. Build the chain to the PREVIOUS head, then replace the freshly-created
    # table with the LEGACY prod shape (nullable lifecycle_phase/updated_at) —
    # exactly the drift observed on the production DB.
    command.upgrade(cfg, "20261024_mdig")
    engine = _connect(url)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE worker_cycle_records"))
        conn.execute(text(LEGACY_DDL))
        conn.execute(text(
            "INSERT INTO worker_cycle_records "
            "(cycle_id, symbol, timeframe, status_code, created_at, updated_at, duration_ms) "
            "VALUES ('cyc_legacy_1', 'XAUUSD', 'M1', 'PROCESSED', '2026-10-01 10:00:00', NULL, 100.0)"
        ))
        conn.execute(text(
            "INSERT INTO worker_cycle_records "
            "(cycle_id, symbol, timeframe, status_code, created_at, updated_at, duration_ms) "
            "VALUES ('cyc_legacy_2', 'XAUUSD', 'M1', 'PROCESSED', '2026-10-01 11:00:00', NULL, 120.0)"
        ))
        # A modern row WITH updated_at must survive the backfill untouched.
        conn.execute(text(
            "INSERT INTO worker_cycle_records "
            "(cycle_id, symbol, timeframe, status_code, created_at, updated_at, duration_ms) "
            "VALUES ('cyc_modern', 'XAUUSD', 'M1', 'PROCESSED', "
            "'2026-10-02 12:00:00', '2026-10-02 12:00:05', 140.0)"
        ))
    engine.dispose()

    # 2. Upgrade to the new head.
    command.upgrade(cfg, "head")

    # 3. Verify: no NULLs remain, legacy rows carry their created_at, the
    # modern row keeps its own value, and the columns enforce NOT NULL.
    engine = _connect(url)
    inspector = inspect(engine)
    cols = {c["name"]: c for c in inspector.get_columns("worker_cycle_records")}
    assert cols["updated_at"]["nullable"] is False, "updated_at must be NOT NULL after wcbf"
    assert cols["lifecycle_phase"]["nullable"] is False, "lifecycle_phase must be NOT NULL after wcbf"

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT cycle_id, updated_at, created_at FROM worker_cycle_records ORDER BY cycle_id"
        )).fetchall()
        null_count = conn.execute(text(
            "SELECT COUNT(*) FROM worker_cycle_records WHERE updated_at IS NULL"
        )).scalar()
    assert null_count == 0
    by_id = {r[0]: (r[1], r[2]) for r in rows}
    assert by_id["cyc_legacy_1"][0] == by_id["cyc_legacy_1"][1], (
        "legacy NULL must be backfilled from created_at")
    assert by_id["cyc_modern"][0] == "2026-10-02 12:00:05", (
        "existing updated_at values must be preserved")
    engine.dispose()

    # 4. Downgrade restores the legacy nullability but PRESERVES the row data
    # (item fix: the old downgrade nulled updated_at, destroying the
    # backfilled creation times). Then upgrade re-applies NOT NULL with data
    # still intact. The destination is pinned explicitly: a relative "-1"
    # silently changed meaning when the M8 revision (20261031_mlwfm) was
    # appended on top of wcbf.
    command.downgrade(cfg, "20261024_mdig")
    engine = _connect(url)
    cols = {c["name"]: c for c in inspect(engine).get_columns("worker_cycle_records")}
    assert cols["updated_at"]["nullable"] is True
    with engine.connect() as conn:
        down_rows = conn.execute(text(
            "SELECT cycle_id, updated_at FROM worker_cycle_records ORDER BY cycle_id"
        )).fetchall()
    down_by_id = {r[0]: r[1] for r in down_rows}
    assert down_by_id["cyc_legacy_1"] == by_id["cyc_legacy_1"][0], (
        "downgrade must not wipe the backfilled updated_at data")
    assert down_by_id["cyc_modern"] == by_id["cyc_modern"][0], (
        "downgrade must not wipe real updated_at values")
    engine.dispose()

    command.upgrade(cfg, "head")
    engine = _connect(url)
    cols = {c["name"]: c for c in inspect(engine).get_columns("worker_cycle_records")}
    assert cols["updated_at"]["nullable"] is False
    with engine.connect() as conn:
        re_rows = conn.execute(text(
            "SELECT cycle_id, updated_at FROM worker_cycle_records ORDER BY cycle_id"
        )).fetchall()
    re_by_id = {r[0]: r[1] for r in re_rows}
    assert re_by_id == down_by_id, "re-upgrade must preserve all updated_at values"
    engine.dispose()
