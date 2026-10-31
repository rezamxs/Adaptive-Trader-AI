"""TDD tests for audit fix H4: schema drift must live in the Alembic chain.

Covers:
1. The boot patches (worker_cycle_records.lifecycle_phase/updated_at and
   ml_models.validation_metrics) are real Alembic revisions — upgrade head
   applies them, downgrade -1 removes them, upgrade restores them.
2. The baseline revision has a real downgrade (drop_all), so
   upgrade -> downgrade base -> upgrade works end to end.
3. init_db.py no longer ALTERs anything (no ad-hoc patch SQL left behind).
"""
import re
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from alembic import command
from alembic.config import Config

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def alembic_cfg(tmp_path, monkeypatch):
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    db_path = tmp_path / "migration_cycle.db"
    url = f"sqlite:///{db_path.as_posix()}"
    # env.py reads ALEMBIC_DATABASE_URL first; settings.DATABASE_URL must never
    # be reachable from these tests.
    monkeypatch.setenv("ALEMBIC_DATABASE_URL", url)
    return cfg, url, db_path


def _tables(engine):
    return set(inspect(engine).get_table_names())


def _columns(engine, table):
    return {c["name"] for c in inspect(engine).get_columns(table)}


def test_upgrade_downgrade_upgrade_cycle(alembic_cfg, tmp_path, monkeypatch):
    cfg, url, db_path = alembic_cfg

    # --- upgrade head -------------------------------------------------------
    command.upgrade(cfg, "head")
    engine = create_engine(url)
    try:
        expected = {
            "documents", "knowledge_chunks", "trading_rules", "dataset_versions",
            "ml_models", "backtest_runs", "trade_signals", "demo_orders",
            "emergency_stop_states", "audit_logs", "worker_cycle_records",
            "alembic_version",
        }
        assert expected <= _tables(engine)
        assert {"lifecycle_phase", "updated_at"} <= _columns(engine, "worker_cycle_records")
        assert "validation_metrics" in _columns(engine, "ml_models")
        assert "content_sha256" in _columns(engine, "documents")
    finally:
        engine.dispose()

    # --- downgrade to baseline (real drop_all, explicit destructive opt-in) --
    monkeypatch.setenv("ALEMBIC_ALLOW_DESTRUCTIVE_DOWNGRADE", "yes")
    command.downgrade(cfg, "base")
    engine = create_engine(url)
    try:
        assert not (_tables(engine) & {"documents", "ml_models", "worker_cycle_records"})
    finally:
        engine.dispose()

    # --- upgrade again -------------------------------------------------------
    command.upgrade(cfg, "head")
    engine = create_engine(url)
    try:
        assert "validation_metrics" in _columns(engine, "ml_models")
        assert "lifecycle_phase" in _columns(engine, "worker_cycle_records")
    finally:
        engine.dispose()


def test_initp_downgrade_removes_boot_patch_columns(alembic_cfg):
    cfg, url, db_path = alembic_cfg
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "20261015_sha256")  # reverts 20261024_mdig and 20261024_initp
    engine = create_engine(url)
    try:
        wc = _columns(engine, "worker_cycle_records")
        assert "lifecycle_phase" not in wc
        assert "updated_at" not in wc
        assert "validation_metrics" not in _columns(engine, "ml_models")
        # sha256 revision's column must survive the initp downgrade
        assert "content_sha256" in _columns(engine, "documents")
    finally:
        engine.dispose()
    command.upgrade(cfg, "head")


def test_init_db_contains_no_adhoc_alter_patches():
    src = (PROJECT_ROOT / "backend" / "app" / "db" / "init_db.py").read_text(encoding="utf-8")
    assert "ALTER TABLE" not in src, "init_db must not carry ad-hoc ALTER patches (H4: Alembic owns drift)"
    assert not re.search(r"\btext\(\s*[\"']", src), "init_db must not execute ad-hoc SQL text"


def test_init_db_is_a_noop_on_an_existing_populated_database(tmp_path, monkeypatch):
    """M20 behavioral proof (the H4 static check above is source-level):
    running the real init_db twice against a populated tmp database must not
    change schema or data — startup can never mutate an existing DB."""
    from sqlalchemy import create_engine, inspect
    from sqlalchemy.orm import sessionmaker
    from backend.app.db import init_db as init_db_module
    from backend.app.models.audit_log import AuditLog

    db_path = tmp_path / "existing.db"
    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    # Bind the real init_db to the tmp engine — the module-level import binds
    # the production engine at import time and must never be touched here.
    monkeypatch.setattr(init_db_module, "engine", engine)
    monkeypatch.setattr(init_db_module, "SessionLocal", sessionmaker(bind=engine))

    try:
        # First boot creates the schema and the guarded safety seed.
        init_db_module.init_db()
        Session = sessionmaker(bind=engine)
        with Session() as session:
            session.add(AuditLog(event_type="M20_PROBE", severity="INFO",
                                 details={"probe": "sentinel"}))
            session.commit()

        def snapshot():
            insp = inspect(engine)
            tables = {}
            with engine.connect() as conn:
                for table in sorted(insp.get_table_names()):
                    cols = tuple(c["name"] for c in insp.get_columns(table))
                    rows = conn.exec_driver_sql(f'SELECT * FROM "{table}"').fetchall()
                    tables[table] = (cols, len(rows))
            return tables

        before = snapshot()

        # Second boot on the populated DB: must be a create-all no-op.
        init_db_module.init_db()

        after = snapshot()
        assert after == before, (
            "init_db on an existing database must not change schema or row counts")
        with Session() as session:
            probe = session.query(AuditLog).filter_by(event_type="M20_PROBE").first()
            assert probe is not None, "existing data must survive a re-boot"
    finally:
        engine.dispose()
