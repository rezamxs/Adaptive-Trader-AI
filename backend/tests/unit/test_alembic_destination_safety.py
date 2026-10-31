"""Alembic destination safety (follow-up to item 21 / H4-incident class).

A stray `alembic upgrade head` in the wrong working directory once reached
the production database because env.py silently fell back to the application
settings URL. Now migrations run ONLY against an explicitly pinned
ALEMBIC_DATABASE_URL; without it, env.py raises a clear operator error and
nothing is migrated.
"""
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def alembic_cfg():
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    return cfg


def test_migration_fails_closed_without_explicit_destination(alembic_cfg, monkeypatch):
    """Negative test: with NO ALEMBIC_DATABASE_URL, migration must abort with
    a clear error naming the variable — never fall back to the app URL."""
    monkeypatch.delenv("ALEMBIC_DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="ALEMBIC_DATABASE_URL"):
        command.upgrade(alembic_cfg, "head")


def test_missing_destination_error_hides_database_url(alembic_cfg, monkeypatch):
    """The fail-closed error must name ALEMBIC_DATABASE_URL and how to pin the
    destination, but never echo the application settings URL — connection
    strings carry credentials that would otherwise land in logs/tracebacks."""
    from backend.app.core.config import settings

    secret_url = (
        "postgresql://ops_user:S3cr3t-Pa55w0rd@prod-db.internal:5432/adaptive_trader"
    )
    monkeypatch.setattr(settings, "DATABASE_URL", secret_url)
    monkeypatch.delenv("ALEMBIC_DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        command.upgrade(alembic_cfg, "head")
    message = str(excinfo.value)
    assert "ALEMBIC_DATABASE_URL" in message
    for fragment in (secret_url, "S3cr3t-Pa55w0rd", "prod-db.internal", "ops_user"):
        assert fragment not in message


def test_migration_runs_on_explicit_tmp_url(alembic_cfg, tmp_path, monkeypatch):
    """Positive test: an explicitly pinned tmp URL migrates cleanly.

    Regression guard (2026-10-31): alembic.ini's version_locations was
    CWD-relative, so running `upgrade head` from any directory other than
    the repo root silently resolved ZERO revision files and "succeeded"
    without executing a single migration (only an empty alembic_version
    table). Asserting the stamped head revision alongside the tables makes
    that silent-empty failure mode impossible to reintroduce."""
    url = f"sqlite:///{(tmp_path / 'explicit_dest.db').as_posix()}"
    monkeypatch.setenv("ALEMBIC_DATABASE_URL", url)
    command.upgrade(alembic_cfg, "head")

    tables = set(inspect(create_engine(url)).get_table_names())
    assert {"worker_cycle_records", "documents", "ml_models"} <= tables

    from alembic.script import ScriptDirectory

    head = ScriptDirectory.from_config(alembic_cfg).get_current_head()
    engine = create_engine(url)
    try:
        import sqlalchemy as sa

        with engine.connect() as conn:
            stamped = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
    finally:
        engine.dispose()
    assert stamped == head, (
        f"alembic_version stamped {stamped!r} but chain head is {head!r} — "
        "upgrade must never report success without running the chain"
    )


def test_destructive_base_downgrade_requires_explicit_confirmation(alembic_cfg, tmp_path, monkeypatch):
    """M21: `downgrade base` executes the baseline downgrade, which is
    Base.metadata.drop_all — every table with all data. It must refuse to run
    without an explicit per-invocation opt-in, and must work on a disposable
    tmp database once the operator sets the opt-in env var."""
    url = f"sqlite:///{(tmp_path / 'downgrade_guard.db').as_posix()}"
    monkeypatch.setenv("ALEMBIC_DATABASE_URL", url)
    command.upgrade(alembic_cfg, "head")

    monkeypatch.delenv("ALEMBIC_ALLOW_DESTRUCTIVE_DOWNGRADE", raising=False)
    with pytest.raises(RuntimeError, match="ALEMBIC_ALLOW_DESTRUCTIVE_DOWNGRADE"):
        command.downgrade(alembic_cfg, "base")
    # The refusal must be clean: schema untouched, version still at head.
    engine = create_engine(url)
    try:
        assert "worker_cycle_records" in set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    monkeypatch.setenv("ALEMBIC_ALLOW_DESTRUCTIVE_DOWNGRADE", "yes")
    command.downgrade(alembic_cfg, "base")
    engine = create_engine(url)
    try:
        assert "worker_cycle_records" not in set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
