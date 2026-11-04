"""M8 migration: ml_walk_forward_metrics column on ml_models.

Covers the full upgrade path on a POPULATED legacy database (schema as of
the previous head 20261029_wcbf): the column is added, legacy rows are
backfilled with the '{}' default, existing data survives, the head is
stamped, re-running the upgrade is a no-op, and the revision is reversible.
The legacy schema is produced by dropping the column the current ORM
baseline creates (create_all reflects current metadata, so a true legacy
schema is simulated by removing the new column).
"""
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

PROJECT_ROOT = None  # resolved in fixture


@pytest.fixture
def alembic_cfg(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "migrations"))
    return cfg


@pytest.fixture
def legacy_db_url(tmp_path, alembic_cfg, monkeypatch):
    """A populated database at the PREVIOUS head, without the new column."""
    url = f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}"
    monkeypatch.setenv("ALEMBIC_DATABASE_URL", url)
    command.upgrade(alembic_cfg, "20261029_wcbf")

    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE ml_models DROP COLUMN ml_walk_forward_metrics"))
        conn.execute(text("UPDATE alembic_version SET version_num='20261029_wcbf'"))
        conn.execute(text(
            "INSERT INTO ml_models (model_id, dataset_version_tag, algorithm, "
            "hyperparameters, features_used, target_name, train_metrics, "
            "validation_metrics, test_metrics, calibration_metrics, "
            "walk_forward_metrics, status, artifact_path, trained_at, updated_at) "
            "VALUES ('m_upg', 'ds_upg', 'logistic_regression', '{}', '[]', 't', "
            "'{}', '{}', '{}', '{}', '{}', 'candidate', 'x.joblib', "
            "'2026-10-30 00:00:00', '2026-10-30 00:00:00')"))
        conn.execute(text(
            "INSERT INTO documents (document_id, file_name, file_type, "
            "file_size_bytes, source_description, processing_status, pages_count, created_at) "
            "VALUES ('doc_upg', 'f.txt', 'txt', 10, 'd', 'completed', 1, '2026-10-30 00:00:00')"))
    engine.dispose()
    return url


def test_upgrade_from_previous_head_adds_column_and_preserves_data(legacy_db_url, alembic_cfg):
    from alembic.script import ScriptDirectory

    command.upgrade(alembic_cfg, "head")

    engine = create_engine(legacy_db_url)
    try:
        columns = {c["name"] for c in inspect(engine).get_columns("ml_models")}
        assert "ml_walk_forward_metrics" in columns
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT model_id, ml_walk_forward_metrics FROM ml_models")).fetchall()
            docs = conn.execute(text("SELECT document_id FROM documents")).fetchall()
            stamped = conn.execute(text(
                "SELECT version_num FROM alembic_version")).scalar()
        assert rows == [("m_upg", "{}")], "legacy rows must be backfilled, not nulled or lost"
        assert docs == [("doc_upg",)], "unrelated tables must be untouched"
        assert stamped == ScriptDirectory.from_config(alembic_cfg).get_current_head(), (
            "the stamped version must equal the actual chain head")
    finally:
        engine.dispose()


def test_upgrade_is_idempotent_and_reversible(legacy_db_url, alembic_cfg):
    command.upgrade(alembic_cfg, "head")
    command.upgrade(alembic_cfg, "head")  # no-op at head

    command.downgrade(alembic_cfg, "20261029_wcbf")
    engine = create_engine(legacy_db_url)
    try:
        columns = {c["name"] for c in inspect(engine).get_columns("ml_models")}
        assert "ml_walk_forward_metrics" not in columns
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT model_id FROM ml_models")).fetchall()
        assert rows == [("m_upg",)], "downgrade must not lose data either"
    finally:
        engine.dispose()

    command.upgrade(alembic_cfg, "head")
    engine = create_engine(legacy_db_url)
    try:
        columns = {c["name"] for c in inspect(engine).get_columns("ml_models")}
        assert "ml_walk_forward_metrics" in columns
    finally:
        engine.dispose()
