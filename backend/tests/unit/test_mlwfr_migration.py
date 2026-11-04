"""M8 phase 2.4 / Minor-17 migration: ml_walk_forward_runs table + artifact_sha256 column.

Covers the full upgrade path on a POPULATED legacy database (schema as of the
previous head 20261102_sdir, simulated by dropping the new table and column
from the current ORM baseline), the head stamp, idempotent re-upgrade and
reversibility. No real database is touched — everything runs on tmp SQLite.
"""
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

PREVIOUS_HEAD = "20261102_sdir"


@pytest.fixture
def alembic_cfg(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "migrations"))
    return cfg


@pytest.fixture
def legacy_db_url(tmp_path, alembic_cfg, monkeypatch):
    """A populated database at the PREVIOUS head, without the new objects."""
    url = f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}"
    monkeypatch.setenv("ALEMBIC_DATABASE_URL", url)
    command.upgrade(alembic_cfg, PREVIOUS_HEAD)

    engine = create_engine(url)
    with engine.begin() as conn:
        # Simulate the legacy schema: the ORM baseline created both objects.
        conn.execute(text("DROP TABLE IF EXISTS ml_walk_forward_runs"))
        conn.execute(text("ALTER TABLE ml_models DROP COLUMN artifact_sha256"))
        conn.execute(text(f"UPDATE alembic_version SET version_num='{PREVIOUS_HEAD}'"))
        conn.execute(text(
            "INSERT INTO ml_models (model_id, dataset_version_tag, algorithm, "
            "hyperparameters, features_used, target_name, train_metrics, "
            "validation_metrics, test_metrics, calibration_metrics, "
            "walk_forward_metrics, ml_walk_forward_metrics, status, artifact_path, "
            "trained_at, updated_at) "
            "VALUES ('m_upg', 'ds_upg', 'logistic_regression', '{}', '[]', 't', "
            "'{}', '{}', '{}', '{}', '{}', '{}', 'candidate', 'x.joblib', "
            "'2026-11-02 00:00:00', '2026-11-02 00:00:00')"))
        conn.execute(text(
            "INSERT INTO documents (document_id, file_name, file_type, "
            "file_size_bytes, source_description, processing_status, pages_count, created_at) "
            "VALUES ('doc_upg', 'f.txt', 'txt', 10, 'd', 'completed', 1, '2026-11-02 00:00:00')"))
    engine.dispose()
    return url


def test_upgrade_creates_table_and_column_and_preserves_data(legacy_db_url, alembic_cfg):
    from alembic.script import ScriptDirectory

    command.upgrade(alembic_cfg, "head")

    engine = create_engine(legacy_db_url)
    try:
        inspector = inspect(engine)
        assert "ml_walk_forward_runs" in inspector.get_table_names()
        run_columns = {c["name"] for c in inspector.get_columns("ml_walk_forward_runs")}
        assert {"id", "run_id", "payload", "digest", "fold_count",
                "model_id", "created_at"} <= run_columns
        model_columns = {c["name"] for c in inspector.get_columns("ml_models")}
        assert "artifact_sha256" in model_columns
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT model_id FROM ml_models")).fetchall()
            docs = conn.execute(text("SELECT document_id FROM documents")).fetchall()
            stamped = conn.execute(text(
                "SELECT version_num FROM alembic_version")).scalar()
        assert rows == [("m_upg",)], "legacy ml_models rows must survive"
        assert docs == [("doc_upg",)], "unrelated tables must be untouched"
        assert stamped == ScriptDirectory.from_config(alembic_cfg).get_current_head()
    finally:
        engine.dispose()


def test_upgrade_is_idempotent_and_reversible(legacy_db_url, alembic_cfg):
    command.upgrade(alembic_cfg, "head")
    command.upgrade(alembic_cfg, "head")  # no-op at head

    command.downgrade(alembic_cfg, PREVIOUS_HEAD)
    engine = create_engine(legacy_db_url)
    try:
        inspector = inspect(engine)
        assert "ml_walk_forward_runs" not in inspector.get_table_names()
        model_columns = {c["name"] for c in inspector.get_columns("ml_models")}
        assert "artifact_sha256" not in model_columns
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT model_id FROM ml_models")).fetchall()
        assert rows == [("m_upg",)], "downgrade must not lose data either"
    finally:
        engine.dispose()

    command.upgrade(alembic_cfg, "head")
    engine = create_engine(legacy_db_url)
    try:
        inspector = inspect(engine)
        assert "ml_walk_forward_runs" in inspector.get_table_names()
        assert "artifact_sha256" in {
            c["name"] for c in inspector.get_columns("ml_models")}
    finally:
        engine.dispose()
