"""M20 phase 2: the boot schema-drift gate in init_db.

create_all stays the fresh-install path, but a database managed by Alembic
must BE at the chain head before boot, and boot refuses to materialize new
model tables outside the migration chain. Cases: fresh-empty (create+stamp),
versioned-at-head (no-op), versioned-behind (RuntimeError), versioned with a
missing model table (RuntimeError), legacy unversioned-populated (warn +
create_all + stamp, data survives).
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.db import init_db as init_db_module


@pytest.fixture
def tmp_engine(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'boot.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    monkeypatch.setattr(init_db_module, "engine", engine)
    monkeypatch.setattr(init_db_module, "SessionLocal", sessionmaker(bind=engine))
    yield engine
    engine.dispose()


def _stamped_version(engine) -> str:
    with engine.connect() as conn:
        return conn.execute(
            init_db_module.sqlalchemy.select(init_db_module._AlembicVersion.c.version_num)
        ).scalar()


def _set_version(engine, version: str) -> None:
    with engine.begin() as conn:
        init_db_module._AlembicVersion.create(bind=conn, checkfirst=True)
        conn.execute(init_db_module._AlembicVersion.insert().values(version_num=version))


def test_fresh_database_is_created_and_stamped_at_chain_head(tmp_engine):
    init_db_module.init_db()
    head = init_db_module._alembic_chain_head()
    assert head is not None, "the repo ships a migration chain"
    assert _stamped_version(tmp_engine) == head, (
        "a fresh boot must stamp the alembic head so drift detection applies "
        "from the very next boot")
    from backend.app.db.base import Base

    with tmp_engine.connect() as conn:
        present = set(init_db_module.sqlalchemy.inspect(conn).get_table_names())
    assert set(Base.metadata.tables) <= present


def test_versioned_at_head_boot_is_a_noop(tmp_engine):
    init_db_module.init_db()
    with tmp_engine.connect() as conn:
        tables_before = sorted(
            init_db_module.sqlalchemy.inspect(conn).get_table_names())
        rows_before = conn.execute(
            init_db_module.sqlalchemy.select(
                init_db_module._AlembicVersion.c.version_num)).fetchall()

    init_db_module.init_db()  # second boot

    with tmp_engine.connect() as conn:
        assert sorted(init_db_module.sqlalchemy.inspect(conn).get_table_names()) == tables_before
        assert conn.execute(
            init_db_module.sqlalchemy.select(
                init_db_module._AlembicVersion.c.version_num)).fetchall() == rows_before


def test_versioned_database_behind_head_refuses_to_boot(tmp_engine):
    init_db_module.init_db()
    with tmp_engine.begin() as conn:
        conn.execute(init_db_module._AlembicVersion.delete())
        conn.execute(init_db_module._AlembicVersion.insert().values(version_num="19990101_fake"))

    with pytest.raises(RuntimeError, match="alembic upgrade head"):
        init_db_module.init_db()


def test_new_model_table_outside_the_chain_refuses_to_boot(tmp_engine):
    """The exact M20 hole: a model gains a table with NO migration — boot must
    refuse instead of silently creating it outside the Alembic chain."""
    init_db_module.init_db()
    # Simulate an out-of-chain model table: drop one the models declare.
    table_name = next(iter(init_db_module.Base.metadata.tables))
    with tmp_engine.begin() as conn:
        conn.exec_driver_sql(f'DROP TABLE "{table_name}"')

    with pytest.raises(RuntimeError, match="outside the Alembic chain"):
        init_db_module.init_db()


def test_legacy_unversioned_populated_database_is_adopted(tmp_engine):
    """A populated DB with NO alembic_version (pre-gate layout) is adopted:
    create_all stays a no-op for existing tables, the head is stamped so drift
    detection starts, and existing data survives."""
    init_db_module.init_db()
    from backend.app.models.audit_log import AuditLog

    from sqlalchemy.orm import Session

    with Session(tmp_engine) as session:
        session.add(AuditLog(event_type="LEGACY_PROBE", severity="INFO", details={}))
        session.commit()

    # Simulate legacy: remove the version row but keep the tables.
    with tmp_engine.begin() as conn:
        conn.execute(init_db_module._AlembicVersion.delete())

    init_db_module.init_db()
    assert _stamped_version(tmp_engine) == init_db_module._alembic_chain_head()
    with Session(tmp_engine) as session:
        assert session.query(AuditLog).filter_by(event_type="LEGACY_PROBE").first() is not None
