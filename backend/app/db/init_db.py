"""Database table creation and initial safety state seeding."""
from pathlib import Path
from typing import Optional

import sqlalchemy
from sqlalchemy.orm import Session
from backend.app.db.base import Base
from backend.app.db.session import engine, SessionLocal
from backend.app.models.emergency_stop import EmergencyStopState
from backend.app.models.audit_log import AuditLog
from backend.app.core.logging import logger

# Bookkeeping table Alembic itself uses (same shape as alembic's stamp):
# created here with pure SQLAlchemy expressions when a FRESH database is
# booted, so the migration-drift gate below can protect it from then on.
_AlembicVersion = sqlalchemy.Table(
    "alembic_version",
    sqlalchemy.MetaData(),
    sqlalchemy.Column("version_num", sqlalchemy.String(32), primary_key=True),
)


def _alembic_chain_head() -> Optional[str]:
    """The head revision of the migration chain, read from alembic.ini.

    Returns None (and only logs) when the chain metadata is unreadable —
    the gate then degrades to the historical create_all-only behavior instead
    of bricking a boot because of a packaging-path issue.
    """
    try:
        from alembic.config import Config as AlembicConfig
        from alembic.script import ScriptDirectory

        # backend/app/db/init_db.py → repo root is three parents up.
        repo_root = Path(__file__).resolve().parents[3]
        cfg = AlembicConfig(str(repo_root / "alembic.ini"))
        return ScriptDirectory.from_config(cfg).get_current_head()
    except Exception as exc:  # pragma: no cover - defensive packaging path
        logger.warning("Alembic chain head unavailable; schema-drift gate skipped: %s", exc)
        return None


def init_db(db: Optional[Session] = None) -> None:
    """Creates missing database tables and verifies initial safety state.

    M20 (phase 2): boot NEVER mutates an existing schema beyond create_all's
    idempotent no-op, and a database managed by Alembic must BE at the chain
    head before boot — otherwise a brand-new table added to the models could
    silently appear outside the migration chain. Specifically:

    - versioned DB behind head  -> RuntimeError (operator runs the migration);
    - unversioned EMPTY DB      -> create_all + stamp head (fresh-install path);
    - unversioned POPULATED DB  -> legacy pre-Alembic layout: create_all with
      a warning + stamp head so drift detection starts from this boot on;
    - versioned at head         -> create_all no-op (behaviorally pinned).

    Schema drift beyond create_all (new columns on pre-existing tables) is
    owned by the Alembic chain (migrations/versions); boot must not ALTER.
    """
    logger.info("Initializing database tables...")

    head = _alembic_chain_head()
    version: Optional[str] = None
    versions_in_db: list = []
    with engine.connect() as conn:
        existing = sqlalchemy.inspect(conn)
        table_names = set(existing.get_table_names())
        if "alembic_version" in table_names:
            # Buglist item 29: .scalar() read only the FIRST alembic_version
            # row. A database with multiple heads (an un-merged migration
            # branch) silently compared one arbitrary version and could boot
            # in a broken state. Read EVERY row: multiple distinct versions
            # fail loudly below and name the fix. (All reads stay inside the
            # connection context.)
            versions_in_db = sorted({
                row[0]
                for row in conn.execute(
                    sqlalchemy.select(_AlembicVersion.c.version_num)
                ).fetchall()
                if row[0]
            })
            if len(versions_in_db) == 1:
                version = versions_in_db[0]
            elif versions_in_db:
                version = ",".join(versions_in_db)

    if head is not None and len(versions_in_db) > 1:
        raise RuntimeError(
            "Database has MULTIPLE alembic_version rows "
            f"({', '.join(versions_in_db)}); the migration chain is branched. "
            "Run `alembic heads` and `alembic merge` (or `alembic upgrade head` "
            "against the correct lineage) before starting the app."
        )

    if head is not None and version is not None and version != head:
        raise RuntimeError(
            f"Database schema is at migration {version!r} but the code expects "
            f"{head!r}. Run `alembic upgrade head` before starting the app; "
            "booting behind the migration chain could silently create new "
            "model tables outside the migration chain."
        )

    if head is not None and version is not None:
        # The M20 hole this gate closes: on a MIGRATED database, create_all
        # would silently materialize any brand-new model table outside the
        # Alembic chain. Refuse and name the missing tables instead.
        with engine.connect() as conn:
            present = set(sqlalchemy.inspect(conn).get_table_names())
        missing = sorted(set(Base.metadata.tables) - present)
        if missing:
            raise RuntimeError(
                f"Model tables {missing} do not exist in the migrated database; "
                "boot refuses to create tables outside the Alembic chain — "
                "add a migration for them first."
            )

    Base.metadata.create_all(bind=engine)

    if head is not None and version is None:
        with engine.begin() as conn:
            _AlembicVersion.create(bind=conn, checkfirst=True)
            conn.execute(_AlembicVersion.insert().values(version_num=head))
        logger.info(
            "Fresh database created from models and stamped at migration head %r.",
            head,
        )

    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    try:
        # Check if an emergency stop state exists; if not, initialize as NOT active
        last_state = db.query(EmergencyStopState).order_by(EmergencyStopState.id.desc()).first()
        if not last_state:
            initial_state = EmergencyStopState(
                is_active=False,
                activated_by="SYSTEM_BOOTSTRAP",
                reason="System initialized with safety circuit breaker inactive by default."
            )
            db.add(initial_state)

            audit = AuditLog(
                event_type="SYSTEM_BOOTSTRAP",
                severity="INFO",
                actor="SYSTEM",
                details={"status": "Database initialized, emergency stop ready."}
            )
            db.add(audit)
            db.commit()
            logger.info("Initialized default emergency stop state (Inactive).")
        else:
            logger.info(f"Loaded existing emergency stop state: Active={last_state.is_active}")
    finally:
        if close_session:
            db.close()


if __name__ == "__main__":
    init_db()
    print("Database tables initialized successfully.")
