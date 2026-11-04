"""Database Session Management and Engine configuration.
Supports SQLite (with WAL mode and FK enforcement) and PostgreSQL seamlessly.
"""
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from backend.app.core.config import settings
from backend.app.core.logging import logger

database_url = settings.DATABASE_URL

# Engine configuration
connect_args = {}
if database_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False
    # Minor item 7: bound, config-driven busy timeout. sqlite3's `timeout`
    # connect argument IS the busy-handler wait (seconds); it avoids building
    # any SQL string. Clamped to >= 0 from the settings integer.
    connect_args["timeout"] = max(0, int(settings.DB_BUSY_TIMEOUT_MS)) / 1000.0

engine = create_engine(
    database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=settings.DEBUG,
)

# Enable SQLite foreign key enforcement and WAL mode for reliability
if database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency for database session management with automatic cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
