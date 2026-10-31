"""Shared test fixtures.

C1 regression guard: API tests must never read or write the real
``adaptive_trader.db``. Tests that exercise the FastAPI app opt in to the
``isolated_app_db`` fixture, which reroutes the app's ``get_db`` dependency
(and the lifespan ``init_db`` / worker ``db_factory`` paths) at an isolated
in-memory SQLite database.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.app.models  # noqa: F401 - register all ORM tables on Base
from backend.app.core.config import settings
from backend.app.db.base import Base
from backend.app.db.session import get_db as real_get_db
from backend.app.main import app


@pytest.fixture()
def isolated_app_db(monkeypatch):
    """Point the app's database access at an isolated in-memory SQLite DB.

    StaticPool pins one shared connection so create_all's tables stay visible
    to every later session; without it a fresh per-connection empty in-memory
    DB makes any request-time query 500.
    """
    engine = create_engine(
        "sqlite://",
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def test_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[real_get_db] = test_get_db

    def test_init_db(db=None):
        Base.metadata.create_all(bind=engine)

    # Lifespan (TestClient context manager) must not touch the real DB
    # through init_db, and must never start the real market worker.
    monkeypatch.setattr("backend.app.main.init_db", test_init_db)
    monkeypatch.setattr(settings, "WORKER_ENABLED", False)

    # Worker diagnostics (read by /health and /system/status) query the DB
    # through the singleton's db_factory; isolate that read path too.
    from backend.app.services.worker.market_worker import get_market_worker
    monkeypatch.setattr(get_market_worker(), "db_factory", factory, raising=False)

    yield factory
    app.dependency_overrides.pop(real_get_db, None)
    engine.dispose()
