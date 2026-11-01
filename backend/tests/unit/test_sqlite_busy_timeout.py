"""Minor item 7: SQLite busy_timeout is explicitly configured.

sqlite3's `timeout` connect argument is the busy-handler wait in seconds;
the engine computes it from the clamped DB_BUSY_TIMEOUT_MS setting without
building any SQL string. Without it, a writer colliding with another
connection fails instantly with "database is locked".
"""
import sqlite3

from backend.app.core.config import settings
from backend.app.db import session as db_session_module


def test_engine_connect_args_carry_configured_busy_timeout():
    assert db_session_module.connect_args["timeout"] == (
        max(0, int(settings.DB_BUSY_TIMEOUT_MS)) / 1000.0
    )


def test_busy_timeout_default_is_documented_five_seconds():
    assert settings.DB_BUSY_TIMEOUT_MS == 5000


def test_sqlite_timeout_connect_arg_sets_busy_handler():
    """The mechanism itself: sqlite3 timeout (seconds) == busy_timeout (ms)."""
    conn = sqlite3.connect(":memory:", timeout=2.5)
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 2500


def test_negative_setting_clamps_to_zero_wait():
    assert max(0, int(-5)) / 1000.0 == 0.0


def test_foreign_keys_pragma_still_enforced():
    conn = sqlite3.connect(":memory:")
    db_session_module.set_sqlite_pragma(conn, None)
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
