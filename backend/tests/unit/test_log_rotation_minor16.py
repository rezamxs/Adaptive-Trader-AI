"""Minor 16: the file log sink is size-rotated, never unbounded.

The app is an always-on worker; logs/app.log grows without limit under the
old plain FileHandler. setup_logging now wires a RotatingFileHandler whose
maxBytes/backupCount come from settings (LOG_ROTATE_MAX_BYTES /
LOG_ROTATE_BACKUP_COUNT).
"""
import logging

import pytest

from backend.app.core.logging import setup_logging


@pytest.fixture
def fresh_logger(tmp_path):
    name = "adaptive_trader_rotation_test"
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    yield name, str(tmp_path)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_file_sink_is_rotating_by_default(fresh_logger):
    name, log_dir = fresh_logger
    logger = setup_logging(logger_name=name, log_dir=log_dir)
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    from logging.handlers import RotatingFileHandler

    assert isinstance(file_handlers[0], RotatingFileHandler), (
        "the file sink must be size-rotated (Minor 16)")
    assert file_handlers[0].maxBytes == 5 * 1024 * 1024
    assert file_handlers[0].backupCount == 3


def test_rotation_settings_are_honored_and_rotation_happens(fresh_logger):
    name, log_dir = fresh_logger
    logger = setup_logging(logger_name=name, log_dir=log_dir, max_bytes=200, backup_count=2)
    (handler,) = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert handler.maxBytes == 200 and handler.backupCount == 2

    for i in range(30):
        logger.info("rotation probe %03d %s", i, "p" * 40)
    for handler in logger.handlers:
        handler.flush()

    import os
    from pathlib import Path

    rotated = sorted(Path(log_dir).glob("app.log*"))
    assert (Path(log_dir) / "app.log").exists()
    assert len(rotated) >= 2, f"expected rotated backups, got {rotated}"
    assert (Path(log_dir) / "app.log").stat().st_size <= 200 + 512, (
        "the active log file must stay bounded near maxBytes")


def test_settings_override_the_defaults(monkeypatch, fresh_logger):
    from backend.app.core import logging as logging_module

    import sys
    import types

    fake_settings = types.SimpleNamespace(LOG_ROTATE_MAX_BYTES=1234, LOG_ROTATE_BACKUP_COUNT=7)
    monkeypatch.setitem(
        sys.modules, "backend.app.core.config",
        types.SimpleNamespace(settings=fake_settings))
    # _rotation_settings imports the config module fresh on every call.
    max_bytes, backup_count = logging_module._rotation_settings()
    assert (max_bytes, backup_count) == (1234, 7)
