"""Structured Logging and Security Audit Trail for Adaptive Trader AI.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

# Minor 16: the file log must be size-bounded — an unbounded FileHandler grows
# forever on a always-on worker. Defaults here; core.config may override via
# LOG_ROTATE_MAX_BYTES / LOG_ROTATE_BACKUP_COUNT (imported lazily to avoid a
# config<->logging import cycle).
_LOG_ROTATE_MAX_BYTES = 5 * 1024 * 1024
_LOG_ROTATE_BACKUP_COUNT = 3


def _rotation_settings() -> "tuple[int, int]":
    try:
        from backend.app.core.config import settings

        return (
            max(1, int(getattr(settings, "LOG_ROTATE_MAX_BYTES", _LOG_ROTATE_MAX_BYTES))),
            max(0, int(getattr(settings, "LOG_ROTATE_BACKUP_COUNT", _LOG_ROTATE_BACKUP_COUNT))),
        )
    except Exception:
        return (_LOG_ROTATE_MAX_BYTES, _LOG_ROTATE_BACKUP_COUNT)


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "./logs",
    logger_name: str = "adaptive_trader",
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
) -> logging.Logger:
    """Configures application-wide structured logging with a rotated file sink.

    Buglist item 40: the module-level ``# Buglist item 40: importing this module used to create logs/app.log.
# Import now wires the console sink only; main.py's lifespan calls
# setup_logging() to attach the rotated file sink at real startup.
logger = _console_only_logger()`` used to run
    AT IMPORT TIME — merely importing the logging module created
    <log_dir>/ and opened logs/app.log. The import path now installs the
    console sink ONLY (no filesystem side effect); the file sink is attached
    by an explicit setup_logging() call from the application entry point
    (main.py lifespan), and the handler itself opens the file lazily
    (delay=True) on the first record."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    if max_bytes is None or backup_count is None:
        cfg_bytes, cfg_count = _rotation_settings()
        max_bytes = max_bytes if max_bytes is not None else cfg_bytes
        backup_count = backup_count if backup_count is not None else cfg_count

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        # Console Formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_format = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(module)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # File Formatter (size-rotated: app.log + app.log.1..N)
        file_handler = RotatingFileHandler(
            f"{log_dir}/app.log",
            encoding="utf-8",
            maxBytes=max_bytes,
            backupCount=backup_count,
            # Open on FIRST emit, not at configuration time: a configured-but
            # -unused handler (tests, import-only paths) creates no file.
            delay=True,
        )
        file_handler.setLevel(numeric_level)
        file_format = logging.Formatter(
            '{"timestamp":"%(asctime)s", "level":"%(levelname)s", "module":"%(module)s", "message":"%(message)s"}',
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def _console_only_logger(
    log_level: str = "INFO",
    logger_name: str = "adaptive_trader",
) -> logging.Logger:
    """Import-time logger: stdout sink ONLY — no directory creation, no file
    handles (buglist item 40). The file sink arrives with setup_logging()."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    lg = logging.getLogger(logger_name)
    if not lg.handlers:
        lg.setLevel(numeric_level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(module)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        lg.addHandler(console_handler)
    return lg


def log_security_event(event_type: str, details: Dict[str, Any], level: str = "WARNING") -> None:
    """Logs security, safety gate, or mode-violation events explicitly."""
    audit_logger = logging.getLogger("adaptive_trader.audit")
    msg = f"SECURITY_AUDIT | Event={event_type} | Details={details}"
    if level == "CRITICAL":
        audit_logger.critical(msg)
    elif level == "ERROR":
        audit_logger.error(msg)
    elif level == "WARNING":
        audit_logger.warning(msg)
    else:
        audit_logger.info(msg)


# Buglist item 40: importing this module used to create logs/app.log.
# Import now wires the console sink only; main.py's lifespan calls
# setup_logging() to attach the rotated file sink at real startup.
logger = _console_only_logger()
