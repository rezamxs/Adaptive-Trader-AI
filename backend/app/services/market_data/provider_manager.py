"""Process-wide shared MT5 provider manager (M12 fix).

The MetaTrader5 python API is process-global: ``mt5.initialize()`` /
``mt5.shutdown()`` control ONE terminal connection shared by every caller in
the process. The previous per-request pattern (a fresh provider per call with
``disconnect()`` in a ``finally``) made every API request, every training run
and every backtest tear down the very terminal the worker and other
concurrent holders were using.

Contract:
- ``acquire_shared_provider()`` returns the process-wide provider instance
  (creating it on first use) and registers the caller as a user. Callers must
  pair it with ``release_shared_provider()``.
- ``release_shared_provider()`` NEVER calls ``mt5.shutdown()`` — the terminal
  is deliberately kept alive for the worker and the next holder. Only
  ``shutdown_shared_provider()`` may tear the connection down (application
  shutdown and tests).
- ``shared_provider_instance()`` exposes the singleton for ownership checks
  (the worker only shuts the terminal down when it holds the shared instance).
"""
import threading
from typing import Optional

from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider

_manager_lock = threading.Lock()
_shared_provider: Optional[MT5MarketDataProvider] = None
_active_users = 0


def acquire_shared_provider() -> MT5MarketDataProvider:
    """Returns the shared provider, registering the caller as a user."""
    global _shared_provider, _active_users
    with _manager_lock:
        if _shared_provider is None:
            _shared_provider = MT5MarketDataProvider()
        _active_users += 1
        return _shared_provider


def release_shared_provider(provider: MT5MarketDataProvider) -> None:
    """Releases one user of the shared provider WITHOUT shutting MT5 down.

    Deliberately no ``mt5.shutdown()`` here (M12): the terminal connection is
    shared with the worker and any other current holder; tearing it down from
    a request path strands them mid-operation. Only
    ``shutdown_shared_provider()`` (application shutdown / tests) disconnects.
    """
    global _active_users
    with _manager_lock:
        if provider is _shared_provider and _active_users > 0:
            _active_users -= 1


def shared_provider_instance() -> Optional[MT5MarketDataProvider]:
    """Returns the current shared provider instance, or None before first use."""
    with _manager_lock:
        return _shared_provider


def shutdown_shared_provider() -> None:
    """The ONLY sanctioned teardown of the shared terminal connection."""
    global _shared_provider, _active_users
    with _manager_lock:
        if _shared_provider is not None:
            _shared_provider.disconnect()
            _shared_provider = None
        _active_users = 0


def reset_provider_manager_for_tests() -> None:
    """Test isolation hook: disconnects and clears the shared singleton."""
    shutdown_shared_provider()
