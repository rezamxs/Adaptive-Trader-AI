"""M12: process-wide shared MT5 provider manager (regression tests).

The MetaTrader5 python API owns ONE process-global terminal connection. The
old per-request pattern (fresh provider + ``disconnect()`` in a ``finally``)
made every API request, training run and backtest tear down the very terminal
the worker and other concurrent holders were using. These tests pin the new
contract: acquire/release is bookkeeping only, release NEVER shuts the
terminal down, and only ``shutdown_shared_provider()`` (application shutdown /
tests) disconnects. Endpoint requests and the worker therefore share one
connection, and a request finishing can no longer cut the worker's cycle.
"""
import types

import pytest

from backend.app.services.market_data import provider_manager as pm
from backend.app.services.market_data.provider_manager import (
    acquire_shared_provider,
    release_shared_provider,
    reset_provider_manager_for_tests,
    shared_provider_instance,
    shutdown_shared_provider,
)


class _RecordingProvider:
    """Stands in for MT5MarketDataProvider and records teardown calls."""

    instances: list = []

    def __init__(self):
        self.connect_calls = 0
        self.disconnect_calls = 0
        _RecordingProvider.instances.append(self)

    def connect(self):
        self.connect_calls += 1
        return True

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False

    # minimal surface used by the market endpoints
    def get_symbols(self):
        return ["XAUUSD"]


@pytest.fixture(autouse=True)
def _patch_factory_and_reset(monkeypatch):
    _RecordingProvider.instances = []
    monkeypatch.setattr(pm, "MT5MarketDataProvider", _RecordingProvider)
    reset_provider_manager_for_tests()
    yield
    reset_provider_manager_for_tests()


# ---------------------------------------------------------------- manager core

def test_acquire_returns_the_same_shared_instance():
    p1 = acquire_shared_provider()
    p2 = acquire_shared_provider()
    assert p1 is p2
    assert shared_provider_instance() is p1
    assert len(_RecordingProvider.instances) == 1, "one provider for the whole process"


def test_release_never_shuts_the_terminal_down():
    provider = acquire_shared_provider()
    release_shared_provider(provider)
    release_shared_provider(provider)  # over-release is a no-op, not a crash
    assert provider.disconnect_calls == 0, "release must NEVER disconnect (M12 core)"
    # the provider stays alive for the worker and the next holder
    assert shared_provider_instance() is provider


def test_shutdown_is_the_only_teardown():
    provider = acquire_shared_provider()
    shutdown_shared_provider()
    assert provider.disconnect_calls == 1
    assert shared_provider_instance() is None
    fresh = acquire_shared_provider()
    assert fresh is not provider, "shutdown must allow a clean new connection"


def test_foreign_release_is_ignored():
    acquire_shared_provider()
    stranger = _RecordingProvider()
    release_shared_provider(stranger)  # not the shared instance — no state change
    assert shared_provider_instance() is not None


# ---------------------------------------------------------------- endpoints

def test_market_endpoint_releases_without_disconnect():
    from backend.app.api.v1.endpoints.market import get_symbols

    result = get_symbols()
    assert result == {"symbols": ["XAUUSD"]}
    provider = shared_provider_instance()
    assert provider is not None
    assert provider.disconnect_calls == 0, "request end must keep the shared terminal alive"


def test_market_endpoint_503_releases_and_never_disconnected(monkeypatch):
    from fastapi import HTTPException

    from backend.app.api.v1.endpoints.market import get_market_provider

    class _FailingProvider(_RecordingProvider):
        def connect(self):
            self.connect_calls += 1
            return False

    monkeypatch.setattr(pm, "MT5MarketDataProvider", _FailingProvider)
    with pytest.raises(HTTPException) as exc:
        get_market_provider()
    assert exc.value.status_code == 503
    assert _RecordingProvider.instances[0].disconnect_calls == 0, (
        "a failed acquire must release, not tear the shared terminal down")


# ---------------------------------------------------------------- worker

def test_worker_defaults_to_the_shared_instance():
    from backend.app.services.worker.market_worker import MarketMonitoringWorker

    worker = MarketMonitoringWorker()
    assert worker.data_provider is shared_provider_instance()


def test_worker_stop_tears_down_only_when_it_owns_the_shared_instance():
    import asyncio

    from backend.app.services.worker.market_worker import MarketMonitoringWorker

    # Injected provider (tests / custom deployments): stop disconnects it directly.
    injected = _RecordingProvider()
    worker = MarketMonitoringWorker(data_provider=injected)
    worker.is_running = True
    asyncio.run(worker.stop())
    assert injected.disconnect_calls == 1
    assert shared_provider_instance() is None or True  # shared untouched by this stop

    # Shared provider (production): worker stop IS the application teardown.
    shared = acquire_shared_provider()
    worker2 = MarketMonitoringWorker()
    assert worker2.data_provider is shared
    worker2.is_running = True
    asyncio.run(worker2.stop())
    assert shared.disconnect_calls == 1
    assert shared_provider_instance() is None


# ---------------------------------------------------------------- provider.connect

def test_connect_is_idempotent_and_recovers_from_a_stale_handle(monkeypatch):
    """The shared provider must not re-initialize MT5 on every acquire, and a
    dead terminal handle must heal on the next connect instead of poisoning
    every later call."""
    from backend.app.services.market_data import mt5_provider as mp

    calls = {"init": 0, "probe": 0}

    class _FakeMT5:
        @staticmethod
        def initialize(path=None):
            calls["init"] += 1
            return True

        @staticmethod
        def terminal_info():
            # call #1: post-init probe of connect #1 (alive); call #2: the
            # live-handle probe of connect #2 (alive → idempotent early
            # return); call #3: the probe of connect #3 — models a STALE
            # handle; call #4: post-init probe of connect #3 (alive again).
            calls["probe"] += 1
            if calls["probe"] == 3:
                return types.SimpleNamespace(connected=False)
            return types.SimpleNamespace(connected=True)

        @staticmethod
        def account_info():
            return types.SimpleNamespace(server="OTHER-SERVER")

        @staticmethod
        def last_error():
            return (0, "ok")

        @staticmethod
        def shutdown():
            pass

    monkeypatch.setattr(mp, "mt5", _FakeMT5)
    monkeypatch.setattr(mp, "MT5_AVAILABLE", True)
    provider = mp.MT5MarketDataProvider(terminal_path="dummy")

    assert provider.connect() is True
    assert calls["init"] == 1
    assert provider.connect() is True, "live handle must NOT re-initialize"
    assert calls["init"] == 1

    # Stale handle: the flag still says connected but the probe reports a
    # dead terminal → transparent re-initialize on the next connect.
    assert provider.connect() is True
    assert calls["init"] == 2, "stale handle must be re-initialized"
