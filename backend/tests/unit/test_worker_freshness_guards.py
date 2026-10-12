"""Worker readiness must reject stale, naive, and future tick timestamps."""
from datetime import datetime, timedelta, timezone

from backend.app.core.config import settings
from backend.app.services.market_data.base import Quote
from backend.app.services.market_data.mock_provider import MockMarketDataProvider
from backend.app.services.worker.market_worker import MarketMonitoringWorker


def _worker():
    return MarketMonitoringWorker(data_provider=MockMarketDataProvider())


def test_future_tick_is_not_market_ready():
    now = datetime.now(timezone.utc)
    quote = Quote(
        symbol="XAUUSD", time=now + timedelta(seconds=30),
        bid=2650.0, ask=2650.2, spread_points=20, is_stale=False,
    )
    status, reason = _worker().evaluate_market_freshness(quote, now)
    assert status == "FUTURE_TICK_TIMESTAMP"
    assert "future" in reason


def test_naive_tick_is_not_market_ready():
    now = datetime.now(timezone.utc)
    quote = Quote(
        symbol="XAUUSD", time=datetime.now(),
        bid=2650.0, ask=2650.2, spread_points=20, is_stale=False,
    )
    status, _ = _worker().evaluate_market_freshness(quote, now)
    assert status == "INVALID_TICK_TIMESTAMP"


def test_old_tick_is_not_market_ready_even_when_flag_is_false():
    now = datetime.now(timezone.utc)
    quote = Quote(
        symbol="XAUUSD",
        time=now - timedelta(seconds=settings.WORKER_STALE_TICK_SECONDS + 30),
        bid=2650.0, ask=2650.2, spread_points=20, is_stale=False,
    )
    status, _ = _worker().evaluate_market_freshness(quote, now)
    assert status in {"NO_RECENT_TICK", "MARKET_CLOSED_OR_STALE", "MARKET_CLOSED"}
