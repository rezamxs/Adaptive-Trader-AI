"""Market Data Endpoints.

Buglist item 17: every route here is token-gated. Each call touches the
shared MT5 terminal (acquire + connect + symbol/quote/candle reads), so an
unauthenticated poller could pressure the worker's terminal connection for
free — the same governance rule the data-bearing endpoints already follow.
"""
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.core.auth import require_api_token
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider
from backend.app.services.market_data.provider_manager import (
    acquire_shared_provider,
    release_shared_provider,
)

router = APIRouter()


def get_market_provider() -> MT5MarketDataProvider:
    """Returns the shared connected MT5 provider or raises 503; never serves mock data silently.

    M12: the provider is the process-wide shared instance — releasing it NEVER
    shuts the terminal down (the worker and concurrent requests hold the same
    connection). Only application shutdown may disconnect.
    """
    provider = acquire_shared_provider()
    connected = False
    try:
        connected = provider.connect()
    finally:
        # Buglist item 16: an exception from connect() must release the
        # acquired counter too, not strand it (the leak path).
        if not connected:
            release_shared_provider(provider)
    if not connected:
        raise HTTPException(
            status_code=503,
            detail="MT5 is unavailable; market data endpoints refuse to serve mock data.",
        )
    return provider


@router.get("/symbols")
def get_symbols(_token: str = Depends(require_api_token)):
    """Lists available trading symbols (item 17: gated)."""
    provider = get_market_provider()
    try:
        symbols = provider.get_symbols()
        return {"symbols": symbols}
    finally:
        release_shared_provider(provider)


@router.get("/quote/{symbol}")
def get_quote(symbol: str, _token: str = Depends(require_api_token)):
    """Returns current live quote for symbol (item 17: gated)."""
    provider = get_market_provider()
    try:
        quote = provider.get_current_quote(symbol)
        if not quote:
            raise HTTPException(status_code=404, detail="Quote not available.")
        return {
            "symbol": quote.symbol,
            "bid": quote.bid,
            "ask": quote.ask,
            "spread_points": quote.spread_points,
            "is_stale": quote.is_stale,
            "time": quote.time.isoformat(),
        }
    finally:
        release_shared_provider(provider)


@router.get("/candles/{symbol}")
def get_candles(
    symbol: str,
    timeframe: TimeFrame = TimeFrame.M15,
    count: int = Query(default=100, ge=10, le=1000),
    _token: str = Depends(require_api_token),
):
    """Returns historical candles for a symbol and timeframe (item 17: gated)."""
    provider = get_market_provider()
    try:
        candles = provider.get_historical_candles(symbol, timeframe, count=count)
        return [{
            "time": c.time.isoformat(),
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "tick_volume": c.tick_volume,
            "spread": c.spread,
        } for c in candles]
    finally:
        release_shared_provider(provider)
