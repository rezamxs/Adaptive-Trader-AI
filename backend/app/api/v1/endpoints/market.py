"""Market Data Endpoints."""
from fastapi import APIRouter, HTTPException, Query
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
    if provider.connect():
        return provider
    release_shared_provider(provider)
    raise HTTPException(
        status_code=503,
        detail="MT5 is unavailable; market data endpoints refuse to serve mock data.",
    )


@router.get("/symbols")
def get_symbols():
    """Lists available trading symbols."""
    provider = get_market_provider()
    try:
        symbols = provider.get_symbols()
        return {"symbols": symbols}
    finally:
        release_shared_provider(provider)


@router.get("/quote/{symbol}")
def get_quote(symbol: str):
    """Returns current live quote for symbol."""
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
    count: int = Query(default=100, ge=10, le=1000)
):
    """Returns historical candles for a symbol and timeframe."""
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
