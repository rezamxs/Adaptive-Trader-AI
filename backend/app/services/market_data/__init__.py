"""Market Data Module."""
from backend.app.services.market_data.base import (
    IMarketDataProvider,
    TimeFrame,
    Candle,
    Quote,
    SymbolInfo,
    AccountDetails,
)
from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider
from backend.app.services.market_data.mock_provider import MockMarketDataProvider

__all__ = [
    "IMarketDataProvider",
    "TimeFrame",
    "Candle",
    "Quote",
    "SymbolInfo",
    "AccountDetails",
    "MT5MarketDataProvider",
    "MockMarketDataProvider",
]
