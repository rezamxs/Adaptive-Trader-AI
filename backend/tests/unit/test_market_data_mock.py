"""Unit tests for Market Data Provider interface using MockMarketDataProvider."""
import pytest
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.market_data.mock_provider import MockMarketDataProvider


def test_mock_provider_connection_and_account():
    """Verify mock provider connection and demo account reporting."""
    provider = MockMarketDataProvider(is_demo_account=True)
    assert provider.is_connected() is True

    acc = provider.get_account_info()
    assert acc is not None
    assert acc.is_demo is True
    assert acc.trade_mode == 0
    assert acc.login_masked == "******99"

    provider.disconnect()
    assert provider.is_connected() is False
    assert provider.get_account_info() is None


def test_mock_provider_symbol_specs():
    """Verify symbol specifications and contract attributes."""
    provider = MockMarketDataProvider()
    symbols = provider.get_symbols()
    assert "EURUSD" in symbols

    spec = provider.get_symbol_info("EURUSD")
    assert spec is not None
    assert spec.digits == 5
    assert spec.volume_min == 0.01
    assert spec.point == 0.00001


def test_mock_provider_historical_dataframe():
    """Verify historical candles dataframe structure, columns, and index."""
    provider = MockMarketDataProvider()
    df = provider.get_historical_dataframe("EURUSD", TimeFrame.M15, count=100)

    assert len(df) == 100
    assert list(df.columns) == ["open", "high", "low", "close", "tick_volume", "spread"]
    assert df.index.name == "time"
    # Verify High >= Low and High >= Open/Close
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()


def test_mock_provider_quote():
    """Verify live quote generation and spread calculation."""
    provider = MockMarketDataProvider()
    quote = provider.get_current_quote("EURUSD")
    assert quote is not None
    assert quote.symbol == "EURUSD"
    assert quote.ask > quote.bid
    assert quote.is_stale is False
