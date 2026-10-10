"""Integration tests for live MetaTrader 5 Data Provider."""
import pytest
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider, MT5_AVAILABLE


@pytest.mark.skipif(not MT5_AVAILABLE, reason="MetaTrader5 python package not installed")
def test_live_mt5_provider_read_only_integration():
    """Live integration test verifying read-only connectivity to local MT5 terminal."""
    provider = MT5MarketDataProvider()
    connected = provider.connect()

    if not connected:
        pytest.skip("MT5 terminal could not be initialized or is not running locally.")

    try:
        assert provider.is_connected() is True

        # Read account info
        acc = provider.get_account_info()
        assert acc is not None
        assert acc.is_demo is True  # Must be strictly DEMO
        assert acc.trade_mode == 0
        assert "**" in acc.login_masked  # Login must be masked

        # Read available symbols
        symbols = provider.get_symbols()
        assert len(symbols) > 0
        assert "EURUSD" in symbols or len(symbols) >= 1

        target_symbol = "EURUSD" if "EURUSD" in symbols else symbols[0]

        # Read contract specifications
        info = provider.get_symbol_info(target_symbol)
        assert info is not None
        assert info.digits > 0
        assert info.point > 0

        # Read historical candles
        df = provider.get_historical_dataframe(target_symbol, TimeFrame.M1, count=10)
        assert len(df) > 0
        assert "close" in df.columns

    finally:
        provider.disconnect()
        assert provider.is_connected() is False
