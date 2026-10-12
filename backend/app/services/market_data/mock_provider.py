"""Mock Market Data Provider for unit tests and reproducible simulations."""
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import numpy as np
import pandas as pd

from backend.app.services.market_data.base import (
    IMarketDataProvider,
    TimeFrame,
    Candle,
    Quote,
    SymbolInfo,
    AccountDetails,
)


class MockMarketDataProvider(IMarketDataProvider):
    """Generates synthetic or fixture-based OHLCV data for offline verification."""

    def __init__(self, is_demo_account: bool = True, default_symbol: str = "EURUSD"):
        self._connected = True
        self.is_demo = is_demo_account
        self.default_symbol = default_symbol

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_account_info(self) -> Optional[AccountDetails]:
        if not self._connected:
            return None
        return AccountDetails(
            login_masked="******99",
            server="MockBroker-Demo",
            currency="USD",
            leverage=50,
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            free_margin=10000.0,
            trade_mode=0 if self.is_demo else 2,
            is_demo=self.is_demo,
        )

    def get_open_positions_count(self) -> Optional[int]:
        """Synthetic providers have no actual open broker positions."""
        return 0 if self._connected else None

    def get_daily_loss_percent(self) -> Optional[float]:
        """Synthetic provider reports only its fixed demo fixture state."""
        return 0.0 if self._connected else None

    def calculate_order_margin(
        self, symbol: str, direction: str, volume: float, price: float
    ) -> Optional[float]:
        """Approximate margin for isolated tests; this is not broker execution evidence."""
        info = self.get_symbol_info(symbol)
        account = self.get_account_info()
        if not info or not account or direction not in {"BUY", "SELL"} or volume <= 0 or price <= 0:
            return None
        return float(volume * info.trade_contract_size * price / max(account.leverage, 1))

    def get_symbols(self) -> List[str]:
        return ["EURUSD", "GBPUSD", "USDJPY"]

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        if not self._connected:
            return None
        return SymbolInfo(
            name=symbol,
            digits=5,
            point=0.00001,
            spread=15,
            tick_size=0.00001,
            tick_value=1.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            trade_contract_size=100000.0,
            currency_base="EUR",
            currency_profit="USD",
            currency_margin="EUR",
        )

    def get_current_quote(self, symbol: str, max_staleness_seconds: int = 60) -> Optional[Quote]:
        if not self._connected:
            return None
        now = datetime.now(timezone.utc)
        return Quote(
            symbol=symbol,
            time=now,
            bid=1.08500,
            ask=1.08515,
            spread_points=15,
            is_stale=False,
        )

    def get_historical_candles(
        self,
        symbol: str,
        timeframe: TimeFrame,
        count: int = 500,
        start_pos: int = 0
    ) -> List[Candle]:
        if not self._connected:
            return []

        candles = []
        base_time = datetime.now(timezone.utc) - timedelta(minutes=count * 15)
        price = 1.0850

        # Deterministic pseudo-random seed based on count
        rng = np.random.RandomState(42)
        returns = rng.normal(loc=0.00002, scale=0.0005, size=count)

        for i in range(count):
            bar_time = base_time + timedelta(minutes=i * 15)
            open_p = price
            close_p = open_p * (1 + returns[i])
            high_p = max(open_p, close_p) + abs(rng.normal(0, 0.0002))
            low_p = min(open_p, close_p) - abs(rng.normal(0, 0.0002))
            price = close_p

            candles.append(Candle(
                time=bar_time,
                open=round(open_p, 5),
                high=round(high_p, 5),
                low=round(low_p, 5),
                close=round(close_p, 5),
                tick_volume=int(rng.randint(50, 500)),
                spread=15,
                is_closed=True,
            ))

        return candles

    def get_historical_dataframe(
        self,
        symbol: str,
        timeframe: TimeFrame,
        count: int = 500
    ) -> pd.DataFrame:
        candles = self.get_historical_candles(symbol, timeframe, count)
        records = [{
            "time": c.time,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "tick_volume": c.tick_volume,
            "spread": c.spread,
        } for c in candles]
        df = pd.DataFrame(records)
        df.set_index("time", inplace=True)
        return df
