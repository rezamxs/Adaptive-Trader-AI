"""Abstract Market Data Provider Interface and Data Transfer Objects."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import pandas as pd


class TimeFrame(str, Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


@dataclass
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: int
    is_closed: bool = True


@dataclass
class Quote:
    symbol: str
    time: datetime
    bid: float
    ask: float
    spread_points: int
    is_stale: bool = False


@dataclass
class SymbolInfo:
    name: str
    digits: int
    point: float
    spread: int
    tick_size: float
    tick_value: float
    volume_min: float
    volume_max: float
    volume_step: float
    trade_contract_size: float
    currency_base: str
    currency_profit: str
    currency_margin: str


@dataclass
class AccountDetails:
    login_masked: str
    server: str
    currency: str
    leverage: int
    balance: float
    equity: float
    margin: float
    free_margin: float
    trade_mode: int  # 0: DEMO, 1: CONTEST, 2: REAL
    is_demo: bool


class IMarketDataProvider(ABC):
    """Abstract interface defining the contract for all market data sources."""

    @abstractmethod
    def connect(self) -> bool:
        """Establishes connection to the data provider."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Closes connection to the data provider."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Returns True if connected to provider and server."""
        pass

    @abstractmethod
    def get_account_info(self) -> Optional[AccountDetails]:
        """Returns masked account details and demo status."""
        pass

    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Returns available tradable symbol names."""
        pass

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Returns contract specifications and trading limits for a symbol."""
        pass

    @abstractmethod
    def get_current_quote(self, symbol: str, max_staleness_seconds: int = 60) -> Optional[Quote]:
        """Returns current bid/ask quote with staleness validation."""
        pass

    @abstractmethod
    def get_historical_candles(
        self,
        symbol: str,
        timeframe: TimeFrame,
        count: int = 500,
        start_pos: int = 0
    ) -> List[Candle]:
        """Returns historical bars ordered chronologically (oldest to newest)."""
        pass

    @abstractmethod
    def get_historical_dataframe(
        self,
        symbol: str,
        timeframe: TimeFrame,
        count: int = 500
    ) -> pd.DataFrame:
        """Returns historical bars as a pandas DataFrame with datetime index."""
        pass
