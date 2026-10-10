"""MetaTrader 5 Market Data Provider implementation (Read-Only).
Handles MT5 terminal connection, symbol specs, candles, and staleness checks.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.security import mask_login, MT5_ACCOUNT_TRADE_MODE_DEMO
from backend.app.services.market_data.base import (
    IMarketDataProvider,
    TimeFrame,
    Candle,
    Quote,
    SymbolInfo,
    AccountDetails,
)

# Map TimeFrame enum to MT5 constants
TIMEFRAME_MAP = {}
if MT5_AVAILABLE:
    TIMEFRAME_MAP = {
        TimeFrame.M1: mt5.TIMEFRAME_M1,
        TimeFrame.M5: mt5.TIMEFRAME_M5,
        TimeFrame.M15: mt5.TIMEFRAME_M15,
        TimeFrame.M30: mt5.TIMEFRAME_M30,
        TimeFrame.H1: mt5.TIMEFRAME_H1,
        TimeFrame.H4: mt5.TIMEFRAME_H4,
        TimeFrame.D1: mt5.TIMEFRAME_D1,
    }


class MT5MarketDataProvider(IMarketDataProvider):
    """Production MT5 Data Provider with safety checks and error handling."""

    def __init__(self, terminal_path: Optional[str] = None):
        self.terminal_path = terminal_path or settings.MT5_PATH
        self._connected = False

    def connect(self) -> bool:
        """Initializes MT5 terminal connection safely."""
        if not MT5_AVAILABLE:
            logger.warning("MetaTrader5 python library is not available.")
            self._connected = False
            return False

        try:
            init_ok = mt5.initialize(path=self.terminal_path) if self.terminal_path else mt5.initialize()
            if not init_ok:
                err = mt5.last_error()
                logger.error(f"MT5 initialization failed: {err}")
                self._connected = False
                return False

            self._connected = True
            term_info = mt5.terminal_info()
            logger.info(f"Connected to MT5 terminal at {getattr(term_info, 'path', 'N/A')}")
            return True
        except Exception as e:
            logger.error(f"Exception during MT5 connect: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Shuts down MT5 terminal connection cleanly."""
        if MT5_AVAILABLE and self._connected:
            try:
                mt5.shutdown()
            except Exception as e:
                logger.warning(f"Error during MT5 shutdown: {e}")
            self._connected = False
            logger.info("MT5 connection closed.")

    def is_connected(self) -> bool:
        """Verifies active terminal and broker connection."""
        if not MT5_AVAILABLE or not self._connected:
            return False
        try:
            term = mt5.terminal_info()
            return bool(term and term.connected)
        except Exception:
            return False

    def get_account_info(self) -> Optional[AccountDetails]:
        """Reads account status and verifies Demo mode without leaking credentials."""
        if not self.is_connected():
            return None

        acc = mt5.account_info()
        if not acc:
            return None

        return AccountDetails(
            login_masked=mask_login(acc.login),
            server=acc.server,
            currency=acc.currency,
            leverage=acc.leverage,
            balance=acc.balance,
            equity=acc.equity,
            margin=acc.margin,
            free_margin=acc.margin_free,
            trade_mode=acc.trade_mode,
            is_demo=(acc.trade_mode == MT5_ACCOUNT_TRADE_MODE_DEMO),
        )

    def get_symbols(self) -> List[str]:
        """Retrieves list of all available symbol names."""
        if not self.is_connected():
            return []
        symbols = mt5.symbols_get()
        if not symbols:
            return []
        return [s.name for s in symbols]

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Retrieves contract details, tick value, and volume constraints."""
        if not self.is_connected():
            return None

        info = mt5.symbol_info(symbol)
        if not info:
            return None

        return SymbolInfo(
            name=info.name,
            digits=info.digits,
            point=info.point,
            spread=info.spread,
            tick_size=info.trade_tick_size,
            tick_value=info.trade_tick_value,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            trade_contract_size=info.trade_contract_size,
            currency_base=info.currency_base,
            currency_profit=info.currency_profit,
            currency_margin=info.currency_margin,
        )

    def get_current_quote(self, symbol: str, max_staleness_seconds: int = 120) -> Optional[Quote]:
        """Gets current Bid and Ask quote with staleness validation."""
        if not self.is_connected():
            return None

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None

        tick_dt = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        age_seconds = (now_utc - tick_dt).total_seconds()
        is_stale = age_seconds > max_staleness_seconds

        spread_points = int(round((tick.ask - tick.bid) / (mt5.symbol_info(symbol).point or 0.0001)))

        return Quote(
            symbol=symbol,
            time=tick_dt,
            bid=tick.bid,
            ask=tick.ask,
            spread_points=spread_points,
            is_stale=is_stale,
        )

    def get_historical_candles(
        self,
        symbol: str,
        timeframe: TimeFrame,
        count: int = 500,
        start_pos: int = 0
    ) -> List[Candle]:
        """Fetches historical candles chronologically."""
        if not self.is_connected():
            return []

        mt5_tf = TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_M15)
        # start_pos=0 includes currently open bar as the last item; start_pos=1 returns only closed bars
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, start_pos, count)
        if rates is None or len(rates) == 0:
            return []

        candles = []
        total = len(rates)
        for i, r in enumerate(rates):
            is_closed = True if (start_pos > 0 or i < total - 1) else False
            candles.append(Candle(
                time=datetime.fromtimestamp(r['time'], tz=timezone.utc),
                open=float(r['open']),
                high=float(r['high']),
                low=float(r['low']),
                close=float(r['close']),
                tick_volume=int(r['tick_volume']),
                spread=int(r['spread']),
                is_closed=is_closed,
            ))
        return candles

    def get_historical_dataframe(
        self,
        symbol: str,
        timeframe: TimeFrame,
        count: int = 500
    ) -> pd.DataFrame:
        """Returns clean pandas DataFrame with ascending DatetimeIndex."""
        candles = self.get_historical_candles(symbol, timeframe, count=count, start_pos=1)
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "tick_volume", "spread"])

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
        df.sort_index(ascending=True, inplace=True)
        return df
