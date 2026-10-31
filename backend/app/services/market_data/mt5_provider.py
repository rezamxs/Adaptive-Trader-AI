"""MetaTrader 5 Market Data Provider implementation (Read-Only).
Handles MT5 terminal connection, symbol specs, candles, and staleness checks.
"""
import threading
import math
from datetime import datetime, timedelta, timezone
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
from backend.app.services.market_data.time_conversion import MT5TimeConverter


# Global re-entrant lock to serialize all C-extension MetaTrader 5 API calls safely
_mt5_lock = threading.RLock()

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
        # Broker-clock offset in seconds; determined at connect() and ONLY
        # non-zero when the connected account's server is the configured one.
        self._offset_seconds = 0.0
        self.time_converter = MT5TimeConverter(
            max_staleness_seconds=settings.WORKER_STALE_TICK_SECONDS
        )

    def _determine_offset_seconds(self) -> float:
        """Server-gated broker-clock offset in seconds.

        settings.MT5_SERVER_TIME_OFFSET_HOURS is the MEASURED offset of the
        configured server (YourBroker-MT5 stamps server-time as epoch). It applies
        ONLY when the connected account's server equals settings.MT5_SERVER;
        any other server runs raw-epoch (UTC) semantics with zero offset, and
        an unreadable account fails closed to zero.
        """
        hours = float(settings.MT5_SERVER_TIME_OFFSET_HOURS)
        if hours == 0.0:
            return 0.0
        acc = mt5.account_info()
        if not acc or acc.server != settings.MT5_SERVER:
            logger.warning(
                "MT5 server-time offset (%.2fh) is configured but the connected "
                "server is not '%s'; offset NOT applied (raw-epoch semantics).",
                hours, settings.MT5_SERVER)
            return 0.0
        logger.info(
            "MT5 server '%s' matches configuration; applying measured broker-clock "
            "offset of %.2fh to all quote/bar/deal timestamps.", settings.MT5_SERVER, hours)
        return hours * 3600.0

    def connect(self) -> bool:
        """Initializes MT5 terminal connection safely with thread synchronization."""
        with _mt5_lock:
            if not MT5_AVAILABLE:
                logger.warning("MetaTrader5 python library is not available.")
                self._connected = False
                self._offset_seconds = 0.0
                return False

            try:
                init_ok = mt5.initialize(path=self.terminal_path) if self.terminal_path else mt5.initialize()
                if not init_ok:
                    err = mt5.last_error()
                    logger.error(f"MT5 initialization failed: {err}")
                    self._connected = False
                    self._offset_seconds = 0.0
                    return False

                self._connected = True
                self._offset_seconds = self._determine_offset_seconds()
                term_info = mt5.terminal_info()
                logger.info(f"Connected to MT5 terminal at {getattr(term_info, 'path', 'N/A')}")

                # MT5 Python API timestamps are Unix epoch seconds. For the
                # configured server they carry broker server-time; the measured
                # server-gated offset above corrects them to true UTC.

                return True
            except Exception as e:
                logger.error(f"Exception during MT5 connect: {e}")
                self._connected = False
                self._offset_seconds = 0.0
                return False

    def disconnect(self) -> None:
        """Shuts down MT5 terminal connection cleanly."""
        with _mt5_lock:
            if MT5_AVAILABLE and self._connected:
                try:
                    mt5.shutdown()
                except Exception as e:
                    logger.warning(f"Error during MT5 shutdown: {e}")
                self._connected = False
                logger.info("MT5 connection closed.")

    def is_connected(self) -> bool:
        """Verifies active terminal and broker connection."""
        with _mt5_lock:
            if not MT5_AVAILABLE or not self._connected:
                return False
            try:
                term = mt5.terminal_info()
                return bool(term and term.connected)
            except Exception:
                return False

    def get_account_info(self) -> Optional[AccountDetails]:
        """Reads account status and verifies Demo mode without leaking credentials."""
        with _mt5_lock:
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

    def get_open_positions_count(self) -> Optional[int]:
        """Return account-wide open positions count, or None if MT5 cannot verify it."""
        with _mt5_lock:
            if not self.is_connected():
                return None
            positions = mt5.positions_get()
            if positions is None:
                logger.error(f"MT5 positions_get failed: {mt5.last_error()}")
                return None
            return len(positions)

    def get_open_positions(self) -> Optional[List[Any]]:
        """Return the raw open-position objects (read-only) for grid-risk
        computation, or None if MT5 cannot verify them (fail closed)."""
        with _mt5_lock:
            if not self.is_connected():
                return None
            positions = mt5.positions_get()
            if positions is None:
                logger.error(f"MT5 positions_get failed: {mt5.last_error()}")
                return None
            return list(positions)

    def get_daily_loss_percent(self) -> Optional[float]:
        """Estimate current-day equity drawdown from UTC-day deal cash flows.

        Any cash movement reported by MT5 is included when reconstructing the
        day's starting balance. If account or deal history cannot be verified,
        return None so risk checks can fail closed.
        """
        with _mt5_lock:
            if not self.is_connected():
                return None
            account = mt5.account_info()
            if account is None:
                return None

            now_utc = datetime.now(timezone.utc)
            day_start_utc = datetime(
                now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc
            )
            # Deal stamps carry the broker clock for the configured server, so
            # the true UTC day must be expressed in stamp space (+ offset).
            from_dt = day_start_utc + timedelta(seconds=self._offset_seconds)
            to_dt = now_utc + timedelta(seconds=self._offset_seconds)
            deals = mt5.history_deals_get(from_dt, to_dt)
            if deals is None:
                logger.error(f"MT5 deal history unavailable for daily drawdown: {mt5.last_error()}")
                return None

            day_cash_flow = 0.0
            excluded_deals = 0
            for deal in deals:
                # Stamp validation: a deal whose timestamp is unparseable or
                # falls outside the corrected UTC day makes the starting-
                # balance reconstruction ambiguous (its cash effect IS in the
                # current balance). Fail closed instead of guessing.
                try:
                    deal_dt = datetime.fromtimestamp(
                        int(deal.time) - self._offset_seconds, tz=timezone.utc)
                except (TypeError, ValueError, OverflowError, OSError):
                    excluded_deals += 1
                    continue
                if deal_dt < day_start_utc or deal_dt > now_utc + timedelta(seconds=5.0):
                    excluded_deals += 1
                    continue
                day_cash_flow += (
                    float(getattr(deal, "profit", 0.0) or 0.0)
                    + float(getattr(deal, "commission", 0.0) or 0.0)
                    + float(getattr(deal, "swap", 0.0) or 0.0)
                    + float(getattr(deal, "fee", 0.0) or 0.0)
                )
            if excluded_deals:
                logger.warning(
                    "Daily-loss reconstruction refused: %d MT5 deal(s) carry invalid "
                    "or out-of-window timestamps.", excluded_deals)
                return None

            start_balance = float(account.balance) - day_cash_flow
            equity = float(account.equity)
            if (
                not math.isfinite(start_balance)
                or not math.isfinite(equity)
                or start_balance <= 0.0
            ):
                return None
            return max(0.0, (start_balance - equity) / start_balance * 100.0)

    def calculate_order_margin(
        self, symbol: str, direction: str, volume: float, price: float
    ) -> Optional[float]:
        """Ask MT5 for broker-calculated initial margin in the account deposit currency."""
        with _mt5_lock:
            if not self.is_connected():
                return None
            if direction not in {"BUY", "SELL"} or volume <= 0 or price <= 0:
                return None
            order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
            margin = mt5.order_calc_margin(order_type, symbol, float(volume), float(price))
            if margin is None:
                logger.warning(f"MT5 margin calculation unavailable for {symbol}: {mt5.last_error()}")
                return None
            value = float(margin)
            if not math.isfinite(value) or value < 0.0:
                return None
            return value

    def get_symbols(self) -> List[str]:
        """Retrieves list of all available symbol names."""
        with _mt5_lock:
            if not self.is_connected():
                return []
            symbols = mt5.symbols_get()
            if not symbols:
                return []
            return [s.name for s in symbols]

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Retrieves contract details, tick value, and volume constraints."""
        with _mt5_lock:
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

    def get_last_tick(self, symbol: str) -> Optional[Any]:
        """Directly retrieves the latest raw tick from MT5 under lock."""
        with _mt5_lock:
            if not self.is_connected():
                return None
            return mt5.symbol_info_tick(symbol)

    def get_current_quote(self, symbol: str, max_staleness_seconds: int = 120) -> Optional[Quote]:
        """Gets Bid/Ask with UTC epoch conversion and staleness validation."""
        with _mt5_lock:
            if not self.is_connected():
                return None

            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return None

            now_utc = datetime.now(timezone.utc)

            # Broker-clock correction: for the configured server the epoch
            # carries broker server-time; subtract the measured offset before
            # UTC interpretation. With offset 0 this is the identity.
            raw_epoch = int(tick.time)
            raw_dt = datetime.fromtimestamp(raw_epoch, tz=timezone.utc)
            try:
                tick_dt = datetime.fromtimestamp(raw_epoch - self._offset_seconds, tz=timezone.utc)
            except (TypeError, ValueError, OverflowError, OSError) as exc:
                logger.warning(f"Invalid MT5 tick timestamp for {symbol}: {exc}")
                return None

            val_res = self.time_converter.validate_timestamp(
                tick_dt,
                now_utc=now_utc,
                max_staleness_seconds=max_staleness_seconds,
            )

            # Live-consistency guard: if the RAW stamp would already be valid,
            # the configured offset is contradicted by this very tick — fail
            # closed instead of accepting ambiguous data.
            if self._offset_seconds and val_res.is_valid:
                raw_res = self.time_converter.validate_timestamp(
                    raw_dt,
                    now_utc=now_utc,
                    max_staleness_seconds=max_staleness_seconds,
                )
                if raw_res.is_valid:
                    logger.error(
                        "MT5 offset ambiguity for %s: raw stamp AND corrected stamp "
                        "are both fresh (raw age %.1fs, corrected age %.1fs); "
                        "refusing to accept ambiguous data.",
                        symbol, raw_res.age_seconds, val_res.age_seconds,
                    )
                    return None

            # Freshness is determined by timestamp validation, not broker timezone guesses.
            is_stale = not val_res.is_valid

            sym_info = mt5.symbol_info(symbol)
            if sym_info is None:
                logger.warning(f"MT5 symbol specifications are unavailable for {symbol}.")
                return None
            try:
                point = float(sym_info.point)
                bid = float(tick.bid)
                ask = float(tick.ask)
            except (TypeError, ValueError, OverflowError):
                return None
            if (
                not math.isfinite(point) or point <= 0
                or not math.isfinite(bid) or not math.isfinite(ask)
                or bid <= 0 or ask < bid
            ):
                logger.warning(f"MT5 returned invalid quote/specification values for {symbol}.")
                return None
            spread_points = int(round((ask - bid) / point))

            return Quote(
                symbol=symbol,
                time=tick_dt,
                bid=bid,
                ask=ask,
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
        """Fetches historical candles chronologically using MT5 UTC epoch timestamps."""
        with _mt5_lock:
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
                try:
                    candle_time = datetime.fromtimestamp(
                        int(r['time']) - self._offset_seconds, tz=timezone.utc)
                except (TypeError, ValueError, OverflowError, OSError) as exc:
                    logger.warning(f"Skipping candle with invalid MT5 timestamp for {symbol}: {exc}")
                    continue
                candles.append(Candle(
                    time=candle_time,
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
