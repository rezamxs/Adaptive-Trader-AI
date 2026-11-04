"""Stage-4 regression tests: server-gated MT5 broker-clock offset (the configured broker).

The YourBroker-MT5 server stamps broker server-time as epoch (verified live
2026-10-28: ticks/bars exactly +3h vs system UTC). The offset must:
- apply ONLY when the connected account's server == settings.MT5_SERVER;
- fail closed when the wrong offset is configured, when the raw stamp would
  already be valid (ambiguity), or when the corrected stamp is stale/future;
- shift the daily-loss deal window and historical candle times accordingly.
All tests are unit-level stubs of the MT5 C extension; none is a live run.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from backend.app.core.config import settings
from backend.app.services.market_data import mt5_provider as mt5_provider_module
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.market_data.mt5_provider import MT5_AVAILABLE, MT5MarketDataProvider

# Item 24 (CI): these tests monkeypatch the MT5 C-extension module and the
# provider's connect() short-circuits to False when the Windows-only
# MetaTrader5 package is absent — the whole module skips on ubuntu CI.
pytestmark = pytest.mark.skipif(
    not MT5_AVAILABLE, reason="requires the Windows-only MetaTrader5 package"
)

XCHIEF = "YourBroker-MT5"
OFFSET_SECONDS = 3 * 3600


class _StubMT5:
    """Minimal MT5 C-extension stub: server name + a future-stamped tick."""

    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388
    TIMEFRAME_D1 = 16408

    def __init__(self, server: str, tick_epoch: int, rates=None, deals=None):
        self._server = server
        self._tick_epoch = tick_epoch
        self._rates = rates or []
        self._deals = deals or []
        self.deal_window = None

    def initialize(self, path=None):
        return True

    def shutdown(self):
        pass

    def last_error(self):
        return (1, "Success")

    def terminal_info(self):
        return SimpleNamespace(connected=True, trade_allowed=True, path="stub")

    def account_info(self):
        return SimpleNamespace(login=12345678, server=self._server,
                               currency="USD", leverage=100, balance=10000.0,
                               equity=10000.0, margin=0.0, margin_free=10000.0,
                               trade_mode=0)

    def symbol_info(self, symbol):
        return SimpleNamespace(
            name=symbol, digits=2, point=0.01, spread=17,
            trade_tick_size=0.01, trade_tick_value=1.0,
            volume_min=0.01, volume_max=100.0, volume_step=0.01,
            trade_contract_size=100.0, currency_base="USD",
            currency_profit="USD", currency_margin="USD", trade_mode=4,
            visible=True, select=True)

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(time=self._tick_epoch, time_msc=self._tick_epoch * 1000,
                               bid=4045.03, ask=4045.20, last=0.0, volume=0.0)

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        return self._rates[:count] if self._rates else None

    def history_deals_get(self, from_dt, to_dt):
        self.deal_window = (from_dt, to_dt)
        return self._deals


def _now():
    return datetime.now(timezone.utc)


def _make_provider(monkeypatch, server, tick_epoch, offset_hours, rates=None, deals=None):
    stub = _StubMT5(server, tick_epoch, rates=rates, deals=deals)
    monkeypatch.setattr(settings, "MT5_SERVER_TIME_OFFSET_HOURS", offset_hours)
    monkeypatch.setattr(settings, "MT5_SERVER", XCHIEF)
    monkeypatch.setattr(mt5_provider_module, "mt5", stub)
    provider = MT5MarketDataProvider()
    assert provider.connect() is True
    return provider, stub


def test_offset_applies_only_to_configured_server(monkeypatch):
    """A +3h-stamped fresh tick is accepted ONLY for the configured server."""
    now = _now()
    raw_epoch = int((now + timedelta(seconds=OFFSET_SECONDS - 10)).timestamp())

    provider, _ = _make_provider(monkeypatch, XCHIEF, raw_epoch, offset_hours=3.0)
    quote = provider.get_current_quote("XAUUSD")
    assert quote is not None, "corrected stamp (age -10s+3h) must be accepted for the configured broker"
    assert quote.is_stale is False
    expected_time = (now - timedelta(seconds=10)).replace(microsecond=0)
    drift = abs((quote.time - expected_time).total_seconds())
    assert drift <= 2.0, f"quote.time must equal the corrected UTC stamp, drift {drift}s"

    other, _ = _make_provider(monkeypatch, "OtherBroker", raw_epoch, offset_hours=3.0)
    other_quote = other.get_current_quote("XAUUSD")
    assert other_quote is not None and other_quote.is_stale is True, (
        "another server must run raw-epoch semantics: the +3h stamp stays "
        "future-stale and can never be accepted as fresh")


def test_wrong_offset_fails_closed(monkeypatch):
    """Configuring 2h against a +3h broker leaves the corrected stamp in the
    future -> the quote is flagged stale and can never be treated as fresh."""
    now = _now()
    raw_epoch = int((now + timedelta(seconds=OFFSET_SECONDS - 10)).timestamp())
    provider, _ = _make_provider(monkeypatch, XCHIEF, raw_epoch, offset_hours=2.0)
    quote = provider.get_current_quote("XAUUSD")
    assert quote is not None and quote.is_stale is True


def test_ambiguous_raw_valid_tick_fails_closed(monkeypatch):
    """If the RAW stamp would already be fresh, the offset is contradicted by
    this very tick -> ambiguous -> the quote is flagged stale (fail closed)."""
    now = _now()
    raw_epoch = int((now - timedelta(seconds=5)).timestamp())
    provider, _ = _make_provider(monkeypatch, XCHIEF, raw_epoch, offset_hours=3.0)
    quote = provider.get_current_quote("XAUUSD")
    assert quote is not None and quote.is_stale is True


def test_stale_after_correction_rejected(monkeypatch):
    """A corrected stamp older than the staleness threshold is flagged stale."""
    now = _now()
    raw_epoch = int((now + timedelta(seconds=OFFSET_SECONDS + 300)).timestamp())
    provider, _ = _make_provider(monkeypatch, XCHIEF, raw_epoch, offset_hours=3.0)
    quote = provider.get_current_quote("XAUUSD")
    assert quote is not None and quote.is_stale is True


def test_zero_offset_keeps_raw_utc_semantics(monkeypatch):
    """offset=0 restores the legacy contract: raw epoch is treated as UTC —
    a fresh raw tick is accepted and a future raw tick is flagged stale."""
    now = _now()
    fresh_raw = int((now - timedelta(seconds=10)).timestamp())
    provider, _ = _make_provider(monkeypatch, XCHIEF, fresh_raw, offset_hours=0.0)
    quote = provider.get_current_quote("XAUUSD")
    assert quote is not None and quote.is_stale is False

    future_raw = int((now + timedelta(minutes=10)).timestamp())
    provider2, _ = _make_provider(monkeypatch, XCHIEF, future_raw, offset_hours=0.0)
    quote2 = provider2.get_current_quote("XAUUSD")
    assert quote2 is not None and quote2.is_stale is True


def test_daily_loss_window_shifted_by_offset(monkeypatch):
    """The deal-history window for the daily loss must be expressed in the
    broker's stamp space (UTC day + offset); offset=0 keeps the raw window."""
    now = _now()
    raw_epoch = int((now + timedelta(seconds=OFFSET_SECONDS - 10)).timestamp())
    provider, stub = _make_provider(monkeypatch, XCHIEF, raw_epoch, offset_hours=3.0)
    loss = provider.get_daily_loss_percent()
    assert loss == 0.0  # no deals in window
    from_dt, to_dt = stub.deal_window
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    assert abs((from_dt - (day_start + timedelta(seconds=OFFSET_SECONDS))).total_seconds()) <= 2.0
    assert abs((to_dt - (now + timedelta(seconds=OFFSET_SECONDS))).total_seconds()) <= 5.0

    provider0, stub0 = _make_provider(monkeypatch, XCHIEF, raw_epoch, offset_hours=0.0)
    provider0.get_daily_loss_percent()
    from_dt0, _ = stub0.deal_window
    assert abs((from_dt0 - day_start).total_seconds()) <= 2.0


def test_historical_candle_times_corrected(monkeypatch):
    """Candle epochs carry the same broker shift and must be corrected."""
    now = _now()
    bar_seconds = int(now.replace(second=0, microsecond=0).timestamp()) + OFFSET_SECONDS
    rates = [
        {"time": bar_seconds - 60, "open": 4044.0, "high": 4045.0,
         "low": 4043.0, "close": 4044.5, "tick_volume": 10, "spread": 17},
        {"time": bar_seconds, "open": 4044.5, "high": 4046.0,
         "low": 4044.0, "close": 4045.0, "tick_volume": 12, "spread": 17},
    ]
    provider, _ = _make_provider(monkeypatch, XCHIEF, bar_seconds, offset_hours=3.0, rates=rates)
    candles = provider.get_historical_candles("XAUUSD", TimeFrame.M1, count=2, start_pos=0)
    assert len(candles) == 2
    expected_last = datetime.fromtimestamp(bar_seconds - OFFSET_SECONDS, tz=timezone.utc)
    assert candles[-1].time == expected_last, "bar time must be corrected to true UTC"
    assert candles[-1].is_closed is False  # pos-0 last bar is the open bar


def test_config_rejects_implausible_offset():
    from backend.app.core.config import Settings
    with pytest.raises(ValueError):
        Settings(MT5_SERVER_TIME_OFFSET_HOURS=99.0)


# ---------------------------------------------------------------------------
# Item 7: daily-loss deal stamp validation — future/out-of-window/unparseable
# deal stamps make the reconstruction ambiguous -> fail closed (None).
# ---------------------------------------------------------------------------

def _deal(epoch_seconds, profit=-50.0):
    from types import SimpleNamespace as _NS
    return _NS(time=epoch_seconds, profit=profit, commission=0.0, swap=0.0, fee=0.0)


def test_daily_loss_counts_in_window_deals(monkeypatch):
    now = _now()
    # Deal placed at the midpoint of the elapsed UTC day so it stays inside
    # the [day_start, now] window regardless of when the suite runs (the
    # old fixed "1h ago" placement crossed midnight at ~00:xx UTC and was
    # correctly rejected by the fail-closed stamp validation).
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = (now - day_start).total_seconds()
    deal_true_utc = day_start + timedelta(seconds=elapsed / 2.0)
    deal_epoch = int((deal_true_utc + timedelta(seconds=OFFSET_SECONDS)).timestamp())
    provider, _ = _make_provider(
        monkeypatch, XCHIEF, int((now + timedelta(seconds=OFFSET_SECONDS - 10)).timestamp()),
        offset_hours=3.0, deals=[_deal(deal_epoch, profit=-50.0)])
    loss = provider.get_daily_loss_percent()
    # start_balance = 10000 - (-50) = 10050; equity 10000 -> loss ~0.4975%
    assert loss == pytest.approx(0.4975, abs=0.01), f"got {loss}"


def test_daily_loss_is_account_wide_manual_deals_included(monkeypatch):
    """M6 scope pin (contract test on synthetic deals, not a real-account
    claim): the daily-loss reconstruction includes EVERY in-window deal —
    manual trades and bot trades alike, plus balance-type cash movements —
    because it filters by neither comment nor deal type. This matches
    docs/strategy-specification.md Group D §1 (account-level drawdown). If
    the policy ever changes to bot-only, this test must be changed
    deliberately with it."""
    now = _now()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = (now - day_start).total_seconds()
    mid_true_utc = day_start + timedelta(seconds=elapsed / 2.0)
    mid_epoch = int((mid_true_utc + timedelta(seconds=OFFSET_SECONDS)).timestamp())
    manual_deal = _deal(mid_epoch, profit=-40.0)
    manual_deal.comment = "manual trade by operator"  # no "Demo AI: " prefix
    bot_deal = _deal(mid_epoch + 60, profit=-60.0)
    bot_deal.comment = "Demo AI: ord_scope_test"
    provider, _ = _make_provider(
        monkeypatch, XCHIEF, int((now + timedelta(seconds=OFFSET_SECONDS - 10)).timestamp()),
        offset_hours=3.0, deals=[manual_deal, bot_deal])
    loss = provider.get_daily_loss_percent()
    # start_balance = 10000 - (-100) = 10100; equity 10000 -> loss ~0.990%
    assert loss == pytest.approx(0.9901, abs=0.01), (
        f"manual + bot deals must both count (account-wide), got {loss}")


def test_daily_loss_deal_exactly_at_day_start_is_counted(monkeypatch):
    """Boundary: a deal stamped exactly AT the UTC day start is in-window."""
    now = _now()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    deal_epoch = int((day_start + timedelta(seconds=OFFSET_SECONDS)).timestamp())
    provider, _ = _make_provider(
        monkeypatch, XCHIEF, int((now + timedelta(seconds=OFFSET_SECONDS - 10)).timestamp()),
        offset_hours=3.0, deals=[_deal(deal_epoch, profit=-25.0)])
    loss = provider.get_daily_loss_percent()
    assert loss == pytest.approx(0.2488, abs=0.01), f"got {loss}"


def test_daily_loss_future_stamped_deal_fails_closed(monkeypatch):
    """A deal stamped AFTER the corrected window (tomorrow in stamp space) is
    ambiguous money -> the whole reconstruction must return None."""
    now = _now()
    in_window = int((now - timedelta(hours=1) + timedelta(seconds=OFFSET_SECONDS)).timestamp())
    future_deal = int((now + timedelta(days=2) + timedelta(seconds=OFFSET_SECONDS)).timestamp())
    provider, _ = _make_provider(
        monkeypatch, XCHIEF, int((now + timedelta(seconds=OFFSET_SECONDS - 10)).timestamp()),
        offset_hours=3.0,
        deals=[_deal(in_window), _deal(future_deal, profit=-500.0)])
    assert provider.get_daily_loss_percent() is None


def test_daily_loss_out_of_window_deal_fails_closed(monkeypatch):
    """A deal from before the UTC day start (yesterday) must also fail closed."""
    now = _now()
    yesterday_deal = int((now - timedelta(days=1) + timedelta(seconds=OFFSET_SECONDS)).timestamp())
    provider, _ = _make_provider(
        monkeypatch, XCHIEF, int((now + timedelta(seconds=OFFSET_SECONDS - 10)).timestamp()),
        offset_hours=3.0, deals=[_deal(yesterday_deal, profit=-100.0)])
    assert provider.get_daily_loss_percent() is None


def test_daily_loss_unparseable_stamp_fails_closed(monkeypatch):
    now = _now()
    bad = _deal(int((now - timedelta(hours=1) + timedelta(seconds=OFFSET_SECONDS)).timestamp()))
    bad.time = "not-a-number"
    provider, _ = _make_provider(
        monkeypatch, XCHIEF, int((now + timedelta(seconds=OFFSET_SECONDS - 10)).timestamp()),
        offset_hours=3.0, deals=[bad])
    assert provider.get_daily_loss_percent() is None


def test_config_rejects_non_finite_offsets():
    """NaN bypasses any `abs(v) > limit` comparison (all NaN comparisons are
    False) — the validator must reject NaN and both infinities explicitly."""
    from backend.app.core.config import Settings
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            Settings(MT5_SERVER_TIME_OFFSET_HOURS=bad)


def test_config_range_and_valid_offset():
    from backend.app.core.config import Settings
    with pytest.raises(ValueError):
        Settings(MT5_SERVER_TIME_OFFSET_HOURS=12.1)
    with pytest.raises(ValueError):
        Settings(MT5_SERVER_TIME_OFFSET_HOURS=-13.0)
    s = Settings(MT5_SERVER_TIME_OFFSET_HOURS=3.0)
    assert s.MT5_SERVER_TIME_OFFSET_HOURS == 3.0
