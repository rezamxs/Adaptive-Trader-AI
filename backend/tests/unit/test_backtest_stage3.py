"""Stage-3 backtest correctness regression tests (audit M1 + M2 + costs).

Deterministic synthetic bars, no real market data: these tests pin the exact
exit prices, drawdown accounting, and commission contract of the engine.
"""
import pandas as pd
import pytest

from backend.app.services.backtest.engine import RealisticBacktestEngine


def _engine(**overrides):
    defaults = dict(
        symbol="XAUUSD", initial_balance=10000.0, spread_points=0,
        slippage_points=0, commission_per_lot=0.0, point_value=0.01,
        contract_size=100.0,
    )
    defaults.update(overrides)
    return RealisticBacktestEngine(**defaults)


def _idx(n):
    return pd.date_range("2026-10-12 10:00:00", periods=n, freq="min", tz="UTC")


def test_gap_through_stop_buy_fills_at_open():
    """BUY: a bar that OPENS beyond the stop must fill at the open, not at the stop."""
    idx = _idx(3)
    df = pd.DataFrame({
        "open": [2000.0, 2000.0, 1900.0],
        "high": [2001.0, 2001.0, 1905.0],
        "low": [1999.0, 1999.0, 1890.0],
        "close": [2000.0, 2000.0, 1900.0],
        "atr_14": [2.0, 2.0, 2.0],
    }, index=idx)
    result = _engine().run(df, pd.Series([1, 0, 0], index=idx))

    trade = result["trades"][0]
    # entry 2000, SL 1998; bar 2 opens at 1900 — the stop fills at the open.
    assert trade["exit_reason"] == "STOP_LOSS"
    assert trade["exit_price"] == 1900.0, (
        f"gap-through must fill at open 1900, got {trade['exit_price']}")
    # 0.5 lots (risk $100 / $200-per-lot), 100-point gap: gross = -100 * 100 * 0.5
    assert trade["gross_profit"] == -5000.0


def test_gap_through_stop_sell_fills_at_open():
    """SELL: a bar whose Ask open is already beyond the stop fills at that open."""
    idx = _idx(3)
    df = pd.DataFrame({
        "open": [2000.0, 2000.0, 2100.0],
        "high": [2001.0, 2001.0, 2110.0],
        "low": [1999.0, 1999.0, 2095.0],
        "close": [2000.0, 2000.0, 2100.0],
        "atr_14": [2.0, 2.0, 2.0],
    }, index=idx)
    result = _engine().run(df, pd.Series([-1, 0, 0], index=idx))

    trade = result["trades"][0]
    # entry 2000 (Bid), SL 2002; bar 2 Ask open = 2100 — fill there, not at 2002.
    assert trade["exit_reason"] == "STOP_LOSS"
    assert trade["exit_price"] == 2100.0, (
        f"gap-through must fill at ask open 2100, got {trade['exit_price']}")
    assert trade["gross_profit"] == -5000.0


def test_floating_drawdown_captures_deep_unrealized_loss():
    """A trade that dips deep underwater before closing in profit must register
    that floating loss in max drawdown — closed-trades-only equity hides it."""
    idx = _idx(4)
    df = pd.DataFrame({
        "open": [2000.0, 2000.0, 1990.0, 1990.0],
        "high": [2000.0, 2000.0, 1995.0, 2160.0],
        "low": [2000.0, 2000.0, 1950.0, 1985.0],
        "close": [2000.0, 2000.0, 1985.0, 2100.0],
        "atr_14": [100.0, 100.0, 100.0, 100.0],
    }, index=idx)
    result = _engine().run(df, pd.Series([1, 0, 0, 0], index=idx))

    assert result["trades"][0]["exit_reason"] == "TAKE_PROFIT"
    assert result["trades"][0]["net_profit"] > 0
    # Worst floating mark on bar 2: (1950 - 2000) * 100 * 0.01 lots = -$50
    assert result["metrics"]["max_drawdown_amount"] == 50.0, (
        f"floating loss must register in max drawdown, "
        f"got {result['metrics']['max_drawdown_amount']}")
    assert result["metrics"]["max_drawdown_percent"] == 0.5
    assert len(result["metrics"]["equity_curve"]) >= 3, (
        "equity curve must include the floating mark between START and close")


def test_commission_is_charged_per_side_and_documented():
    """commission_per_lot is a per-side rate: charged on entry AND on exit.
    The trade's commission field reports the round-turn total."""
    idx = _idx(3)
    df = pd.DataFrame({
        "open": [2000.0, 2000.0, 2000.0],
        "high": [2001.0, 2001.0, 2001.0],
        "low": [1999.0, 1999.0, 1999.0],
        "close": [2000.0, 2000.0, 2000.0],
        "atr_14": [2.0, 2.0, 2.0],
    }, index=idx)
    result = _engine(commission_per_lot=7.0).run(df, pd.Series([1, 0, 0], index=idx))

    trade = result["trades"][0]
    # 0.5 lots, flat trade, gross 0: two sides * $7 * 0.5 = $7.00 total cost.
    assert trade["commission"] == 7.0, (
        f"round-turn commission must be 2 * per-side * lots, got {trade['commission']}")
    assert trade["net_profit"] == -7.0
    assert result["metrics"]["final_balance"] == 9993.0
    assert result["assumptions"]["commission_model"] == "PER_SIDE"


def test_last_bar_tp_sl_managed_before_end_of_data():
    """TP/SL must also be evaluated on the FINAL bar; only positions that hit
    neither barrier are liquidated as END_OF_DATA."""
    idx = _idx(3)
    df = pd.DataFrame({
        "open": [2000.0, 2000.0, 2000.0],
        "high": [2001.0, 2001.0, 2005.0],
        "low": [1999.0, 1999.0, 1999.0],
        "close": [2000.0, 2000.0, 2000.0],
        "atr_14": [2.0, 2.0, 2.0],
    }, index=idx)
    result = _engine().run(df, pd.Series([1, 0, 0], index=idx))

    trade = result["trades"][0]
    assert trade["exit_reason"] == "TAKE_PROFIT", (
        f"TP on the final bar must close the trade, got {trade['exit_reason']}")
    assert trade["exit_price"] == 2003.0
    assert trade["gross_profit"] == 150.0  # 3.0 * 100 * 0.5 lots


def test_floating_drawdown_marks_the_exit_bar():
    """The dip that happens on the TP-exit bar itself must also register in the
    drawdown — the position was exposed to that bar's adverse extreme before
    the limit fill."""
    idx = _idx(3)
    df = pd.DataFrame({
        "open": [2000.0, 2000.0, 2000.0],
        "high": [2000.0, 2000.0, 2020.0],
        "low": [2000.0, 2000.0, 1991.0],
        "close": [2000.0, 2000.0, 2010.0],
        "atr_14": [10.0, 10.0, 10.0],
    }, index=idx)
    result = _engine().run(df, pd.Series([1, 0, 0], index=idx))

    assert result["trades"][0]["exit_reason"] == "TAKE_PROFIT"
    # Adverse mark on the exit bar: (1991 - 2000) * 100 * 0.1 lots = -$90
    assert result["metrics"]["max_drawdown_amount"] == 90.0, (
        f"pre-exit dip on the exit bar must register, "
        f"got {result['metrics']['max_drawdown_amount']}")


def test_floating_peak_before_trough_registers_full_swing():
    """M2 regression (reviewer scenario): on a bar where NO barrier is hit, the
    floating PEAK (at the high) precedes the floating TROUGH (at the low) under
    the conservative OHLC convention, so a same-bar swing of +100 then -10
    around the entry must register a $110 drawdown — not the $10 trough-only
    value. BUY, entry 2000, volume 0.01, ATR=100, SL 1900, TP 3000."""
    idx = _idx(3)
    df = pd.DataFrame({
        "open": [2000.0, 2000.0, 2000.0],
        "high": [2000.0, 2000.0, 2100.0],
        "low": [2000.0, 2000.0, 1990.0],
        "close": [2000.0, 2000.0, 2050.0],
        "atr_14": [100.0, 100.0, 100.0],
    }, index=idx)
    result = _engine().run(df, pd.Series([1, 0, 0], index=idx), tp_atr_mult=10.0)

    trade = result["trades"][0]
    assert trade["exit_reason"] == "END_OF_DATA"  # neither barrier hit on bar 2
    # 0.01 lots: peak (2100-2000)*100*0.01 = +$100 -> 10100, then trough
    # (1990-2000)*100*0.01 = -$10 -> 9990. Peak-to-trough = $110.
    curve = [p["equity"] for p in result["metrics"]["equity_curve"]]
    assert 10100.0 in curve, "floating peak at the bar high must be booked"
    assert 9990.0 in curve, "floating trough at the bar low must be booked"
    assert curve.index(10100.0) < curve.index(9990.0), (
        "conservative OHLC convention: favorable extreme booked before adverse")
    assert result["metrics"]["max_drawdown_amount"] == 110.0, (
        f"peak-to-trough floating swing must register 110, "
        f"got {result['metrics']['max_drawdown_amount']}")
    assert result["assumptions"]["floating_mark_policy"].startswith(
        "FAVORABLE_THEN_ADVERSE_WHILE_OPEN")


def test_end_of_data_marks_use_pre_close_balance():
    """END_OF_DATA regression: the final bar's peak/trough marks must come from
    the PRE-close equity base. Computing them from the post-close balance bakes
    the realized close P&L into both marks (artificial 9991/9970 values).

    BUY entry 2000, volume 0.01, ATR=100, SL 1900, TP 3000; final bar
    High=2001, Low=1980, Close=1990 -> closes at -10 with zero costs. Marks:
    peak 10000+1 = 10001, trough 10000-20 = 9980 -> Max DD exactly $21."""
    idx = _idx(3)
    df = pd.DataFrame({
        "open": [2000.0, 2000.0, 2000.0],
        "high": [2000.0, 2000.0, 2001.0],
        "low": [2000.0, 2000.0, 1980.0],
        "close": [2000.0, 2000.0, 1990.0],
        "atr_14": [100.0, 100.0, 100.0],
    }, index=idx)
    result = _engine().run(df, pd.Series([1, 0, 0], index=idx), tp_atr_mult=10.0)

    trade = result["trades"][0]
    assert trade["exit_reason"] == "END_OF_DATA"
    assert trade["net_profit"] == -10.0  # (1990-2000)*100*0.01, zero costs
    curve = [p["equity"] for p in result["metrics"]["equity_curve"]]
    assert 10001.0 in curve, "peak must be booked from the pre-close base"
    assert 9980.0 in curve, "trough must be booked from the pre-close base"
    assert 9991.0 not in curve and 9970.0 not in curve, (
        "artificial post-close marks must never enter the equity curve")
    assert curve.index(10001.0) < curve.index(9980.0), "favorable extreme first"
    # The final bar's marks are booked EXACTLY ONCE (by END_OF_DATA, not also
    # by the main loop), followed by the closing record.
    final_time = str(idx[-1])
    final_entries = [p["equity"] for p in result["metrics"]["equity_curve"]
                     if p["time"] == final_time]
    assert final_entries == [10001.0, 9980.0, 9990.0], (
        f"final bar must book exactly one peak, one trough, then the close; "
        f"got {final_entries}")
    assert result["metrics"]["max_drawdown_amount"] == 21.0, (
        f"peak-to-trough on the final bar must register 21, "
        f"got {result['metrics']['max_drawdown_amount']}")


def test_gap_through_stop_buy_with_spread_and_slippage():
    """M1 with nonzero costs: BUY stop fills at min(open, SL) - slippage on the
    bid; entry had paid spread+slippage on the ask."""
    idx = _idx(3)
    df = pd.DataFrame({
        "open": [2000.0, 2000.0, 1900.0],
        "high": [2001.0, 2001.0, 1905.0],
        "low": [1999.0, 1999.0, 1890.0],
        "close": [2000.0, 2000.0, 1900.0],
        "atr_14": [2.0, 2.0, 2.0],
    }, index=idx)
    result = _engine(spread_points=20, slippage_points=2).run(
        df, pd.Series([1, 0, 0], index=idx))

    trade = result["trades"][0]
    # entry = 2000 + 0.20 + 0.02 = 2000.22, SL 1998.22; gap open 1900 -> fill
    # at 1900 - 0.02 = 1899.98 (bid exit, spread not re-charged on exit).
    assert trade["entry_price"] == 2000.22
    assert trade["exit_reason"] == "STOP_LOSS"
    assert trade["exit_price"] == pytest.approx(1899.98)
    assert trade["gross_profit"] == pytest.approx(-5012.0)  # -100.24 * 100 * 0.5


def test_gap_through_stop_sell_with_spread_and_slippage():
    """M1 with nonzero costs, SELL: the stop-buy fills at Ask, so the gap fill
    is max(ask_open, SL) + slippage with the spread included in the ask."""
    idx = _idx(3)
    df = pd.DataFrame({
        "open": [2000.0, 2000.0, 2100.0],
        "high": [2001.0, 2001.0, 2110.0],
        "low": [1999.0, 1999.0, 2095.0],
        "close": [2000.0, 2000.0, 2100.0],
        "atr_14": [2.0, 2.0, 2.0],
    }, index=idx)
    result = _engine(spread_points=20, slippage_points=2).run(
        df, pd.Series([-1, 0, 0], index=idx))

    trade = result["trades"][0]
    # entry = 2000 - 0.02 = 1999.98 (bid), SL 2001.98; ask open = 2100 + 0.20
    # -> fill at 2100.20 + 0.02 = 2100.22.
    assert trade["entry_price"] == 1999.98
    assert trade["exit_reason"] == "STOP_LOSS"
    assert trade["exit_price"] == pytest.approx(2100.22)
    assert trade["gross_profit"] == pytest.approx(-5012.0)


def test_gap_collision_branch_fills_at_open():
    """Coverage: same-bar TP+SL collision where the bar ALSO gapped beyond the
    SL — the conservative collision policy fills at the gap-aware stop price."""
    idx = _idx(3)
    df = pd.DataFrame({
        "open": [2000.0, 2000.0, 1900.0],
        "high": [2001.0, 2001.0, 2004.0],
        "low": [1999.0, 1999.0, 1899.0],
        "close": [2000.0, 2000.0, 1950.0],
        "atr_14": [2.0, 2.0, 2.0],
    }, index=idx)
    result = _engine(slippage_points=1).run(df, pd.Series([1, 0, 0], index=idx))

    trade = result["trades"][0]
    # entry = 2000.01, SL 1998.01, TP 2003.01; bar 2 opens at 1900 (gap) and
    # its high 2004 also exceeds TP -> collision -> gap-aware stop fill.
    assert trade["exit_reason"] == "STOP_LOSS_COLLISION"
    assert trade["exit_price"] == pytest.approx(1899.99)  # min(1900, SL) - 0.01


def test_rounding_contract_fields_reconcile_to_the_cent():
    """Edge case from review: gross 1.004 with 0.006 round-turn commission.
    The CENTS_AT_EVENT contract rounds each money event to cents FIRST and
    derives net from the rounded components, so recorded fields, balance_after
    and final_balance reconcile exactly (no one-cent drift)."""
    idx = _idx(3)
    df = pd.DataFrame({
        "open": [2000.0, 2000.0, 2000.0],
        "high": [2001.0, 2001.0, 2001.1],
        "low": [1999.0, 1999.0, 1999.9],
        "close": [2000.0, 2000.0, 2000.02008],
        "atr_14": [2.0, 2.0, 2.0],
    }, index=idx)
    result = _engine(commission_per_lot=0.012).run(df, pd.Series([1, 0, 0], index=idx))

    trade = result["trades"][0]
    # 0.5 lots: gross = 0.02008 * 100 * 0.5 = 1.004 -> 1.00
    # per-side commission 0.012 * 0.5 = 0.006 -> 0.01 per side -> 0.02 total
    assert trade["gross_profit"] == 1.00
    assert trade["commission"] == 0.02
    assert trade["net_profit"] == 0.98
    assert trade["net_profit"] == round(trade["gross_profit"] - trade["commission"], 2), (
        "recorded fields must reconcile exactly under the rounding contract")
    assert trade["balance_after"] == 10000.98
    assert result["metrics"]["final_balance"] == 10000.98
    assert result["metrics"]["net_profit"] == 0.98
    assert result["assumptions"]["rounding_contract"] == "CENTS_AT_EVENT"
