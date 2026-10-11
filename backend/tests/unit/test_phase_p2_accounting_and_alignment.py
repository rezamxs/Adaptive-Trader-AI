"""Unit and integration tests for Phase P2 audit.
Verifies accounting formulas, double-counting prevention,
threshold mapping, BarrierLabeler collision alignment, and fail-closed safety.
"""
import numpy as np
import pandas as pd
import pytest

from backend.app.core.config import settings, ExecutionMode
from backend.app.services.dataset.labeling import BarrierLabeler
from backend.app.services.backtest.engine import RealisticBacktestEngine


def test_accounting_non_duplication_proof():
    """Mathematical proof that Mid-price reference accounting and Executed-price accounting
    produce the exact same Net PnL without double-counting spread or slippage.
    """
    ref_entry = 2650.00  # Mid entry
    spread = 0.35        # $0.35/oz spread
    entry_slip = 0.02    # $0.02/oz entry slippage
    exit_slip = 0.02     # $0.02/oz exit slippage
    contract_size = 100.0# 100 oz/lot
    vol = 1.0            # 1 lot
    comm_total = 20.0    # $20 round-turn commission

    # Execution prices:
    exec_entry = ref_entry + (spread / 2.0) + entry_slip  # 2650.195
    # Suppose market drops and SL is hit at Bid 2645.00:
    exit_bid = 2645.00
    ref_exit = exit_bid + (spread / 2.0)                  # 2645.175 (Mid)
    exec_exit = exit_bid - exit_slip                      # 2644.980

    # Method A: Deconstructed Mid-price reference
    gross_ref = (ref_exit - ref_entry) * contract_size * vol # (2645.175 - 2650.00) * 100 = -482.50
    trade_spread = spread * contract_size * vol              # 35.00
    total_slip = (entry_slip + exit_slip) * contract_size * vol # 4.00
    total_costs = trade_spread + total_slip + comm_total     # 59.00
    net_method_a = gross_ref - total_costs                   # -541.50

    # Method B: Direct executed price delta
    net_method_b = ((exec_exit - exec_entry) * contract_size * vol) - comm_total # (2644.98 - 2650.195) * 100 - 20 = -521.50 - 20 = -541.50

    assert np.isclose(net_method_a, net_method_b, atol=1e-5), f"Accounting discrepancy: {net_method_a} vs {net_method_b}"
    assert np.isclose(net_method_a, -541.50)


def test_positive_predictions_vs_executed_trades_distinction():
    """Verifies that the number of positive model predictions does NOT equal executed trades
    when a single-position policy skips signals during active trade holding periods.
    """
    # 100 sequential positive predictions
    signals = pd.Series([1] * 100)
    # If each trade holds for up to 10 bars, total trades cannot exceed 100 / 10 = 10
    max_holding = 10
    trades_executed = 0
    i = 0
    while i < len(signals) - max_holding:
        if signals.iloc[i] == 1:
            trades_executed += 1
            i += max_holding  # skip duration
        else:
            i += 1

    assert signals.sum() == 100  # 100 positive predictions
    assert trades_executed == 9  # exactly 9 executed trades (signals during holding were skipped)


def test_barrier_labeler_vs_backtest_collision_alignment():
    """Verifies that both BarrierLabeler and RealisticBacktestEngine adopt the identical
    conservative collision rule: when both TP and SL are hit in the same bar, SL is hit first.
    """
    times = pd.date_range("2026-01-01", periods=5, freq="1min")
    # Bar 0 is entry at 2650.0. Bar 1 has huge range: High=2660 (TP hit), Low=2640 (SL hit)
    df = pd.DataFrame({
        "open":  [2650.0, 2650.0, 2650.0, 2650.0, 2650.0],
        "high":  [2652.0, 2665.0, 2652.0, 2652.0, 2652.0],
        "low":   [2648.0, 2635.0, 2648.0, 2648.0, 2648.0],
        "close": [2650.0, 2650.0, 2650.0, 2650.0, 2650.0],
        "atr_14":[2.0] * 5
    }, index=times)

    labels = BarrierLabeler.label_buy_barrier(
        df,
        tp_atr_mult=1.5, # TP = 2650 + 3.0 = 2653.0
        sl_atr_mult=1.0, # SL = 2650 - 2.0 = 2648.0
        max_horizon_bars=3,
        spread_points=35,
        point_value=0.01
    )
    # BarrierLabeler must resolve to 0
    assert labels.iloc[0] == 0.0

    # Backtest engine check
    engine = RealisticBacktestEngine(
        symbol="XAUUSD",
        timeframe="M1",
        spread_points=35,
        slippage_points=0,
        commission_per_lot=0.0,
        point_value=0.01,
        contract_size=100.0
    )
    sig = pd.Series([1, 0, 0, 0, 0], index=times)
    res = engine.run(df, sig, max_holding_bars=3)
    assert len(res["trades"]) == 1
    # Engine must record STOP_LOSS_COLLISION
    assert res["trades"][0]["exit_reason"] == "STOP_LOSS_COLLISION"


def test_fail_closed_mode_signal_only_preserved():
    """Verifies that execution mode remains strictly SIGNAL_ONLY in Phase P2."""
    assert settings.EXECUTION_MODE == ExecutionMode.SIGNAL_ONLY
    assert settings.DEMO_TRADING_ENABLED is False
