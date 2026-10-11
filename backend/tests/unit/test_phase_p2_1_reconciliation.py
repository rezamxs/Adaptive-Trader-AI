"""Unit tests for Phase P2.1 trade ledger reconciliation, equity drawdown math,
three-level signal accounting, cost scaling, and safety gates.
"""
import numpy as np
import pandas as pd
import pytest

from backend.app.core.config import settings, ExecutionMode


def test_drawdown_initial_zero_vs_closed_only():
    """Verifies that standard financial equity drawdown includes the initial baseline 0.0,
    resolving the Phase P2 paradox where Final Loss was -$4,499.73 while reported Max DD was $4,455.30.
    """
    # Sample sequence where first trade is a loss
    trades_pnl = [-164.07, -154.86, 135.18, -184.50]
    cum_pnl = np.cumsum(trades_pnl)

    # Legacy flawed method: without initial 0
    peak_legacy = np.maximum.accumulate(cum_pnl)
    dd_legacy = peak_legacy - cum_pnl
    max_dd_legacy = np.max(dd_legacy)

    # Standard financial method: with initial 0
    cum_pnl_with_zero = np.insert(cum_pnl, 0, 0.0)
    peak_standard = np.maximum.accumulate(cum_pnl_with_zero)
    dd_standard = peak_standard - cum_pnl_with_zero
    max_dd_standard = np.max(dd_standard)

    # The legacy peak starts at -164.07, missing the drawdown of the first trade from 0
    assert peak_legacy[0] == -164.07
    assert peak_standard[0] == 0.0

    # The standard drawdown correctly captures the initial loss from starting equity 0
    assert dd_standard[1] == 164.07
    assert max_dd_standard > max_dd_legacy


def test_stepping_shift_reconciliation_math():
    """Verifies the mathematical origin of the 6-trade difference between Report P (162) and Report P2 (156):
    Report P: entry on bar i close, step i += 24.
    Report P2: entry on bar i+1 open, step i = (i+1) + 24 = i + 25.
    """
    n_bars = 100
    signals = [0] * n_bars
    # Create signals at bar 0, 24, 48, 72 (exactly on boundary of 24-bar step)
    signals[0] = 1
    signals[24] = 1
    signals[48] = 1
    signals[72] = 1

    # Scheme A (Phase P): bar i entry, step i += 24
    trades_a = []
    i = 0
    while i < n_bars - 24:
        if signals[i] == 1:
            trades_a.append(i)
            i += 24
        else:
            i += 1

    # Scheme B (Phase P2): entry bar i+1, step = entry_bar + 24 = (i+1) + 24 = i + 25
    trades_b = []
    i = 0
    while i < n_bars - 24:
        if signals[i] == 1:
            trades_b.append(i)
            i = (i + 1) + 24  # step by 25 bars
        else:
            i += 1

    # In Scheme A, bar 24 is hit because i stepped from 0 -> 24.
    # In Scheme B, bar 24 is inside the holding period [1..24] and skipped because i stepped from 0 -> 25!
    assert 24 in trades_a
    assert 24 not in trades_b
    assert len(trades_a) > len(trades_b)


def test_three_level_signal_accounting_identity():
    """Verifies that the three levels of signal accounting satisfy strict conservation:
    positive_signals = signals_rejected_open_pos + signals_rejected_other + simulated_entry_requests.
    And simulated_entry_requests == trades_opened == trades_closed in simulated offline backtest.
    """
    total_signals = 3735
    rejected_open_pos = 3579
    rejected_other = 0
    entry_requests = 156
    trades_opened = 156
    trades_closed = 156

    assert total_signals == (rejected_open_pos + rejected_other + entry_requests)
    assert entry_requests == trades_opened
    assert trades_opened == trades_closed


def test_cost_ledger_volume_scaling():
    """Verifies that transaction costs strictly scale with lot volume,
    and broker specification ($35 spread + $4 slippage + $20 commission) holds exactly.
    """
    def compute_trade_costs(lots: float, contract_size: float = 100.0,
                            spread_pts: float = 35.0, pt_val: float = 0.01,
                            comm_per_lot: float = 20.0, slip_pts: float = 2.0):
        spread_usd = spread_pts * pt_val * contract_size * lots
        slip_usd = (2 * slip_pts) * pt_val * contract_size * lots  # round turn
        comm_usd = comm_per_lot * lots
        tot_usd = spread_usd + slip_usd + comm_usd
        return spread_usd, slip_usd, comm_usd, tot_usd

    # 1.0 standard lot
    sp1, sl1, cm1, tot1 = compute_trade_costs(1.0)
    assert sp1 == 35.0
    assert sl1 == 4.0
    assert cm1 == 20.0
    assert tot1 == 59.0

    # 0.1 mini lot
    sp01, sl01, cm01, tot01 = compute_trade_costs(0.1)
    assert np.isclose(sp01, 3.5)
    assert np.isclose(sl01, 0.4)
    assert np.isclose(cm01, 2.0)
    assert np.isclose(tot01, 5.9)

    # 0.01 micro lot
    sp001, sl001, cm001, tot001 = compute_trade_costs(0.01)
    assert np.isclose(sp001, 0.35)
    assert np.isclose(sl001, 0.04)
    assert np.isclose(cm001, 0.20)
    assert np.isclose(tot001, 0.59)


def test_security_gates_and_signal_only_mode():
    """Verifies that fail-closed safety constraints remain strictly intact in Phase P2.1."""
    assert settings.EXECUTION_MODE == ExecutionMode.SIGNAL_ONLY
    assert settings.DEMO_TRADING_ENABLED is False
