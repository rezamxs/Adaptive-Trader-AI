"""Unit tests for Realistic Backtest Engine, Metrics, and Stress Scenarios."""
import pytest
import numpy as np
import pandas as pd
from backend.app.services.market_data.mock_provider import MockMarketDataProvider
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.dataset.features import FeatureEngineer
from backend.app.services.backtest.engine import RealisticBacktestEngine
from backend.app.services.backtest.metrics import BacktestMetrics
from backend.app.services.backtest.stress import BacktestStressTester
from backend.app.services.backtest.report import BacktestReportGenerator


def test_backtest_metrics_calculation():
    """Verify statistical formulas for profit factor, expectancy, drawdown."""
    trades = [
        {"net_profit": 150.0, "exit_time": "2026-01-01", "holding_bars": 5},
        {"net_profit": -100.0, "exit_time": "2026-01-02", "holding_bars": 3},
        {"net_profit": 200.0, "exit_time": "2026-01-03", "holding_bars": 6},
        {"net_profit": -50.0, "exit_time": "2026-01-04", "holding_bars": 2},
    ]
    res = BacktestMetrics.calculate(trades, initial_balance=10000.0)

    assert res["initial_balance"] == 10000.0
    assert res["final_balance"] == 10200.0
    assert res["net_profit"] == 200.0
    assert res["gross_profit"] == 350.0
    assert res["gross_loss"] == 150.0
    assert res["profit_factor"] == round(350.0 / 150.0, 2)
    assert res["total_trades"] == 4
    assert res["winning_trades"] == 2
    assert res["win_rate"] == 50.0


def test_lot_size_quantization_and_clipping():
    """Verify lot sizing respects broker min_lot, max_lot, and lot_step."""
    engine = RealisticBacktestEngine(
        initial_balance=10000.0,
        risk_per_trade_pct=1.0,  # $100 risk
        lot_step=0.01,
        min_lot=0.01,
        max_lot=5.0,
    )
    # Entry 1.1000, SL 1.0950 -> distance 50 pips ($500 per standard lot)
    # Target risk = $100 -> raw lot = 100 / 500 = 0.20 lots
    lots = engine.calculate_lot_size(balance=10000.0, entry_price=1.1000, stop_loss=1.0950)
    assert lots == 0.20

    # Very small SL distance (e.g. 1 pip) -> would exceed max_lot
    huge_lots = engine.calculate_lot_size(balance=10000.0, entry_price=1.1000, stop_loss=1.0999)
    assert huge_lots == 5.0  # clipped to max_lot

    # Very large SL distance -> clipped to min_lot
    tiny_lots = engine.calculate_lot_size(balance=10000.0, entry_price=1.1000, stop_loss=1.0000)
    assert tiny_lots == 0.01  # clipped to min_lot


def test_realistic_backtest_costs_impact():
    """Verify execution costs (spread, commission, slippage) demonstrably reduce net profit."""
    provider = MockMarketDataProvider()
    raw_df = provider.get_historical_dataframe("EURUSD", TimeFrame.M15, count=200)
    df, _ = FeatureEngineer.transform(raw_df)

    # Generate a simple alternating signal series
    signals = pd.Series(np.zeros(len(df)), index=df.index)
    signals.iloc[::20] = 1  # Buy every 20 bars

    # Frictionless engine (0 spread, 0 slippage, 0 commission)
    engine_free = RealisticBacktestEngine(
        spread_points=0,
        slippage_points=0,
        commission_per_lot=0.0
    )
    res_free = engine_free.run(df, signals)

    # Realistic engine
    engine_cost = RealisticBacktestEngine(
        spread_points=20,
        slippage_points=4,
        commission_per_lot=7.0
    )
    res_cost = engine_cost.run(df, signals)

    # Net profit with costs MUST be lower than frictionless
    assert res_cost["metrics"]["net_profit"] < res_free["metrics"]["net_profit"]


def test_stress_testing_and_report_generation():
    """Verify stress test execution across scenarios and report markdown output."""
    provider = MockMarketDataProvider()
    raw_df = provider.get_historical_dataframe("EURUSD", TimeFrame.M15, count=200)
    df, _ = FeatureEngineer.transform(raw_df)

    signals = pd.Series(np.zeros(len(df)), index=df.index)
    signals.iloc[::15] = 1

    engine = RealisticBacktestEngine()
    bt_res = engine.run(df, signals)

    stress_res = BacktestStressTester.run_stress_scenarios(df, signals)
    assert "baseline" in stress_res["scenarios"]
    assert "adverse_stress" in stress_res["scenarios"]

    # Generate report
    report_md = BacktestReportGenerator.generate_markdown(bt_res, stress_res)
    assert "# گزارش تحلیلی بکتست" in report_md
    assert "حداکثر افت سرمایه" in report_md
    assert "adverse_stress" in report_md


def test_open_position_is_liquidated_at_end_of_dataset():
    """An open final trade must be closed and included in reported performance."""
    idx = pd.date_range("2026-10-12 10:00:00", periods=3, freq="min", tz="UTC")
    df = pd.DataFrame({
        "open": [2000.0, 2000.0, 2000.0],
        "high": [2001.0, 2001.0, 2002.5],
        "low": [1999.0, 1999.0, 1999.0],
        "close": [2000.0, 2000.0, 2001.0],
        "atr_14": [2.0, 2.0, 2.0],
    }, index=idx)
    signals = pd.Series([0, 1, 0], index=idx)
    engine = RealisticBacktestEngine(
        symbol="XAUUSD", initial_balance=10000.0, spread_points=0,
        slippage_points=0, commission_per_lot=7.0, point_value=0.01,
        contract_size=100.0,
    )
    result = engine.run(df, signals)

    assert result["metrics"]["total_trades"] == 1
    trade = result["trades"][0]
    assert trade["exit_reason"] == "END_OF_DATA"
    assert trade["exit_time"] == str(idx[-1])
    assert trade["commission"] > 0
    assert result["metrics"]["net_profit"] == trade["net_profit"]
