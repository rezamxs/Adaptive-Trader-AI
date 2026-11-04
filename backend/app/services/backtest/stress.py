"""Stress Testing module for Backtested Strategies under Adverse Market Conditions."""
from typing import Any, Dict, List
import pandas as pd
from backend.app.services.backtest.engine import RealisticBacktestEngine


class BacktestStressTester:
    """Executes sensitivity and stress tests with adverse spreads, slippage, and fees."""

    @classmethod
    def run_stress_scenarios(
        cls,
        df: pd.DataFrame,
        signals: pd.Series,
        symbol: str = "EURUSD",
        timeframe: str = "M15",
        initial_balance: float = 10000.0,
        base_spread_points: int = 15,
        point_value: float = 0.00001,
        contract_size: float = 100000.0,
        tick_size: float = 0.0,
        tick_value: float = 0.0,
        lot_step: float = 0.01,
        min_lot: float = 0.01,
        max_lot: float = 10.0,
        base_slippage_points: int = 3,
        commission_per_lot: float = 7.0,
    ) -> Dict[str, Any]:
        """Runs cost stress scenarios using the same broker contract as the base test."""
        base_spread = max(0, int(base_spread_points))
        base_slippage = max(0, int(base_slippage_points))
        base_commission = max(0.0, float(commission_per_lot))
        scenarios = {
            "baseline": {"spread_points": base_spread, "slippage_points": base_slippage, "commission": base_commission},
            "elevated_spread": {"spread_points": max(base_spread + 1, int(round(base_spread * 2))), "slippage_points": base_slippage, "commission": base_commission},
            "high_slippage": {"spread_points": base_spread, "slippage_points": max(base_slippage + 1, int(round(base_slippage * 3))), "commission": base_commission},
            "adverse_stress": {"spread_points": max(base_spread + 1, int(round(base_spread * 2.5))), "slippage_points": max(base_slippage + 1, int(round(base_slippage * 3))), "commission": base_commission * 1.5},
        }

        results = {}
        for name, params in scenarios.items():
            engine = RealisticBacktestEngine(
                symbol=symbol,
                timeframe=timeframe,
                initial_balance=initial_balance,
                spread_points=params["spread_points"],
                slippage_points=params["slippage_points"],
                commission_per_lot=params["commission"],
                point_value=point_value,
                contract_size=contract_size,
                tick_size=tick_size,
                tick_value=tick_value,
                lot_step=lot_step,
                min_lot=min_lot,
                max_lot=max_lot,
            )
            res = engine.run(df, signals, strategy_name=f"Stress Test ({name})")
            results[name] = {
                "params": params,
                "net_profit": res["metrics"]["net_profit"],
                "profit_factor": res["metrics"]["profit_factor"],
                "max_drawdown_percent": res["metrics"]["max_drawdown_percent"],
                "win_rate": res["metrics"]["win_rate"],
                "total_trades": res["metrics"]["total_trades"],
            }

        # Calculate robustness score
        base_profit = results["baseline"]["net_profit"]
        adverse_profit = results["adverse_stress"]["net_profit"]

        robust = (adverse_profit > 0) if base_profit > 0 else False

        return {
            "scenarios": results,
            "is_cost_robust": robust,
            "adverse_profit_retention_pct": round((adverse_profit / base_profit) * 100, 2) if base_profit > 0 else 0.0,
        }
