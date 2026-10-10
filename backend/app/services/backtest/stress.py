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
        initial_balance: float = 10000.0
    ) -> Dict[str, Any]:
        """Runs four execution cost stress scenarios."""
        scenarios = {
            "baseline": {"spread_points": 15, "slippage_points": 3, "commission": 7.0},
            "elevated_spread": {"spread_points": 30, "slippage_points": 3, "commission": 7.0},
            "high_slippage": {"spread_points": 15, "slippage_points": 9, "commission": 7.0},
            "adverse_stress": {"spread_points": 35, "slippage_points": 10, "commission": 10.0},
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
