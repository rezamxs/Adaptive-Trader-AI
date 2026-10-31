"""Backtest Performance Metrics and Statistical Evaluation."""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class BacktestMetrics:
    """Computes comprehensive quantitative metrics for simulated trade histories."""

    @staticmethod
    def calculate(
        trades: List[Dict[str, Any]],
        initial_balance: float = 10000.0,
        equity_timeline: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Calculates profit, risk, drawdowns, and expectancy metrics."""
        if not trades:
            return {
                "initial_balance": initial_balance,
                "final_balance": initial_balance,
                "net_profit": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "max_drawdown_percent": 0.0,
                "max_drawdown_amount": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "average_trade": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "win_loss_ratio": 0.0,
                "avg_holding_bars": 0.0,
                "equity_curve": [{"time": "START", "equity": initial_balance}],
            }

        profits = [t["net_profit"] for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        gross_profit = float(np.sum(wins)) if wins else 0.0
        gross_loss = float(abs(np.sum(losses))) if losses else 0.0
        net_profit = float(np.sum(profits))
        final_balance = initial_balance + net_profit
        # The engine's realized balance path carries full float precision while
        # recorded net_profit values are rounded to cents; the last recorded
        # balance_after is authoritative so final_balance stays consistent with
        # the trade records down to the cent.
        if trades and trades[-1].get("balance_after") is not None:
            final_balance = float(trades[-1]["balance_after"])
            net_profit = round(final_balance - initial_balance, 2)

        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(abs(np.mean(losses))) if losses else 0.0
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)
        average_trade = net_profit / total_trades if total_trades > 0 else 0.0

        holding_bars = [t.get("holding_bars", 1) for t in trades]
        avg_holding_bars = float(np.mean(holding_bars)) if holding_bars else 0.0

        # Construct Equity Curve & Calculate Max Drawdown. When the engine
        # supplies an equity_timeline it contains BOTH closed marks and
        # worst-case intra-bar floating marks (M2), so unrealized losses
        # register in the drawdown; without one, fall back to the
        # closed-trades-only path.
        max_dd_amount = 0.0
        max_dd_pct = 0.0
        if equity_timeline:
            equity_curve = [{"time": "START", "equity": round(initial_balance, 2)}]
            peak = initial_balance
            for event in equity_timeline:
                eq = float(event["equity"])
                equity_curve.append({
                    "time": str(event.get("time", "N/A")),
                    "equity": round(eq, 2),
                })
                if eq > peak:
                    peak = eq
                dd_amount = peak - eq
                dd_pct = (dd_amount / peak) * 100.0 if peak > 0 else 0.0
                if dd_amount > max_dd_amount:
                    max_dd_amount = dd_amount
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
        else:
            equity = initial_balance
            equity_curve = [{"time": "START", "equity": initial_balance}]
            peak = initial_balance

            for t in trades:
                equity += t["net_profit"]
                equity_curve.append({
                    "time": str(t.get("exit_time", "N/A")),
                    "equity": round(equity, 2),
                })
                if equity > peak:
                    peak = equity
                dd_amount = peak - equity
                dd_pct = (dd_amount / peak) * 100.0 if peak > 0 else 0.0
                if dd_amount > max_dd_amount:
                    max_dd_amount = dd_amount
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct

        return {
            "initial_balance": round(initial_balance, 2),
            "final_balance": round(final_balance, 2),
            "net_profit": round(net_profit, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "max_drawdown_percent": round(max_dd_pct, 2),
            "max_drawdown_amount": round(max_dd_amount, 2),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate * 100.0, 2),
            "profit_factor": round(profit_factor, 2),
            "expectancy": round(expectancy, 2),
            "average_trade": round(average_trade, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "win_loss_ratio": round(win_loss_ratio, 2),
            "avg_holding_bars": round(avg_holding_bars, 1),
            "equity_curve": equity_curve,
        }
