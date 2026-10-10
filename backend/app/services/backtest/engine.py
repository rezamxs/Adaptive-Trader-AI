"""Realistic Event-Driven / Bar-by-Bar Backtest Engine."""
import uuid
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.models.backtest import BacktestRun
from backend.app.services.backtest.metrics import BacktestMetrics


class RealisticBacktestEngine:
    """Executes realistic backtests incorporating spreads, slippage, commission, and lot steps."""

    def __init__(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "M15",
        initial_balance: float = 10000.0,
        spread_points: int = 15,
        slippage_points: int = 3,
        commission_per_lot: float = 7.0,
        risk_per_trade_pct: float = 1.0,
        point_value: float = 0.00001,
        contract_size: float = 100000.0,
        lot_step: float = 0.01,
        min_lot: float = 0.01,
        max_lot: float = 10.0,
        db: Optional[Session] = None
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.spread_points = spread_points
        self.slippage_points = slippage_points
        self.commission_per_lot = commission_per_lot
        self.risk_per_trade_pct = risk_per_trade_pct
        self.point_value = point_value
        self.contract_size = contract_size
        self.lot_step = lot_step
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.db = db

    def calculate_lot_size(self, balance: float, entry_price: float, stop_loss: float) -> float:
        """Calculates volume based on risk percentage, stop loss distance, and broker lot limits."""
        risk_amount = balance * (self.risk_per_trade_pct / 100.0)
        sl_distance = abs(entry_price - stop_loss)
        if sl_distance <= 0:
            return self.min_lot

        # Cost per 1 standard lot for sl_distance price move
        risk_per_standard_lot = sl_distance * self.contract_size
        if risk_per_standard_lot <= 0:
            return self.min_lot

        raw_lots = risk_amount / risk_per_standard_lot
        # Step quantization
        quantized_lots = round(raw_lots / self.lot_step) * self.lot_step
        clipped_lots = max(self.min_lot, min(self.max_lot, quantized_lots))
        return round(clipped_lots, 2)

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,  # 1 for BUY, -1 for SELL, 0 for HOLD
        strategy_name: str = "Quantitative Adaptive Strategy",
        model_id: Optional[str] = None,
        tp_atr_mult: float = 1.5,
        sl_atr_mult: float = 1.0,
        atr_col: str = "atr_14",
        max_holding_bars: int = 50
    ) -> Dict[str, Any]:
        """Executes chronological bar-by-bar backtest."""
        run_id = f"bt_{uuid.uuid4().hex[:10]}"
        n_bars = len(df)
        trades: List[Dict[str, Any]] = []

        times = df.index
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        atrs = df[atr_col].values if atr_col in df.columns else np.full(n_bars, 0.0015)
        sig_values = signals.values

        balance = self.initial_balance
        open_trade: Optional[Dict[str, Any]] = None

        cost_per_point = self.point_value * self.contract_size  # $1 per pip for standard lot

        for i in range(n_bars - 1):
            current_time = times[i]
            next_open = opens[i + 1]

            # 1. Manage Active Trade on bar i
            if open_trade is not None:
                high_bar = highs[i]
                low_bar = lows[i]
                holding_bars = i - open_trade["entry_bar_idx"]

                tp = open_trade["take_profit"]
                sl = open_trade["stop_loss"]
                direction = open_trade["direction"]
                vol = open_trade["volume"]

                exit_price = None
                exit_reason = None

                if direction == "BUY":
                    tp_hit = (high_bar >= tp)
                    sl_hit = (low_bar <= sl)

                    if tp_hit and sl_hit:
                        # Conservative same-bar collision: SL hit first
                        exit_price = sl - (self.slippage_points * self.point_value)
                        exit_reason = "STOP_LOSS_COLLISION"
                    elif sl_hit:
                        exit_price = sl - (self.slippage_points * self.point_value)
                        exit_reason = "STOP_LOSS"
                    elif tp_hit:
                        exit_price = tp - (self.slippage_points * self.point_value)
                        exit_reason = "TAKE_PROFIT"
                    elif holding_bars >= max_holding_bars:
                        exit_price = closes[i]
                        exit_reason = "TIME_EXPIRY"
                else:  # SELL
                    tp_hit = (low_bar <= tp)
                    sl_hit = (high_bar >= sl)

                    if tp_hit and sl_hit:
                        exit_price = sl + (self.slippage_points * self.point_value)
                        exit_reason = "STOP_LOSS_COLLISION"
                    elif sl_hit:
                        exit_price = sl + (self.slippage_points * self.point_value)
                        exit_reason = "STOP_LOSS"
                    elif tp_hit:
                        exit_price = tp + (self.slippage_points * self.point_value)
                        exit_reason = "TAKE_PROFIT"
                    elif holding_bars >= max_holding_bars:
                        exit_price = closes[i]
                        exit_reason = "TIME_EXPIRY"

                if exit_price is not None:
                    # Calculate PnL
                    points = (exit_price - open_trade["entry_price"]) / self.point_value if direction == "BUY" else (open_trade["entry_price"] - exit_price) / self.point_value
                    gross_pnl = points * (self.point_value * self.contract_size * vol)
                    commission = self.commission_per_lot * vol
                    net_pnl = gross_pnl - commission
                    balance += net_pnl

                    trades.append({
                        "trade_id": f"t_{len(trades)+1}",
                        "direction": direction,
                        "volume": float(vol),
                        "entry_time": str(open_trade["entry_time"]),
                        "entry_price": float(round(open_trade["entry_price"], 5)),
                        "exit_time": str(current_time),
                        "exit_price": float(round(exit_price, 5)),
                        "take_profit": float(round(tp, 5)),
                        "stop_loss": float(round(sl, 5)),
                        "holding_bars": int(holding_bars),
                        "exit_reason": str(exit_reason),
                        "gross_profit": float(round(gross_pnl, 2)),
                        "commission": float(round(commission, 2)),
                        "net_profit": float(round(net_pnl, 2)),
                        "balance_after": float(round(balance, 2)),
                    })
                    open_trade = None

            # 2. Check for New Signal on bar i (Executed at open of bar i+1)
            if open_trade is None and i < n_bars - 1:
                sig = sig_values[i]
                if sig != 0 and not np.isnan(sig):
                    atr = atrs[i]
                    if not np.isnan(atr) and atr > 0:
                        direction = "BUY" if sig > 0 else "SELL"
                        # Execution price includes spread + slippage
                        spread_cost = self.spread_points * self.point_value
                        slippage_cost = self.slippage_points * self.point_value

                        if direction == "BUY":
                            entry_price = next_open + spread_cost + slippage_cost
                            stop_loss = entry_price - (sl_atr_mult * atr)
                            take_profit = entry_price + (tp_atr_mult * atr)
                        else:
                            entry_price = next_open - slippage_cost
                            stop_loss = entry_price + (sl_atr_mult * atr)
                            take_profit = entry_price - (tp_atr_mult * atr)

                        lots = self.calculate_lot_size(balance, entry_price, stop_loss)

                        open_trade = {
                            "direction": direction,
                            "volume": lots,
                            "entry_bar_idx": i + 1,
                            "entry_time": times[i + 1],
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                        }

        # 3. Calculate Performance Metrics
        metrics = BacktestMetrics.calculate(trades, initial_balance=self.initial_balance)

        assumptions = {
            "spread_points": self.spread_points,
            "slippage_points": self.slippage_points,
            "commission_per_lot": self.commission_per_lot,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "lot_step": self.lot_step,
            "min_lot": self.min_lot,
            "execution_delay_bars": 1,
            "same_bar_collision_policy": "CONSERVATIVE_STOP_LOSS",
        }

        limitations = [
            "Tick-level slippage may vary during major macroeconomic news releases.",
            "Historical spread is modeled as fixed average points rather than dynamic tick spreads.",
            "Swap/financing fees are not included for weekend holds.",
            "No historical result can guarantee future performance in different market regimes."
        ]

        # 4. Save Record in Database
        if self.db:
            start_date = times[0] if isinstance(times[0], pd.Timestamp) else pd.Timestamp(times[0])
            end_date = times[-1] if isinstance(times[-1], pd.Timestamp) else pd.Timestamp(times[-1])
            
            run_record = BacktestRun(
                run_id=run_id,
                strategy_name=strategy_name,
                model_id=model_id,
                symbol=self.symbol,
                timeframe=self.timeframe,
                start_date=start_date.to_pydatetime(),
                end_date=end_date.to_pydatetime(),
                initial_balance=float(metrics["initial_balance"]),
                final_balance=float(metrics["final_balance"]),
                net_profit=float(metrics["net_profit"]),
                gross_profit=float(metrics["gross_profit"]),
                gross_loss=float(metrics["gross_loss"]),
                max_drawdown_percent=float(metrics["max_drawdown_percent"]),
                max_drawdown_amount=float(metrics["max_drawdown_amount"]),
                total_trades=int(metrics["total_trades"]),
                winning_trades=int(metrics["winning_trades"]),
                losing_trades=int(metrics["losing_trades"]),
                win_rate=float(metrics["win_rate"]),
                profit_factor=float(metrics["profit_factor"]),
                expectancy=float(metrics["expectancy"]),
                average_trade=float(metrics["average_trade"]),
                assumptions=assumptions,
                limitations=limitations,
                equity_curve=metrics["equity_curve"],
                trades_summary=trades[:100],  # store up to 100 trade records
            )
            self.db.add(run_record)
            self.db.commit()
            logger.info(f"Recorded backtest run {run_id} in database.")

        return {
            "run_id": run_id,
            "strategy_name": strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "metrics": metrics,
            "trades": trades,
            "assumptions": assumptions,
            "limitations": limitations,
        }
