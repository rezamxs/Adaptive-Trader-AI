"""Realistic Event-Driven / Bar-by-Bar Backtest Engine."""
import uuid
import math
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
        tick_size: float = 0.0,
        tick_value: float = 0.0,
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
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.lot_step = lot_step
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.db = db

    def calculate_lot_size(self, balance: float, entry_price: float, stop_loss: float) -> float:
        """Size a position using broker tick economics, rounding down and enforcing a hard risk cap."""
        risk_amount = balance * (self.risk_per_trade_pct / 100.0)
        hard_max_risk_amount = risk_amount * 1.5
        sl_distance = abs(entry_price - stop_loss)
        if (
            not math.isfinite(sl_distance)
            or sl_distance <= 0
            or not math.isfinite(risk_amount)
            or risk_amount <= 0
            or not math.isfinite(self.lot_step)
            or self.lot_step <= 0
        ):
            return 0.0

        if self.tick_size > 0 and self.tick_value > 0:
            risk_per_standard_lot = (sl_distance / self.tick_size) * self.tick_value
        else:
            risk_per_standard_lot = sl_distance * self.contract_size
        if not math.isfinite(risk_per_standard_lot) or risk_per_standard_lot <= 0:
            return 0.0

        raw_lots = risk_amount / risk_per_standard_lot
        stepped_lots = math.floor((raw_lots / self.lot_step) + 1e-12) * self.lot_step
        stepped_lots = round(stepped_lots, 8)

        if stepped_lots < self.min_lot:
            # Minimum broker size is allowed only if it remains inside the hard cap.
            if self.min_lot * risk_per_standard_lot > hard_max_risk_amount:
                return 0.0
            stepped_lots = self.min_lot

        final_lots = min(self.max_lot, stepped_lots)
        if final_lots <= 0 or final_lots < self.min_lot:
            return 0.0
        if final_lots * risk_per_standard_lot > hard_max_risk_amount:
            return 0.0
        return round(final_lots, 8)

    def _calculate_gross_pnl(
        self, entry_price: float, exit_price: float, direction: str, volume: float
    ) -> float:
        """Calculate PnL in deposit currency using broker tick value when available."""
        price_change = exit_price - entry_price if direction == "BUY" else entry_price - exit_price
        if self.tick_size > 0 and self.tick_value > 0:
            return (price_change / self.tick_size) * self.tick_value * volume
        return price_change * self.contract_size * volume

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
        if n_bars == 0:
            # Fail loud: an empty frame previously produced a silent
            # "successful" empty run, or an IndexError from the DB-save block.
            raise ValueError("Backtest received an empty frame; nothing to simulate.")
        trades: List[Dict[str, Any]] = []

        times = df.index
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        if atr_col not in df.columns:
            # Fail loud instead of fabricating a hardcoded ATR: on XAUUSD
            # (~$2000) a 0.0015 fallback ATR yields sub-point SL/TP and
            # near-max lots — silently garbage-in position sizing.
            raise ValueError(
                f"Backtest requires the ATR column '{atr_col}' in the frame; "
                "refusing to substitute a hardcoded fallback ATR."
            )
        atrs = df[atr_col].values
        sig_values = signals.values

        balance = self.initial_balance
        open_trade: Optional[Dict[str, Any]] = None
        # Chronological equity marks (floating + closed) for drawdown (M2).
        equity_timeline: List[Dict[str, Any]] = []

        for i in range(n_bars):
            current_time = times[i]

            # 1. Manage Active Trade on bar i
            if open_trade is not None:
                high_bar = highs[i]
                low_bar = lows[i]
                holding_bars = i - open_trade["entry_bar_idx"]
                spread_cost = self.spread_points * self.point_value
                slippage_cost = self.slippage_points * self.point_value

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
                        # Conservative same-bar collision: SL hit first. A bar
                        # that OPENED beyond the stop fills at the open (gap).
                        exit_price = min(opens[i], sl) - (self.slippage_points * self.point_value)
                        exit_reason = "STOP_LOSS_COLLISION"
                    elif sl_hit:
                        exit_price = min(opens[i], sl) - (self.slippage_points * self.point_value)
                        exit_reason = "STOP_LOSS"
                    elif tp_hit:
                        exit_price = tp - (self.slippage_points * self.point_value)
                        exit_reason = "TAKE_PROFIT"
                    elif holding_bars >= max_holding_bars:
                        exit_price = closes[i] - slippage_cost
                        exit_reason = "TIME_EXPIRY"
                else:  # SELL
                    # MT5 historical OHLC is bid-based. A short position closes at
                    # Ask, approximated as Bid + spread, so spread affects exit triggers.
                    ask_high = high_bar + spread_cost
                    ask_low = low_bar + spread_cost
                    tp_hit = (ask_low <= tp)
                    sl_hit = (ask_high >= sl)

                    if tp_hit and sl_hit:
                        exit_price = max(opens[i] + spread_cost, sl) + (self.slippage_points * self.point_value)
                        exit_reason = "STOP_LOSS_COLLISION"
                    elif sl_hit:
                        exit_price = max(opens[i] + spread_cost, sl) + (self.slippage_points * self.point_value)
                        exit_reason = "STOP_LOSS"
                    elif tp_hit:
                        exit_price = tp + (self.slippage_points * self.point_value)
                        exit_reason = "TAKE_PROFIT"
                    elif holding_bars >= max_holding_bars:
                        exit_price = closes[i] + spread_cost + slippage_cost
                        exit_reason = "TIME_EXPIRY"

                # M2 floating-equity marks. OHLC cannot prove the real intra-bar
                # tick path, so a conservative convention is booked (documented
                # in assumptions/limitations — NOT a tick-level simulation):
                #   - bar with no barrier hit: FAVORABLE extreme first, then
                #     ADVERSE extreme, so a floating peak before a possible
                #     drop is never lost (peak-to-trough inside one bar);
                #   - TP-exit bar: the ADVERSE extreme may precede the limit
                #     fill (dip booked first); nothing beyond the fill accrues;
                #   - SL-exit bar: the FAVORABLE extreme may precede the stop
                #     fill (rally booked first); nothing beyond the fill
                #     accrues (gap fills included via the fill price).
                # Marks are net of the exit-side commission (entry side already
                # deducted from balance). On the FINAL bar with no barrier hit
                # the END_OF_DATA block books the peak/trough marks exactly
                # once — booking them here too would duplicate them.
                if exit_price is not None or i < n_bars - 1:
                    exit_side_commission = round(self.commission_per_lot * vol, 2)
                    base_equity = balance - exit_side_commission
                    if direction == "BUY":
                        favorable_price, adverse_price = high_bar, low_bar
                    else:
                        favorable_price = low_bar + spread_cost
                        adverse_price = high_bar + spread_cost
                    if exit_reason == "TAKE_PROFIT":
                        favorable_price = None  # limit fill caps the favorable excursion
                    elif exit_reason in ("STOP_LOSS", "STOP_LOSS_COLLISION"):
                        adverse_price = None  # stop fill caps the adverse excursion
                    if favorable_price is not None:
                        equity_timeline.append({
                            "time": str(current_time),
                            "equity": float(base_equity + self._calculate_gross_pnl(
                                open_trade["entry_price"], favorable_price, direction, vol)),
                            "kind": "floating_peak",
                        })
                    if adverse_price is not None:
                        equity_timeline.append({
                            "time": str(current_time),
                            "equity": float(base_equity + self._calculate_gross_pnl(
                                open_trade["entry_price"], adverse_price, direction, vol)),
                            "kind": "floating_trough",
                        })

                if exit_price is not None:
                    # CENTS_AT_EVENT rounding contract: every money event is
                    # rounded to cents when it happens and net derives from the
                    # rounded components, so gross - commission == net and each
                    # balance step agrees with the recorded fields exactly.
                    gross_pnl = round(self._calculate_gross_pnl(
                        open_trade["entry_price"], exit_price, direction, vol
                    ), 2)
                    exit_commission = round(self.commission_per_lot * vol, 2)
                    total_commission = round(
                        open_trade["entry_commission"] + exit_commission, 2)
                    balance += round(gross_pnl - exit_commission, 2)

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
                        "gross_profit": float(gross_pnl),
                        "commission": float(total_commission),
                        "net_profit": float(round(gross_pnl - total_commission, 2)),
                        "balance_after": float(round(balance, 2)),
                    })
                    equity_timeline.append(
                        {"time": str(current_time), "equity": float(balance), "kind": "closed"})
                    open_trade = None

            # 2. Check for New Signal on bar i (Executed at open of bar i+1)
            if open_trade is None and i < n_bars - 1:
                next_open = opens[i + 1]
                sig = sig_values[i]
                if sig != 0 and not np.isnan(sig):
                    atr = atrs[i]
                    if not np.isnan(atr) and atr > 0:
                        direction = "BUY" if sig > 0 else "SELL"
                        # Execution price includes spread + slippage
                        spread_cost = self.spread_points * self.point_value
                        slippage_cost = self.slippage_points * self.point_value

                        # MT5 historical OHLC is bid-based: buy entries execute at
                        # Ask (Bid + spread), while sell entries execute at Bid.
                        if direction == "BUY":
                            entry_price = next_open + spread_cost + slippage_cost
                            stop_loss = entry_price - (sl_atr_mult * atr)
                            take_profit = entry_price + (tp_atr_mult * atr)
                        else:
                            entry_price = next_open - slippage_cost
                            stop_loss = entry_price + (sl_atr_mult * atr)
                            take_profit = entry_price - (tp_atr_mult * atr)

                        lots = self.calculate_lot_size(balance, entry_price, stop_loss)
                        if lots > 0:
                            # commission_per_lot is PER SIDE: charge the entry
                            # side now (rounded to cents); the exit side is
                            # charged at exit under the same contract.
                            entry_commission = round(self.commission_per_lot * lots, 2)
                            balance -= entry_commission
                            open_trade = {
                                "direction": direction,
                                "volume": lots,
                                "entry_bar_idx": i + 1,
                                "entry_time": times[i + 1],
                                "entry_price": entry_price,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                                "entry_commission": entry_commission,
                            }

        # 3. Liquidate any position still open at the final available close.
        # Omitting it biases trade count and PnL when the data ends before TP/SL.
        if open_trade is not None and n_bars > 0:
            final_idx = n_bars - 1
            direction = open_trade["direction"]
            vol = open_trade["volume"]
            exit_price = float(closes[final_idx])
            slippage_cost = self.slippage_points * self.point_value
            spread_cost = self.spread_points * self.point_value
            if direction == "BUY":
                exit_price -= slippage_cost
            else:
                # Closing a short is a buy at Ask; bid OHLC needs spread added.
                exit_price += spread_cost + slippage_cost

            # Final bar exposure: both extremes were live (no barrier hit), so
            # book the favorable-then-adverse marks before the close. They MUST
            # use the PRE-close equity base: computing them from the post-close
            # balance would bake the realized close P&L into both marks
            # (review regression — artificial curve values).
            exit_side_commission = round(self.commission_per_lot * vol, 2)
            base_equity = balance - exit_side_commission
            if direction == "BUY":
                favorable_price, adverse_price = highs[final_idx], lows[final_idx]
            else:
                favorable_price = lows[final_idx] + spread_cost
                adverse_price = highs[final_idx] + spread_cost
            equity_timeline.append({
                "time": str(times[final_idx]),
                "equity": float(base_equity + self._calculate_gross_pnl(
                    open_trade["entry_price"], favorable_price, direction, vol)),
                "kind": "floating_peak",
            })
            equity_timeline.append({
                "time": str(times[final_idx]),
                "equity": float(base_equity + self._calculate_gross_pnl(
                    open_trade["entry_price"], adverse_price, direction, vol)),
                "kind": "floating_trough",
            })

            gross_pnl = round(self._calculate_gross_pnl(
                open_trade["entry_price"], exit_price, direction, vol
            ), 2)
            exit_commission = exit_side_commission
            total_commission = round(open_trade["entry_commission"] + exit_commission, 2)
            balance += round(gross_pnl - exit_commission, 2)
            trades.append({
                "trade_id": f"t_{len(trades)+1}",
                "direction": direction,
                "volume": float(vol),
                "entry_time": str(open_trade["entry_time"]),
                "entry_price": float(round(open_trade["entry_price"], 5)),
                "exit_time": str(times[final_idx]),
                "exit_price": float(round(exit_price, 5)),
                "take_profit": float(round(open_trade["take_profit"], 5)),
                "stop_loss": float(round(open_trade["stop_loss"], 5)),
                "holding_bars": int(max(0, final_idx - open_trade["entry_bar_idx"])),
                "exit_reason": "END_OF_DATA",
                "gross_profit": float(gross_pnl),
                "commission": float(total_commission),
                "net_profit": float(round(gross_pnl - total_commission, 2)),
                "balance_after": float(round(balance, 2)),
            })
            equity_timeline.append(
                {"time": str(times[final_idx]), "equity": float(balance), "kind": "closed"})
            open_trade = None

        # 4. Calculate Performance Metrics
        metrics = BacktestMetrics.calculate(
            trades, initial_balance=self.initial_balance, equity_timeline=equity_timeline
        )

        assumptions = {
            "spread_points": self.spread_points,
            "slippage_points": self.slippage_points,
            "commission_per_lot": self.commission_per_lot,
            "commission_model": "PER_SIDE",
            "rounding_contract": "CENTS_AT_EVENT",
            "max_drawdown_includes_floating": True,
            "floating_mark_policy": (
                "FAVORABLE_THEN_ADVERSE_WHILE_OPEN; "
                "ADVERSE_BEFORE_TP_FILL; FAVORABLE_BEFORE_STOP_FILL"),
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "lot_step": self.lot_step,
            "min_lot": self.min_lot,
            "point_value": self.point_value,
            "contract_size": self.contract_size,
            "execution_delay_bars": 1,
            "same_bar_collision_policy": "CONSERVATIVE_STOP_LOSS",
            "gap_through_stop_policy": "FILL_AT_BAR_OPEN",
            "price_basis": "MT5_BID_OHLC",
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
        }

        limitations = [
            "Tick-level slippage may vary during major macroeconomic news releases.",
            "Historical spread is modeled as fixed average points rather than dynamic tick spreads.",
            "A gap through the stop is filled at the bar's printed open plus slippage; in extreme gaps the printed open may not be tradeable.",
            "Intra-bar equity marks use OHLC extremes with a conservative ordering (favorable extreme booked before the adverse one while a trade is open). OHLC cannot prove the real tick path, so these marks are a worst-case convention, not a tick-level simulation.",
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
