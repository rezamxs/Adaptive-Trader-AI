"""Backtesting API Endpoints."""
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import pandas as pd
import math
import numpy as np

from backend.app.db.session import get_db
from backend.app.models.backtest import BacktestRun
from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.dataset.features import FeatureEngineer
from backend.app.services.backtest.engine import RealisticBacktestEngine
from backend.app.services.backtest.stress import BacktestStressTester
from backend.app.services.backtest.report import BacktestReportGenerator
from backend.app.services.knowledge.manager import KnowledgeManager
from backend.app.services.signal.generator import SignalGenerator

router = APIRouter()


class RunBacktestRequest(BaseModel):
    symbol: str = "EURUSD"
    timeframe: TimeFrame = TimeFrame.M15
    strategy_name: str = "Approved Strategy Backtest"
    bars_count: int = Field(default=300, ge=50, le=100000)
    spread_points: Optional[int] = Field(default=None, ge=0, le=100000)
    slippage_points: int = Field(default=3, ge=0, le=100000)
    commission_per_lot: float = Field(default=7.0, ge=0.0, le=100000.0)
    initial_balance: float = Field(default=10000.0, gt=0.0)


@router.get("/runs")
def list_backtest_runs(db: Session = Depends(get_db)):
    """Lists past backtest runs."""
    runs = db.query(BacktestRun).order_by(BacktestRun.id.desc()).all()
    return [{
        "run_id": r.run_id,
        "strategy_name": r.strategy_name,
        "symbol": r.symbol,
        "timeframe": r.timeframe,
        "net_profit": r.net_profit,
        "max_drawdown_percent": r.max_drawdown_percent,
        "profit_factor": r.profit_factor,
        "total_trades": r.total_trades,
        "win_rate": r.win_rate,
        "created_at": r.created_at.isoformat(),
    } for r in runs]


@router.post("/run")
def execute_backtest(req: RunBacktestRequest, db: Session = Depends(get_db)):
    """Executes realistic backtest with cost modeling and stress analysis."""
    provider = MT5MarketDataProvider()
    if not provider.connect():
        raise HTTPException(
            status_code=503,
            detail="MT5 is unavailable; backtesting with synthetic fallback data is disabled.",
        )

    try:
        df_raw = provider.get_historical_dataframe(req.symbol, req.timeframe, count=req.bars_count)
        symbol_info = provider.get_symbol_info(req.symbol)
    finally:
        provider.disconnect()

    if symbol_info is None:
        raise HTTPException(status_code=503, detail=f"MT5 did not return broker specifications for {req.symbol}.")

    try:
        point_value = float(symbol_info.point)
        tick_size = float(symbol_info.tick_size)
        tick_value = float(symbol_info.tick_value)
        contract_size = float(symbol_info.trade_contract_size)
        volume_step = float(symbol_info.volume_step)
        min_lot = float(symbol_info.volume_min)
        max_lot = float(symbol_info.volume_max)
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(status_code=503, detail=f"MT5 returned invalid broker specifications for {req.symbol}.")

    if (
        not all(math.isfinite(value) and value > 0 for value in
                (point_value, tick_size, tick_value, contract_size, volume_step, min_lot, max_lot))
        or max_lot < min_lot
    ):
        raise HTTPException(
            status_code=503,
            detail=f"MT5 tick economics or volume limits are invalid for {req.symbol}; refusing inaccurate backtest.",
        )

    if len(df_raw) < 50:
        raise HTTPException(status_code=400, detail="Insufficient bars to run backtest.")

    if req.spread_points is None:
        historical_spread = (
            pd.to_numeric(df_raw["spread"], errors="coerce").dropna()
            if "spread" in df_raw.columns else pd.Series(dtype=float)
        )
        effective_spread_points = (
            max(0, int(round(float(historical_spread.median()))))
            if not historical_spread.empty
            else max(0, int(symbol_info.spread))
        )
    else:
        effective_spread_points = req.spread_points

    # 1. Transform features
    df_feat, _ = FeatureEngineer.transform(df_raw)

    # 2. A backtest must represent an explicitly approved strategy, never an
    # implicit fallback heuristic that the user has not reviewed or approved.
    km = KnowledgeManager(db)
    approved_rules = km.get_active_strategy_rules()
    if not approved_rules:
        raise HTTPException(
            status_code=400,
            detail="No approved strategy rules are available; approve a strategy before backtesting.",
        )

    # 3. Generate signals using only approved rules within their declared scope.
    signals = pd.Series(np.zeros(len(df_feat)), index=df_feat.index)
    for i in range(len(df_feat)):
        bar = df_feat.iloc[i]
        direction, _, _ = SignalGenerator.evaluate_rules(
            bar, approved_rules, symbol=req.symbol, timeframe=req.timeframe.value
        )
        if direction == "BUY":
            signals.iloc[i] = 1
        elif direction == "SELL":
            signals.iloc[i] = -1

    # 4. Run Backtest
    engine = RealisticBacktestEngine(
        symbol=req.symbol,
        timeframe=req.timeframe.value,
        initial_balance=req.initial_balance,
        spread_points=effective_spread_points,
        slippage_points=req.slippage_points,
        commission_per_lot=req.commission_per_lot,
        point_value=point_value,
        contract_size=contract_size,
        tick_size=tick_size,
        tick_value=tick_value,
        lot_step=volume_step,
        min_lot=min_lot,
        max_lot=max_lot,
        db=db,
    )
    bt_res = engine.run(df_feat, signals, strategy_name=req.strategy_name)

    # 5. Run Stress Scenarios
    stress_res = BacktestStressTester.run_stress_scenarios(
        df_feat, signals, symbol=req.symbol, timeframe=req.timeframe.value,
        initial_balance=req.initial_balance,
        base_spread_points=effective_spread_points,
        point_value=float(symbol_info.point),
        contract_size=float(symbol_info.trade_contract_size),
        tick_size=float(symbol_info.tick_size),
        tick_value=float(symbol_info.tick_value),
        lot_step=float(symbol_info.volume_step),
        min_lot=float(symbol_info.volume_min),
        max_lot=float(symbol_info.volume_max),
        base_slippage_points=req.slippage_points,
        commission_per_lot=req.commission_per_lot,
    )

    # 6. Generate Markdown Report
    report_md = BacktestReportGenerator.generate_markdown(bt_res, stress_res)

    return {
        "run_id": bt_res["run_id"],
        "strategy_name": bt_res["strategy_name"],
        "metrics": bt_res["metrics"],
        "stress_test": stress_res,
        "assumptions": bt_res["assumptions"],
        "limitations": bt_res["limitations"],
        "trades_count": len(bt_res["trades"]),
    }
