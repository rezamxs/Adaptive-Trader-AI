"""Backtesting API Endpoints."""
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np

from backend.app.db.session import get_db
from backend.app.models.backtest import BacktestRun
from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider
from backend.app.services.market_data.mock_provider import MockMarketDataProvider
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
    bars_count: int = 300
    spread_points: int = 15
    slippage_points: int = 3
    commission_per_lot: float = 7.0
    initial_balance: float = 10000.0


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
        provider = MockMarketDataProvider()

    try:
        df_raw = provider.get_historical_dataframe(req.symbol, req.timeframe, count=req.bars_count)
    finally:
        provider.disconnect()

    if len(df_raw) < 50:
        raise HTTPException(status_code=400, detail="Insufficient bars to run backtest.")

    # 1. Transform features
    df_feat, _ = FeatureEngineer.transform(df_raw)

    # 2. Get approved rules
    km = KnowledgeManager(db)
    approved_rules = km.get_active_strategy_rules()

    # 3. Generate signals for each bar
    signals = pd.Series(np.zeros(len(df_feat)), index=df_feat.index)
    for i in range(len(df_feat)):
        bar = df_feat.iloc[i]
        if approved_rules:
            direction, _, _ = SignalGenerator.evaluate_rules(bar, approved_rules)
            if direction == "BUY":
                signals.iloc[i] = 1
            elif direction == "SELL":
                signals.iloc[i] = -1
        else:
            # Fallback to simple RSI rule if no rules are approved yet
            rsi = bar.get("rsi_14", 50.0)
            if rsi < 30:
                signals.iloc[i] = 1
            elif rsi > 70:
                signals.iloc[i] = -1

    # 4. Run Backtest
    engine = RealisticBacktestEngine(
        symbol=req.symbol,
        timeframe=req.timeframe.value,
        initial_balance=req.initial_balance,
        spread_points=req.spread_points,
        slippage_points=req.slippage_points,
        commission_per_lot=req.commission_per_lot,
        db=db,
    )
    bt_res = engine.run(df_feat, signals, strategy_name=req.strategy_name)

    # 5. Run Stress Scenarios
    stress_res = BacktestStressTester.run_stress_scenarios(
        df_feat, signals, symbol=req.symbol, timeframe=req.timeframe.value, initial_balance=req.initial_balance
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
