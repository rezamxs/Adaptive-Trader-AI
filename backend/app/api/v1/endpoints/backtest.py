"""Backtesting API Endpoints."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import pandas as pd
import math
import numpy as np

from backend.app.db.session import get_db
from backend.app.core.auth import require_api_token
from backend.app.models.backtest import BacktestRun
from backend.app.services.market_data.provider_manager import (
    acquire_shared_provider,
    release_shared_provider,
)
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.dataset.features import FeatureEngineer
from backend.app.services.backtest.engine import RealisticBacktestEngine
from backend.app.services.backtest.stress import BacktestStressTester
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
    # M8: when set, the run is a REAL walk-forward evaluation (chronological
    # out-of-sample folds by the same engine/costs) whose persisted run
    # carries the provenance block the model-promotion gate verifies.
    walk_forward: bool = Field(default=False)
    wf_fold_count: int = Field(default=5, ge=2, le=20)


@router.get("/runs")
def list_backtest_runs(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    _token: str = Depends(require_api_token),
):
    """Lists past backtest runs (M15: strategy performance evidence — gated)."""
    runs = db.query(BacktestRun).order_by(BacktestRun.id.desc()).limit(limit).all()
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
def execute_backtest(req: RunBacktestRequest, db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Executes realistic backtest with cost modeling and stress analysis."""
    provider = acquire_shared_provider()
    if not provider.connect():
        release_shared_provider(provider)
        raise HTTPException(
            status_code=503,
            detail="MT5 is unavailable; backtesting with synthetic fallback data is disabled.",
        )

    try:
        df_raw = provider.get_historical_dataframe(req.symbol, req.timeframe, count=req.bars_count)
        symbol_info = provider.get_symbol_info(req.symbol)
    finally:
        release_shared_provider(provider)

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
        if not historical_spread.empty:
            effective_spread_points = max(0, int(round(float(historical_spread.median()))))
        else:
            # symbol_info.spread may be None/invalid on broker failures — the
            # spec fields above are guarded, so guard this one the same way
            # instead of leaking a raw TypeError as a 500.
            try:
                broker_spread = int(symbol_info.spread)
            except (TypeError, ValueError, OverflowError):
                raise HTTPException(
                    status_code=503,
                    detail=f"MT5 returned invalid spread data for {req.symbol}.",
                )
            effective_spread_points = max(0, broker_spread)
    else:
        effective_spread_points = req.spread_points

    # 1. Transform features
    df_feat, _ = FeatureEngineer.transform(df_raw)

    # 2. A backtest must represent an explicitly approved strategy, never an
    # implicit fallback heuristic that the user has not reviewed or approved.
    # M22: a missing approval is a precondition on SERVER state (409), not a
    # malformed request body (400).
    km = KnowledgeManager(db)
    approved_rules = km.get_active_strategy_rules()
    if not approved_rules:
        raise HTTPException(
            status_code=409,
            detail="No approved strategy rules are available; approve a strategy before backtesting.",
        )

    # 2b. M8 walk-forward mode: a REAL evaluation over chronological OOS
    # folds by the same engine/costs, persisted with the provenance block the
    # model-promotion gate verifies. Returns the payload ready for
    # ModelRegistry.attach_walk_forward_metrics.
    if req.walk_forward:
        from backend.app.services.backtest.walk_forward import run_walk_forward_backtest
        try:
            wf_result = run_walk_forward_backtest(
                db, df_raw, symbol=req.symbol, timeframe=req.timeframe.value,
                approved_rules=approved_rules, fold_count=req.wf_fold_count,
                initial_balance=req.initial_balance,
                spread_points=effective_spread_points,
                slippage_points=req.slippage_points,
                commission_per_lot=req.commission_per_lot,
                point_value=point_value, contract_size=contract_size,
                tick_size=tick_size, tick_value=tick_value,
                lot_step=volume_step, min_lot=min_lot, max_lot=max_lot,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "run_id": wf_result["run_id"],
            "strategy_name": "walk_forward",
            "walk_forward_metrics": wf_result["walk_forward_metrics"],
            "fold_run_ids": wf_result["fold_run_ids"],
        }

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

    return {
        "run_id": bt_res["run_id"],
        "strategy_name": bt_res["strategy_name"],
        "metrics": bt_res["metrics"],
        "stress_test": stress_res,
        "assumptions": bt_res["assumptions"],
        "limitations": bt_res["limitations"],
        "trades_count": len(bt_res["trades"]),
    }
