"""Signal Generation and Demo Execution Endpoints."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.auth import require_api_token
from backend.app.db.session import get_db
from backend.app.models.signal import TradeSignal
from backend.app.models.order import DemoOrder
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.market_data.provider_manager import (
    acquire_shared_provider,
    release_shared_provider,
)
from backend.app.services.signal.manager import SignalManager
from backend.app.services.execution.demo_executor import (
    DemoTradingExecutor,
    EXEC_MSG_DUPLICATE,
    EXEC_MSG_UNRESOLVED,
    EXEC_MSG_SAFETY_GATE,
    EXEC_MSG_RISK_CHECKS,
    EXEC_MSG_RISK_STATUS,
    EXEC_MSG_STALE_QUOTE,
    EXEC_MSG_NO_SPECS,
)

router = APIRouter()


def _execution_failure_status(message: str) -> int:
    """M22: demo-executor failures are heterogeneous — the HTTP status must
    say what kind of failure happened instead of flattening everything to a
    request-body 400. The fragments come from the executor's OWN constants
    (single source of truth in demo_executor.py) — rewording a message there
    updates the mapping automatically.

    409 — conflict with a prior dispatch (duplicate idempotency) or an
          unresolved outcome (EXECUTION_UNKNOWN reconciliation required).
    403 — policy refusal (safety gate, risk-status/risk-check rejection).
    503 — upstream market-data condition (stale quote, missing symbol specs).
    400 — request/record data problems (volume validity).
    """
    m = message or ""
    if EXEC_MSG_DUPLICATE in m or EXEC_MSG_UNRESOLVED in m:
        return 409
    if EXEC_MSG_SAFETY_GATE in m or EXEC_MSG_RISK_CHECKS in m or EXEC_MSG_RISK_STATUS in m:
        return 403
    if EXEC_MSG_STALE_QUOTE in m or EXEC_MSG_NO_SPECS in m:
        return 503
    return 400


class GenerateSignalRequest(BaseModel):
    symbol: str = "EURUSD"
    timeframe: TimeFrame = TimeFrame.M15


@router.get("/")
def list_signals(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Lists recent trade signals (M15: entry/SL/TP levels — gated)."""
    # limit is bounded: SQLite treats LIMIT -1 as unlimited, so an unbounded
    # query param could dump the whole table (memory DoS / bulk exfiltration).
    signals = db.query(TradeSignal).order_by(TradeSignal.id.desc()).limit(limit).all()
    return [{
        "signal_id": s.signal_id,
        "symbol": s.symbol,
        "timeframe": s.timeframe,
        "direction": s.direction,
        "entry_price": s.entry_price,
        "stop_loss": s.stop_loss,
        "take_profit": s.take_profit,
        "strength": s.strength,
        "risk_status": s.risk_status,
        "calculated_volume": s.calculated_volume,
        "execution_mode": s.execution_mode,
        "bar_time": s.bar_time.isoformat(),
        "created_at": s.created_at.isoformat(),
    } for s in signals]


@router.post("/generate")
def generate_signal(req: GenerateSignalRequest, db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Evaluates latest bar and generates a signal in current execution mode."""
    provider = acquire_shared_provider()
    if not provider.connect():
        release_shared_provider(provider)
        raise HTTPException(
            status_code=503,
            detail="MT5 is unavailable; signal generation from synthetic market data is disabled.",
        )

    try:
        manager = SignalManager(db=db, data_provider=provider)
        sig = manager.process_latest_bar(symbol=req.symbol, timeframe=req.timeframe)
        if not sig:
            return {"status": "NO_SIGNAL", "message": "No actionable rule or model conditions met on the latest bar."}

        return {
            "status": "SIGNAL_GENERATED",
            "signal_id": sig.signal_id,
            "direction": sig.direction,
            "symbol": sig.symbol,
            "entry_price": sig.entry_price,
            "stop_loss": sig.stop_loss,
            "take_profit": sig.take_profit,
            "risk_status": sig.risk_status,
            "calculated_volume": sig.calculated_volume,
            "execution_mode": sig.execution_mode,
        }
    finally:
        release_shared_provider(provider)


@router.get("/orders")
def list_demo_orders(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Lists executed demo orders (M15: broker tickets and executed prices — gated)."""
    orders = db.query(DemoOrder).order_by(DemoOrder.id.desc()).limit(limit).all()
    return [{
        "order_id": o.order_id,
        "signal_id": o.signal_id,
        "symbol": o.symbol,
        "order_type": o.order_type,
        "volume": o.volume,
        "requested_price": o.requested_price,
        "executed_price": o.executed_price,
        "status": o.status,
        "mt5_ticket": o.mt5_ticket,
        "error_message": o.error_message,
        "created_at": o.created_at.isoformat(),
    } for o in orders]


@router.post("/{signal_id}/execute-demo")
def execute_demo_order(signal_id: str, db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Dispatches an approved signal to DEMO account through the isolated demo executor."""
    sig = db.query(TradeSignal).filter_by(signal_id=signal_id).first()
    if not sig:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found.")

    provider = acquire_shared_provider()
    if not provider.connect():
        release_shared_provider(provider)
        raise HTTPException(
            status_code=503,
            detail="MT5 is unavailable; demo execution requires a verified terminal and account.",
        )

    try:
        executor = DemoTradingExecutor(db=db, data_provider=provider)
        success, order, msg = executor.execute_demo_signal(sig)
        if not success:
            raise HTTPException(status_code=_execution_failure_status(msg), detail=msg)

        return {
            "status": "SUCCESS",
            "order_id": order.order_id,
            "mt5_ticket": order.mt5_ticket,
            "executed_price": order.executed_price,
            "volume": order.volume,
            "message": msg,
        }
    finally:
        release_shared_provider(provider)
