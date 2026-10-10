"""Signal Generation and Demo Execution Endpoints."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.session import get_db
from backend.app.models.signal import TradeSignal
from backend.app.models.order import DemoOrder
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider
from backend.app.services.market_data.mock_provider import MockMarketDataProvider
from backend.app.services.signal.manager import SignalManager
from backend.app.services.execution.demo_executor import DemoTradingExecutor

router = APIRouter()


class GenerateSignalRequest(BaseModel):
    symbol: str = "EURUSD"
    timeframe: TimeFrame = TimeFrame.M15


@router.get("/")
def list_signals(limit: int = 50, db: Session = Depends(get_db)):
    """Lists recent trade signals."""
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
def generate_signal(req: GenerateSignalRequest, db: Session = Depends(get_db)):
    """Evaluates latest bar and generates a signal in current execution mode."""
    provider = MT5MarketDataProvider()
    if not provider.connect():
        provider = MockMarketDataProvider()

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
        provider.disconnect()


@router.get("/orders")
def list_demo_orders(limit: int = 50, db: Session = Depends(get_db)):
    """Lists executed demo orders."""
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
def execute_demo_order(signal_id: str, db: Session = Depends(get_db)):
    """Dispatches an approved signal to DEMO account through the isolated demo executor."""
    sig = db.query(TradeSignal).filter_by(signal_id=signal_id).first()
    if not sig:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found.")

    provider = MT5MarketDataProvider()
    if not provider.connect():
        provider = MockMarketDataProvider()

    try:
        executor = DemoTradingExecutor(db=db, data_provider=provider)
        success, order, msg = executor.execute_demo_signal(sig)
        if not success:
            raise HTTPException(status_code=400, detail=msg)

        return {
            "status": "SUCCESS",
            "order_id": order.order_id,
            "mt5_ticket": order.mt5_ticket,
            "executed_price": order.executed_price,
            "volume": order.volume,
            "message": msg,
        }
    finally:
        provider.disconnect()
