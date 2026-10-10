"""DemoOrder Model for tracking orders sent strictly to Demo accounts."""
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, Float, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.db.base import Base


class DemoOrder(Base):
    __tablename__ = "demo_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    signal_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    
    mt5_ticket: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    order_type: Mapped[str] = mapped_column(String(16), nullable=False)  # BUY, SELL
    
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    requested_price: Mapped[float] = mapped_column(Float, nullable=False)
    executed_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    slippage_points: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False)
    take_profit: Mapped[float] = mapped_column(Float, nullable=False)
    
    status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True, nullable=False)  # PENDING, FILLED, REJECTED, CANCELLED, CLOSED
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    commission: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    swap: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    profit: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
