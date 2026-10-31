"""TradeSignal Model with strict idempotency and audit attributes."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import String, Integer, Float, DateTime, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.db.base import Base, utc_now_naive


class TradeSignal(Base):
    __tablename__ = "trade_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    
    symbol: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)  # BUY, SELL, HOLD
    strength: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False)
    take_profit: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Decisions and rules provenance
    rules_triggered: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    model_version_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    features_snapshot: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    
    # Risk verification outcome
    risk_status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False)  # APPROVED, REJECTED
    risk_rejection_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    calculated_volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # System mode under which signal was generated
    execution_mode: Mapped[str] = mapped_column(String(32), default="SIGNAL_ONLY", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="NEW", nullable=False)
    
    bar_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive, nullable=False)
