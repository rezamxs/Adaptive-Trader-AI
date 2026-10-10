"""EmergencyStopState Model for persistent, server-restart resilient safety locking."""
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, Boolean, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.db.base import Base


class EmergencyStopState(Base):
    __tablename__ = "emergency_stop_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    activated_by: Mapped[str] = mapped_column(String(64), default="SYSTEM", nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    
    activated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    deactivated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    deactivated_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
