"""AuditLog Model for persistent recording of security and administrative events."""
from datetime import datetime
from typing import Any, Dict
from sqlalchemy import String, Integer, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.db.base import Base, utc_now_naive


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    severity: Mapped[str] = mapped_column(String(16), default="INFO", nullable=False)  # INFO, WARNING, ERROR, CRITICAL
    actor: Mapped[str] = mapped_column(String(64), default="SYSTEM", nullable=False)
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    # indexed: retention pruning and time-range queries (migration 20261031_auidx)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive, nullable=False, index=True)
