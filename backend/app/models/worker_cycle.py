"""Model for tracking Market Monitoring Worker execution cycles and preventing duplicate closed bar processing."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Index
from backend.app.db.base import Base, utc_now_naive


class WorkerCycleRecord(Base):
    """Persistent audit record of a single market monitoring worker iteration."""
    __tablename__ = "worker_cycle_records"

    id = Column(Integer, primary_key=True, index=True)
    cycle_id = Column(String(50), unique=True, index=True, nullable=False)
    symbol = Column(String(20), index=True, nullable=False)
    timeframe = Column(String(10), nullable=False)
    bar_key = Column(String(80), unique=True, index=True, nullable=True)  # Deduplication key: e.g. "XAUUSD_M1_202610092354"
    bar_time = Column(DateTime, nullable=True)
    status_code = Column(String(50), index=True, nullable=False)  # e.g. PROCESSED, NO_NEW_CLOSED_BAR, MARKET_CLOSED, NO_ACTIVE_MODEL
    decision = Column(String(20), nullable=True)  # BUY, SELL, HOLD, NONE
    signal_id = Column(String(50), nullable=True)
    duration_ms = Column(Float, nullable=False, default=0.0)
    details = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    # Buglist item 26: server_default matches migration 20261029_wcbf so a
    # fresh create_all schema equals the migrated one (the wcbf rebuild set
    # the server-side default; the model previously declared only the ORM default).
    lifecycle_phase = Column(String(30), default="COMPLETED", server_default="COMPLETED", nullable=False)  # DISCOVERED, PROCESSING, COMPLETED, FAILED
    created_at = Column(DateTime, default=utc_now_naive, nullable=False)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=utc_now_naive, nullable=False)

    __table_args__ = (
        Index("idx_worker_sym_tf_bar", "symbol", "timeframe", "bar_time"),
    )
