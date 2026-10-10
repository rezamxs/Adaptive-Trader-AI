"""DatasetVersion Model for reproducible time-series ML pipelines."""
from datetime import datetime
from typing import Any, Dict, List
from sqlalchemy import String, Integer, DateTime, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.db.base import Base


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_tag: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    total_bars: Mapped[int] = mapped_column(Integer, nullable=False)
    
    features_list: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    target_definition: Mapped[str] = mapped_column(Text, nullable=False)
    target_stats: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    
    file_path: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
