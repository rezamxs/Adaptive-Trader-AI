"""TradingRule Model with mandatory human approval workflow and source tracing."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import String, Integer, Float, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.db.base import Base, utc_now_naive


class TradingRule(Base):
    __tablename__ = "trading_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rule_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(32), index=True, nullable=False)  # entry, exit, filter, risk
    # M4: explicit direction — the ONLY source of rule direction. Never inferred
    # from the rule name. NULL (undecided) rules never cast a direction vote.
    direction: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)  # 'BUY' | 'SELL' | NULL
    # Step 3.3: extractor suggestion from the document TEXT (never the name).
    # Informational only — the generator reads `direction`, never this field.
    suggested_direction: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)  # 'BUY' | 'SELL' | NULL
    description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Structured rule components
    conditions: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    parameters: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    symbol_scope: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    timeframe_scope: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    invalidation_conditions: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    
    # Source provenance
    # Buglist item 13: DB-enforced reference to the source document. SET NULL
    # keeps an approved rule alive even if its source document row vanishes.
    source_document: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("documents.document_id", ondelete="SET NULL"),
        nullable=True)
    source_pages: Mapped[List[int]] = mapped_column(JSON, default=list, nullable=False)
    
    # Subjective / Ambiguous terms tracking
    assumptions: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    ambiguities: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Human Approval Status (Only "approved" can be activated)
    approval_status: Mapped[str] = mapped_column(String(32), default="pending", index=True, nullable=False)  # pending, approved, rejected
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive, onupdate=utc_now_naive, nullable=False)
