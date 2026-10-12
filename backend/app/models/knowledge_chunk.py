"""KnowledgeChunk Model for storing granular, sourced learning material with full content preservation."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import String, Integer, DateTime, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.db.base import Base


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    document_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    source_file: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    location_type: Mapped[str] = mapped_column(String(32), nullable=False)  # page, session, table_row, archive_member
    location_reference: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Structured retrieval facets
    key_concepts: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    parameters: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    exceptions_and_limits: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
