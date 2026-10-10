"""Document Model for storing trading knowledge source files."""
from datetime import datetime
from sqlalchemy import String, Integer, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)  # pdf, txt, md, csv, xlsx
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    source_description: Mapped[str] = mapped_column(String(255), nullable=True)
    processing_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)  # pending, completed, failed
    pages_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
