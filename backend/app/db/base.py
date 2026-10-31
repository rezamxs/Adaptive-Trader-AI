"""SQLAlchemy Base and Declarative metadata."""
from datetime import datetime, timezone
from sqlalchemy import DateTime, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now_naive() -> datetime:
    """Naive UTC timestamp — datetime.utcnow() semantics without the
    DeprecationWarning (removed in a future Python). DB columns stay
    naive-UTC by contract: the dashboard's asUtcIso normalizer relies on it.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps."""
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive, onupdate=utc_now_naive, nullable=False)
