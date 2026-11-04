"""MLWalkForwardRun: the persisted per-fold-retrained ML walk-forward evaluation.

Closes the M8 self-attestation residual (roadmap 2.4): until now the ML
walk-forward evidence existed only inside the model record it was attached to
and was re-signed at attach time, so nothing anchored it to an independent
persisted artifact. The evaluator's output is now persisted AT GENERATION
time together with its digest; a model can attach ONLY evidence that matches
a persisted, digest-verified run row, and the promotion gate requires that
binding. The row is created once and its ``model_id`` binding is immutable
afterwards.
"""
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import String, Integer, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base, utc_now_naive


class MLWalkForwardRun(Base):
    __tablename__ = "ml_walk_forward_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Stable external identifier carried inside the attached evidence payload.
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    # The digest-STRIPPED evaluation payload exactly as produced by
    # run_ml_walk_forward (JSON-safe plain data).
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    # Tamper-evident digest over the payload, computed at persistence time.
    digest: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_fingerprint: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    fold_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Set once at attach time; immutable afterwards.
    model_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive, nullable=False)
