"""MLModelRecord Model for tracking model experiments, versioning, and promotions."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import String, Integer, Float, DateTime, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.db.base import Base, utc_now_naive


class MLModelRecord(Base):
    __tablename__ = "ml_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    dataset_version_tag: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)  # logistic_regression, random_forest, decision_tree
    hyperparameters: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    features_used: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    target_name: Mapped[str] = mapped_column(String(64), nullable=False)
    
    # Validation & Evaluation Metrics
    train_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    validation_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    test_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    calibration_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    walk_forward_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    # M8: per-fold-retrained ML walk-forward evidence (classification-metric
    # generalization). Attached and digest-attested via
    # ModelRegistry.attach_ml_walk_forward_metrics; re-verified at promotion.
    ml_walk_forward_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    
    # Model Lifecycle Status
    status: Mapped[str] = mapped_column(String(32), default="candidate", index=True, nullable=False)  # candidate, active, archived, rejected
    artifact_path: Mapped[str] = mapped_column(String(255), nullable=False)
    promotion_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # H1: attestation digest over (model_id, validation/test/walk-forward metrics),
    # committed at registration and re-verified at promotion time.
    metrics_digest: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Minor-17 hardening: SHA-256 over the artifact FILE BYTES, recorded at
    # every joblib.dump and verified BEFORE any joblib.load (the runtime load
    # path, the attach path, and the promotion artifact gates). joblib.load
    # executes pickle code, so byte integrity must be proven BEFORE the load,
    # not after. Legacy rows without a digest fail closed on load.
    artifact_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive, onupdate=utc_now_naive, nullable=False)
