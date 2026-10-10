"""Model Governance, Versioning, and Controlled Promotion Workflow."""
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
from sqlalchemy.orm import Session

from backend.app.core.logging import logger, log_security_event
from backend.app.core.errors import ModelGovernanceError
from backend.app.models.ml_model import MLModelRecord
from backend.app.models.audit_log import AuditLog


class ModelRegistry:
    """Oversees model promotions, audits, artifact loading, and lifecycle state."""

    def __init__(self, db: Session):
        self.db = db

    def get_active_model_record(self) -> Optional[MLModelRecord]:
        """Returns the currently active MLModelRecord, if any."""
        return self.db.query(MLModelRecord).filter(MLModelRecord.status == "active").first()

    def load_active_model_artifact(self) -> Optional[Dict[str, Any]]:
        """Loads and unpickles the active model artifact from disk."""
        active_rec = self.get_active_model_record()
        if not active_rec:
            return None

        path = Path(active_rec.artifact_path)
        if not path.exists():
            logger.error(f"Active model artifact file not found at {path}")
            return None

        return joblib.load(path)

    def list_models(self, status: Optional[str] = None) -> List[MLModelRecord]:
        """Lists models optionally filtered by status."""
        query = self.db.query(MLModelRecord)
        if status:
            query = query.filter(MLModelRecord.status == status)
        return query.order_by(MLModelRecord.id.desc()).all()

    def compare_models(self, candidate_id: str) -> Dict[str, Any]:
        """Compares candidate metrics against currently active model or baseline."""
        candidate = self.db.query(MLModelRecord).filter_by(model_id=candidate_id).first()
        if not candidate:
            raise ValueError(f"Candidate model {candidate_id} not found.")

        active = self.get_active_model_record()

        comparison = {
            "candidate_id": candidate.model_id,
            "candidate_algorithm": candidate.algorithm,
            "candidate_test_metrics": candidate.test_metrics,
            "active_id": active.model_id if active else None,
            "active_algorithm": active.algorithm if active else None,
            "active_test_metrics": active.test_metrics if active else None,
            "recommendation": "UNKNOWN",
        }

        cand_auc = candidate.test_metrics.get("roc_auc", 0.5)
        cand_brier = candidate.test_metrics.get("brier_score", 0.25)

        if active:
            act_auc = active.test_metrics.get("roc_auc", 0.5)
            if cand_auc > act_auc and cand_brier <= 0.25:
                comparison["recommendation"] = "PROCEED_TO_BACKTEST: Candidate exhibits superior Out-of-Time AUC."
            else:
                comparison["recommendation"] = "RETAIN_ACTIVE: Candidate does not demonstrate statistically superior performance."
        else:
            if cand_auc > 0.52 and cand_brier <= 0.25:
                comparison["recommendation"] = "PROCEED_TO_BACKTEST: Candidate outperforms random baseline."
            else:
                comparison["recommendation"] = "INSUFFICIENT_EVIDENCE: Metric evidence is not convincing."

        return comparison

    def promote_candidate(self, candidate_id: str, actor: str = "USER", notes: Optional[str] = None) -> MLModelRecord:
        """Controlled promotion of a candidate model to ACTIVE status.
        Archives previously active model.
        """
        candidate = self.db.query(MLModelRecord).filter_by(model_id=candidate_id).first()
        if not candidate:
            raise ValueError(f"Candidate model {candidate_id} not found.")

        if candidate.status == "rejected":
            raise ModelGovernanceError(f"Cannot promote model {candidate_id} because it was marked REJECTED.")

        # Archive current active model
        active_current = self.get_active_model_record()
        if active_current and active_current.model_id != candidate_id:
            active_current.status = "archived"
            logger.info(f"Archived previous active model {active_current.model_id}.")

        # Set candidate to active
        candidate.status = "active"
        candidate.promotion_notes = notes or "Promoted after multi-metric evaluation against baseline."
        self.db.commit()
        self.db.refresh(candidate)

        log_security_event(
            "MODEL_PROMOTED",
            {
                "model_id": candidate_id,
                "actor": actor,
                "previous_active": active_current.model_id if active_current else None,
            },
            "INFO"
        )
        return candidate
