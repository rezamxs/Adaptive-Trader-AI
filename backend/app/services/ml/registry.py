"""Model Governance, Versioning, and Controlled Promotion Workflow."""
from pathlib import Path
from typing import Any, Dict, List, Optional
import math
import joblib
from sqlalchemy.orm import Session

from backend.app.core.logging import logger, log_security_event
from backend.app.core.errors import ModelGovernanceError
from backend.app.models.ml_model import MLModelRecord
from backend.app.models.audit_log import AuditLog


class ModelRegistry:
    """Oversees model promotions, audits, artifact loading, and lifecycle state."""

    # Keep these gates aligned with the minimum evidence accepted by compare_models.
    MIN_PROMOTION_ROC_AUC = 0.52
    MAX_PROMOTION_BRIER_SCORE = 0.25

    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def _metric_value(metrics: Any, key: str) -> Optional[float]:
        """Return a finite probability/score in [0, 1], or None when invalid."""
        if not isinstance(metrics, dict) or key not in metrics:
            return None
        try:
            value = float(metrics[key])
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return None
        return value

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

        cand_auc = self._metric_value(candidate.test_metrics, "roc_auc")
        cand_brier = self._metric_value(candidate.test_metrics, "brier_score")

        if cand_auc is None or cand_brier is None:
            comparison["recommendation"] = (
                "INSUFFICIENT_EVIDENCE: Candidate ROC-AUC or Brier score is missing or invalid."
            )
            return comparison

        if active:
            act_auc = self._metric_value(active.test_metrics, "roc_auc")
            if act_auc is None:
                comparison["recommendation"] = (
                    "INSUFFICIENT_EVIDENCE: Active model ROC-AUC is missing or invalid."
                )
            elif cand_auc > act_auc and cand_brier <= self.MAX_PROMOTION_BRIER_SCORE:
                comparison["recommendation"] = "PROCEED_TO_BACKTEST: Candidate exhibits superior Out-of-Time AUC."
            else:
                comparison["recommendation"] = "RETAIN_ACTIVE: Candidate does not demonstrate statistically superior performance."
        else:
            if cand_auc > self.MIN_PROMOTION_ROC_AUC and cand_brier <= self.MAX_PROMOTION_BRIER_SCORE:
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
        if candidate.status != "candidate":
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id} from status {candidate.status!r}; "
                "only candidates can be promoted."
            )

        # A model must have a readable artifact before the active model is touched.
        artifact_path = Path(candidate.artifact_path)
        if not artifact_path.is_file():
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: artifact file is missing."
            )

        metrics = candidate.test_metrics
        cand_auc = self._metric_value(metrics, "roc_auc")
        cand_brier = self._metric_value(metrics, "brier_score")
        if cand_auc is None or cand_brier is None:
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: valid test ROC-AUC and Brier score are required."
            )
        if cand_auc <= self.MIN_PROMOTION_ROC_AUC or cand_brier > self.MAX_PROMOTION_BRIER_SCORE:
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: test metrics do not pass the minimum gate "
                f"(ROC-AUC must be > {self.MIN_PROMOTION_ROC_AUC:.2f}; "
                f"Brier score must be <= {self.MAX_PROMOTION_BRIER_SCORE:.2f})."
            )

        # When an active model exists, fail closed if its score is unavailable or
        # the candidate does not improve the active model's recorded ROC-AUC.
        active_current = self.get_active_model_record()
        if active_current and active_current.model_id != candidate_id:
            active_auc = self._metric_value(active_current.test_metrics, "roc_auc")
            if active_auc is None:
                raise ModelGovernanceError(
                    "Cannot promote candidate: the active model has no valid ROC-AUC for comparison."
                )
            if cand_auc <= active_auc:
                raise ModelGovernanceError(
                    f"Cannot promote model {candidate_id}: ROC-AUC {cand_auc:.4f} "
                    f"does not exceed active model ROC-AUC {active_auc:.4f}."
                )

        # Mutate model states only after every promotion gate has passed.
        if active_current and active_current.model_id != candidate_id:
            active_current.status = "archived"
            logger.info(f"Archived previous active model {active_current.model_id}.")

        candidate.status = "active"
        candidate.promotion_notes = notes or "Promoted after test-metric gates; backtest approval must be verified separately."
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
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
