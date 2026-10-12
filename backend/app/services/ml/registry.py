"""Model Governance, Versioning, and Controlled Promotion Workflow."""
from pathlib import Path
from typing import Any, Dict, List, Optional
import math
import joblib
from sqlalchemy.orm import Session

from backend.app.core.logging import logger, log_security_event
from backend.app.core.config import settings
from backend.app.core.errors import ModelGovernanceError
from backend.app.models.ml_model import MLModelRecord
from backend.app.models.audit_log import AuditLog


class ModelRegistry:
    """Oversees model promotions, audits, artifact loading, and lifecycle state."""

    # Keep these gates aligned with the minimum evidence accepted by compare_models.
    MIN_PROMOTION_ROC_AUC = 0.52
    MAX_PROMOTION_BRIER_SCORE = 0.25
    MIN_PROMOTION_EVALUATION_SAMPLES = 30

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

    @classmethod
    def _sample_count(cls, metrics: Any) -> int:
        if not isinstance(metrics, dict):
            return 0
        try:
            count = int(metrics.get("sample_count", 0))
        except (TypeError, ValueError, OverflowError):
            return 0
        return count if count >= 0 else 0

    def get_active_model_record(self) -> Optional[MLModelRecord]:
        """Returns the currently active MLModelRecord, if any."""
        return self.db.query(MLModelRecord).filter(MLModelRecord.status == "active").first()

    def load_active_model_artifact(self) -> Optional[Dict[str, Any]]:
        """Loads and unpickles the active model artifact from disk."""
        active_rec = self.get_active_model_record()
        if not active_rec:
            return None

        path = Path(active_rec.artifact_path).resolve()
        artifact_root = Path(settings.ARTIFACTS_MODELS_DIR).resolve()
        try:
            path.relative_to(artifact_root)
        except ValueError:
            logger.error("Active model artifact path is outside the configured artifact directory.")
            return None
        if not path.is_file() or path.suffix.lower() != ".joblib":
            logger.error(f"Active model artifact file is missing or has an unexpected extension: {path}")
            return None

        try:
            payload = joblib.load(path)
        except Exception as exc:
            logger.error(f"Active model artifact could not be loaded: {exc}")
            return None
        if (
            not isinstance(payload, dict)
            or not callable(getattr(payload.get("model"), "predict_proba", None))
            or not isinstance(payload.get("feature_names"), (list, tuple))
            or not payload.get("feature_names")
        ):
            logger.error("Active model artifact is missing required inference components.")
            return None
        return payload

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
            "candidate_validation_metrics": candidate.validation_metrics,
            "candidate_test_metrics": candidate.test_metrics,
            "active_id": active.model_id if active else None,
            "active_algorithm": active.algorithm if active else None,
            "active_validation_metrics": active.validation_metrics if active else None,
            "active_test_metrics": active.test_metrics if active else None,
            "recommendation": "UNKNOWN",
        }

        cand_val_auc = self._metric_value(candidate.validation_metrics, "roc_auc")
        cand_val_brier = self._metric_value(candidate.validation_metrics, "brier_score")
        cand_test_auc = self._metric_value(candidate.test_metrics, "roc_auc")
        cand_test_brier = self._metric_value(candidate.test_metrics, "brier_score")
        cand_val_n = self._sample_count(candidate.validation_metrics)
        cand_test_n = self._sample_count(candidate.test_metrics)

        if (
            cand_val_auc is None or cand_val_brier is None
            or cand_test_auc is None or cand_test_brier is None
            or cand_val_n < self.MIN_PROMOTION_EVALUATION_SAMPLES
            or cand_test_n < self.MIN_PROMOTION_EVALUATION_SAMPLES
        ):
            comparison["recommendation"] = (
                "INSUFFICIENT_EVIDENCE: Valid validation/test metrics and at least "
                f"{self.MIN_PROMOTION_EVALUATION_SAMPLES} samples per partition are required."
            )
            return comparison

        absolute_gate_passes = (
            cand_val_auc > self.MIN_PROMOTION_ROC_AUC
            and cand_test_auc > self.MIN_PROMOTION_ROC_AUC
            and cand_val_brier <= self.MAX_PROMOTION_BRIER_SCORE
            and cand_test_brier <= self.MAX_PROMOTION_BRIER_SCORE
        )
        if not absolute_gate_passes:
            comparison["recommendation"] = (
                "INSUFFICIENT_EVIDENCE: Validation and independent test metrics must "
                f"both pass ROC-AUC > {self.MIN_PROMOTION_ROC_AUC:.2f} and Brier <= "
                f"{self.MAX_PROMOTION_BRIER_SCORE:.2f}."
            )
            return comparison

        if active:
            active_val_auc = self._metric_value(active.validation_metrics, "roc_auc")
            active_val_n = self._sample_count(active.validation_metrics)
            if active_val_auc is None or active_val_n < self.MIN_PROMOTION_EVALUATION_SAMPLES:
                comparison["recommendation"] = (
                    "INSUFFICIENT_EVIDENCE: Active model has no valid temporal validation metrics; "
                    "re-evaluate it before replacement."
                )
            elif cand_val_auc > active_val_auc:
                comparison["recommendation"] = (
                    "PROCEED_TO_BACKTEST: Candidate validation AUC is superior and independent test gates pass."
                )
            else:
                comparison["recommendation"] = (
                    "RETAIN_ACTIVE: Candidate does not improve the active model on temporal validation."
                )
        else:
            comparison["recommendation"] = (
                "PROCEED_TO_BACKTEST: Validation and independent test metric gates pass."
            )
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
        artifact_path = Path(candidate.artifact_path).resolve()
        artifact_root = Path(settings.ARTIFACTS_MODELS_DIR).resolve()
        try:
            artifact_path.relative_to(artifact_root)
        except ValueError as exc:
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: artifact path is outside the configured artifact directory."
            ) from exc
        if not artifact_path.is_file() or artifact_path.suffix.lower() != ".joblib":
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: artifact file is missing or invalid."
            )

        val_metrics = candidate.validation_metrics
        test_metrics = candidate.test_metrics
        val_auc = self._metric_value(val_metrics, "roc_auc")
        val_brier = self._metric_value(val_metrics, "brier_score")
        test_auc = self._metric_value(test_metrics, "roc_auc")
        test_brier = self._metric_value(test_metrics, "brier_score")
        if (
            val_auc is None or val_brier is None or test_auc is None or test_brier is None
            or self._sample_count(val_metrics) < self.MIN_PROMOTION_EVALUATION_SAMPLES
            or self._sample_count(test_metrics) < self.MIN_PROMOTION_EVALUATION_SAMPLES
        ):
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: validation and test partitions must each have "
                f"valid ROC-AUC/Brier metrics and at least {self.MIN_PROMOTION_EVALUATION_SAMPLES} samples."
            )
        if (
            val_auc <= self.MIN_PROMOTION_ROC_AUC
            or test_auc <= self.MIN_PROMOTION_ROC_AUC
            or val_brier > self.MAX_PROMOTION_BRIER_SCORE
            or test_brier > self.MAX_PROMOTION_BRIER_SCORE
        ):
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: validation and independent test metrics do not pass "
                f"(ROC-AUC must be > {self.MIN_PROMOTION_ROC_AUC:.2f}; "
                f"Brier score must be <= {self.MAX_PROMOTION_BRIER_SCORE:.2f})."
            )

        # Test-set values are reserved for absolute gates. Compare candidates to
        # the active model on validation only to avoid repeatedly tuning against test.
        active_current = self.get_active_model_record()
        if active_current and active_current.model_id != candidate_id:
            active_val_auc = self._metric_value(active_current.validation_metrics, "roc_auc")
            if (
                active_val_auc is None
                or self._sample_count(active_current.validation_metrics) < self.MIN_PROMOTION_EVALUATION_SAMPLES
            ):
                raise ModelGovernanceError(
                    "Cannot promote candidate: the active model must be re-evaluated on a valid temporal validation partition."
                )
            if val_auc <= active_val_auc:
                raise ModelGovernanceError(
                    f"Cannot promote model {candidate_id}: validation ROC-AUC {val_auc:.4f} "
                    f"does not exceed active model validation ROC-AUC {active_val_auc:.4f}."
                )

        # Verify that the persisted artifact is loadable and has the required inference contract.
        try:
            artifact = joblib.load(artifact_path)
        except Exception as exc:
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: artifact cannot be loaded ({type(exc).__name__})."
            ) from exc
        if (
            not isinstance(artifact, dict)
            or not callable(getattr(artifact.get("model"), "predict_proba", None))
            or not isinstance(artifact.get("feature_names"), (list, tuple))
            or not artifact.get("feature_names")
        ):
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: artifact is missing inference components."
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
