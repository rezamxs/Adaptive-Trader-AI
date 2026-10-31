"""Model Evaluation and Probability Calibration Metrics."""
from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)


class ModelEvaluator:
    """Computes comprehensive statistical, classification, and calibration metrics."""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Calculates standard classification and probability metrics."""
        metrics: Dict[str, float] = {}

        metrics["accuracy"] = float(round(accuracy_score(y_true, y_pred), 4))
        metrics["precision"] = float(round(precision_score(y_true, y_pred, zero_division=0), 4))
        metrics["recall"] = float(round(recall_score(y_true, y_pred, zero_division=0), 4))
        metrics["f1"] = float(round(f1_score(y_true, y_pred, zero_division=0), 4))

        # Probability metrics
        if y_prob is not None and len(np.unique(y_true)) > 1:
            # Positive class probability
            prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            try:
                metrics["roc_auc"] = float(round(roc_auc_score(y_true, prob_pos), 4))
            except ValueError:
                metrics["roc_auc"] = 0.5

            try:
                metrics["brier_score"] = float(round(brier_score_loss(y_true, prob_pos), 4))
                metrics["log_loss"] = float(round(log_loss(y_true, prob_pos, labels=[0, 1]), 4))
            except Exception:
                # Fail-closed sentinels: the promotion gate rejects Brier scores
                # strictly above MAX_PROMOTION_BRIER_SCORE (0.25). A sentinel of
                # 0.25 exactly PASSES that gate, so a measurement failure would
                # fabricate passing evidence. 1.0 (worst possible Brier for a
                # binary problem) always fails the gate; roc_auc=0.5 already
                # fails its >0.52 gate the same way.
                metrics["brier_score"] = 1.0
                metrics["log_loss"] = 0.693
        else:
            # Single-class labels cannot produce an honest probability metric:
            # fail closed (see comment above) instead of emitting gate-passing 0.25.
            metrics["roc_auc"] = 0.5
            metrics["brier_score"] = 1.0
            metrics["log_loss"] = 0.693

        return metrics

    @staticmethod
    def calculate_calibration_curve(
        y_true: np.ndarray,
        y_prob_pos: np.ndarray,
        n_bins: int = 5
    ) -> List[Dict[str, float]]:
        """Calculates empirical probability vs predicted probability across bins."""
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_records = []

        for i in range(n_bins):
            mask = (y_prob_pos >= bins[i]) & (y_prob_pos < bins[i + 1])
            if np.any(mask):
                avg_pred = float(np.mean(y_prob_pos[mask]))
                true_freq = float(np.mean(y_true[mask]))
                bin_records.append({
                    "bin": i + 1,
                    "bin_range": f"{bins[i]:.2f}-{bins[i+1]:.2f}",
                    "avg_predicted_prob": round(avg_pred, 4),
                    "true_positive_rate": round(true_freq, 4),
                    "sample_count": int(np.sum(mask)),
                })
        return bin_records
