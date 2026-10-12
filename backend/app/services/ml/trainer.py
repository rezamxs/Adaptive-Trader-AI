"""Model Training and Artifact Serialization."""
import uuid
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.models.ml_model import MLModelRecord
from backend.app.services.ml.models import (
    IStrategyModel,
    BaselineModel,
    LogisticRegressionStrategyModel,
    RandomForestStrategyModel,
)
from backend.app.services.ml.evaluator import ModelEvaluator


class ModelTrainer:
    """Trains, evaluates, and registers models as candidate artifacts."""

    def __init__(self, db: Optional[Session] = None):
        self.db = db

    def train_candidate(
        self,
        dataset_result: Dict[str, Any],
        algorithm: str = "logistic_regression",
        hyperparameters: Optional[Dict[str, Any]] = None,
        target_name: str = "target_tp_before_sl"
    ) -> Dict[str, Any]:
        """Trains a new candidate model, calculates metrics, saves artifacts."""
        hyperparameters = hyperparameters or {}
        
        # Instantiate model
        if algorithm == "logistic_regression":
            model: IStrategyModel = LogisticRegressionStrategyModel(**hyperparameters)
        elif algorithm == "random_forest":
            model = RandomForestStrategyModel(**hyperparameters)
        elif algorithm == "baseline":
            model = BaselineModel(**hyperparameters)
        else:
            raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose 'logistic_regression', 'random_forest', or 'baseline'.")

        X_train, y_train = dataset_result["train_data"]
        X_val, y_val = dataset_result["val_data"]
        X_test, y_test = dataset_result["test_data"]
        feature_names = dataset_result["feature_names"]
        scaler = dataset_result.get("scaler")
        dataset_version_tag = dataset_result["version_tag"]

        # 1. Fit Model
        model.fit(X_train, y_train)

        # 2. Evaluate on Train partition
        train_pred = model.predict(X_train)
        train_prob = model.predict_proba(X_train)
        train_metrics = ModelEvaluator.evaluate(y_train, train_pred, train_prob)

        # 3. Evaluate on held-out temporal validation partition. This partition is
        # distinct from train and test and is required for model-governance review.
        val_pred = model.predict(X_val) if len(X_val) > 0 else np.array([])
        val_prob = model.predict_proba(X_val) if len(X_val) > 0 else np.array([])
        validation_metrics = (
            ModelEvaluator.evaluate(y_val, val_pred, val_prob) if len(X_val) > 0 else {}
        )
        validation_metrics["sample_count"] = int(len(y_val))

        # 4. Evaluate on Independent Out-of-Time Test partition
        test_pred = model.predict(X_test) if len(X_test) > 0 else np.array([])
        test_prob = model.predict_proba(X_test) if len(X_test) > 0 else np.array([])
        test_metrics = ModelEvaluator.evaluate(y_test, test_pred, test_prob) if len(X_test) > 0 else {}

        test_metrics["sample_count"] = int(len(y_test))

        # 5. Compute Calibration Metrics on Test Set
        prob_pos_test = test_prob[:, 1] if test_prob.ndim == 2 else test_prob
        calibration_curve = ModelEvaluator.calculate_calibration_curve(y_test, prob_pos_test) if len(y_test) > 0 else []

        # 6. Serialize Artifact
        model_id = f"model_{algorithm[:6]}_{uuid.uuid4().hex[:8]}"
        artifact_dir = settings.ARTIFACTS_MODELS_DIR
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{model_id}.joblib"

        artifact_payload = {
            "model_id": model_id,
            "algorithm": algorithm,
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "hyperparameters": hyperparameters,
            "dataset_version_tag": dataset_version_tag,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        }
        joblib.dump(artifact_payload, artifact_path)

        # 6. Record Candidate in Database (Strictly as "candidate")
        if self.db:
            record = MLModelRecord(
                model_id=model_id,
                dataset_version_tag=dataset_version_tag,
                algorithm=algorithm,
                hyperparameters=hyperparameters,
                features_used=feature_names,
                target_name=target_name,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                calibration_metrics={"bins": calibration_curve},
                walk_forward_metrics={},
                status="candidate",  # NEVER automatically active
                artifact_path=str(artifact_path),
            )
            self.db.add(record)
            self.db.commit()
            logger.info(f"Registered new model {model_id} as CANDIDATE in database.")

        return {
            "model_id": model_id,
            "status": "candidate",
            "artifact_path": str(artifact_path),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "calibration_curve": calibration_curve,
            "features_used": feature_names,
            "model_instance": model,
        }
