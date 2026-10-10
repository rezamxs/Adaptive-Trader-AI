"""Interpretable Machine Learning Models and Strategy Baselines."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier


class IStrategyModel(ABC):
    """Abstract interface for all predictive models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "IStrategyModel":
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        pass


class BaselineModel(IStrategyModel):
    """Rule-based dummy baseline predicting majority class or constant probability."""

    def __init__(self, strategy: str = "prior"):
        self.strategy = strategy
        self.model = DummyClassifier(strategy=strategy)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return {"algorithm": "baseline_dummy", "strategy": self.strategy}


class LogisticRegressionStrategyModel(IStrategyModel):
    """Interpretable, well-calibrated linear model."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, class_weight: Optional[str] = "balanced"):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=42
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionStrategyModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "algorithm": "logistic_regression",
            "C": self.C,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
        }


class RandomForestStrategyModel(IStrategyModel):
    """Non-linear tree ensemble with bounded depth to prevent overfitting."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 5, min_samples_leaf: int = 10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestStrategyModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "algorithm": "random_forest",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
        }
