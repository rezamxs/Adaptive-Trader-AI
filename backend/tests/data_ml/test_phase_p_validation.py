"""Unit tests specifically covering Phase P validation requirements.
Verifies target consistency, class imbalance dominance, threshold mechanics,
balanced accuracy properties, and fail-closed safety constraints.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from backend.app.core.config import settings, ExecutionMode
from backend.app.services.dataset.labeling import BarrierLabeler
from backend.app.services.ml.evaluator import ModelEvaluator
from backend.app.services.ml.models import BaselineModel, RandomForestStrategyModel


def test_majority_class_baseline_accuracy_dominance():
    """Demonstrates why raw accuracy is deceptive on imbalanced datasets:
    Predicting all 0s on an 80/20 imbalanced dataset yields 80% accuracy
    while having 0 Recall, 0 F1, and a Balanced Accuracy of exactly 50%.
    """
    y_true = np.array([0] * 80 + [1] * 20)
    baseline = BaselineModel(strategy="most_frequent")
    X_dummy = np.zeros((100, 1))
    baseline.fit(X_dummy, y_true)
    y_pred = baseline.predict(X_dummy)
    
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    assert acc == 0.80
    assert bacc == 0.50
    assert rec == 0.00
    assert f1 == 0.00


def test_decision_threshold_sensitivity_on_recall():
    """Verifies that an imbalanced model's low recall at default threshold 0.50
    is a thresholding artifact that shifts as threshold is adjusted.
    """
    np.random.seed(42)
    # 70% negatives, 30% positives
    y_true = np.array([0] * 70 + [1] * 30)
    # Probabilities centered around prior (~0.30)
    probs = np.clip(np.random.normal(loc=0.30, scale=0.10, size=100), 0.01, 0.99)
    # Shift positives slightly higher
    probs[y_true == 1] += 0.10
    
    # At default threshold 0.50
    pred_50 = (probs >= 0.50).astype(int)
    rec_50 = recall_score(y_true, pred_50, zero_division=0)
    
    # At prior-adjusted threshold 0.35
    pred_35 = (probs >= 0.35).astype(int)
    rec_35 = recall_score(y_true, pred_35, zero_division=0)
    
    assert rec_35 > rec_50
    assert rec_50 < 0.40  # Low recall at 0.50


def test_barrier_labeler_tail_truncation():
    """Verifies that the final max_horizon_bars rows strictly receive NaN
    because future price progression is incomplete.
    """
    n = 50
    horizon = 10
    df = pd.DataFrame({
        "close": np.linspace(100, 110, n),
        "high": np.linspace(101, 111, n),
        "low": np.linspace(99, 109, n),
        "atr_14": np.full(n, 1.0)
    })
    
    labels = BarrierLabeler.label_buy_barrier(df, max_horizon_bars=horizon)
    
    # First n - horizon bars should be valid numbers
    assert not labels.iloc[:n - horizon].isna().any()
    # The last horizon bars must strictly be NaN
    assert labels.iloc[n - horizon:].isna().all()


def test_fail_closed_mode_signal_only():
    """Verifies that system is strictly configured in SIGNAL_ONLY mode
    and demo trading is disabled.
    """
    assert settings.EXECUTION_MODE == ExecutionMode.SIGNAL_ONLY
    assert settings.DEMO_TRADING_ENABLED is False
