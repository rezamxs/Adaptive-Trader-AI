"""Signal Generator synthesizing approved trading rules and active ML model inference."""
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from backend.app.models.rule import TradingRule
from backend.app.services.dataset.features import FeatureEngineer


class SignalGenerator:
    """Generates directional trading signals from rules and predictive models."""

    @staticmethod
    def evaluate_rules(
        latest_bar_features: pd.Series,
        approved_rules: List[TradingRule]
    ) -> Tuple[str, float, List[str]]:
        """Evaluates approved rules on the latest bar features.
        
        Returns:
            (direction, strength, triggered_rule_ids)
        """
        triggered = []
        buy_votes = 0
        sell_votes = 0

        for r in approved_rules:
            # Check conditions
            is_rule_active = True
            for cond in r.conditions:
                indicator = cond.get("indicator")
                op = cond.get("operator")
                thresh = cond.get("threshold")

                if indicator == "RSI":
                    rsi_val = latest_bar_features.get("rsi_14", 50.0)
                    if op == "<" and not (rsi_val < float(thresh)):
                        is_rule_active = False
                    elif op == ">" and not (rsi_val > float(thresh)):
                        is_rule_active = False

                elif indicator == "EMA":
                    dist = latest_bar_features.get("dist_ema_21", 0.0)
                    if op == ">" and dist <= 0:
                        is_rule_active = False
                    elif op == "<" and dist >= 0:
                        is_rule_active = False

            if is_rule_active and len(r.conditions) > 0:
                triggered.append(r.rule_id)
                if r.category == "entry":
                    if "buy" in r.name.lower() or "خرید" in r.name:
                        buy_votes += 1
                    elif "sell" in r.name.lower() or "فروش" in r.name:
                        sell_votes += 1

        if buy_votes > sell_votes:
            return "BUY", 1.0, triggered
        elif sell_votes > buy_votes:
            return "SELL", 1.0, triggered
        return "HOLD", 0.0, triggered

    @staticmethod
    def evaluate_model(
        latest_bar_features: pd.Series,
        model_artifact: Optional[Dict[str, Any]],
        prob_threshold: float = 0.55
    ) -> Tuple[Optional[str], float]:
        """Runs inference using active ML model on scaled feature vector."""
        if not model_artifact:
            return None, 0.0

        model = model_artifact.get("model")
        scaler = model_artifact.get("scaler")
        feat_names = model_artifact.get("feature_names", [])

        if not model or not feat_names:
            return None, 0.0

        raw_vec = np.array([latest_bar_features.get(f, 0.0) for f in feat_names]).reshape(1, -1)
        scaled_vec = scaler.transform(raw_vec) if scaler else raw_vec

        # Training class 1 means BUY reaches TP before SL; class 0 means that
        # BUY setup fails or expires. A low class-1 probability is not a SELL label.
        try:
            proba = model.predict_proba(scaled_vec)
            if getattr(proba, "ndim", 0) != 2 or proba.shape[0] != 1:
                return None, 0.0

            classes = getattr(model, "classes_", None)
            if classes is not None:
                class_labels = list(classes)
                if 1 not in class_labels:
                    return None, 0.0
                positive_index = class_labels.index(1)
            else:
                if proba.shape[1] != 2:
                    return None, 0.0
                positive_index = 1

            if positive_index >= proba.shape[1]:
                return None, 0.0
            pos_prob = float(proba[0, positive_index])
            if not np.isfinite(pos_prob) or not 0.0 <= pos_prob <= 1.0:
                return None, 0.0
        except (AttributeError, IndexError, TypeError, ValueError):
            return None, 0.0

        if pos_prob >= prob_threshold:
            return "BUY", round(pos_prob, 4)
        # Class 0 has no directional SELL meaning. Remain neutral instead.
        return "HOLD", round(pos_prob, 4)
