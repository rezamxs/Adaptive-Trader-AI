"""Signal Generator synthesizing approved trading rules and active ML model inference."""
from datetime import datetime
import re
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
        approved_rules: List[TradingRule],
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> Tuple[str, float, List[str]]:
        """Evaluate approved rules; malformed or out-of-scope rules fail closed.

        Only RSI and EMA-distance conditions are currently supported. Every
        condition must be recognized and satisfied before a rule can vote.
        Direction votes come exclusively from the explicit rule.direction
        field (M4): a rule with no explicit direction triggers but casts no
        vote — the rule name is never inspected.
        """
        triggered: List[str] = []
        buy_votes = 0
        sell_votes = 0
        supported_operators = {
            "<": lambda left, right: left < right,
            "<=": lambda left, right: left <= right,
            ">": lambda left, right: left > right,
            ">=": lambda left, right: left >= right,
            "==": lambda left, right: left == right,
            "!=": lambda left, right: left != right,
        }

        def scope_matches(scope: Any, requested: Optional[str]) -> bool:
            if not scope or requested is None:
                return True
            if not isinstance(scope, (list, tuple, set)):
                return False
            values = {str(item).strip().upper() for item in scope}
            return bool(values.intersection({"*", "ALL", "ANY"}) or str(requested).strip().upper() in values)

        for rule in approved_rules:
            if not scope_matches(getattr(rule, "symbol_scope", None), symbol):
                continue
            if not scope_matches(getattr(rule, "timeframe_scope", None), timeframe):
                continue

            conditions = getattr(rule, "conditions", None)
            if not isinstance(conditions, list) or not conditions:
                continue

            rule_active = True
            for condition in conditions:
                if not isinstance(condition, dict):
                    rule_active = False
                    break

                indicator = str(condition.get("indicator", "")).strip().upper()
                operator = str(condition.get("operator", "")).strip()
                raw_threshold = condition.get("threshold")
                if operator not in supported_operators:
                    rule_active = False
                    break

                if indicator == "RSI":
                    feature_name = "rsi_14"
                elif indicator in {"EMA", "SMA"}:
                    feature_name = "dist_ema_21"
                    # RuleExtractor may encode a moving-average condition as text,
                    # e.g. "Price > EMA(21)". Parse only known periods explicitly.
                    if isinstance(raw_threshold, str):
                        expression = re.sub(r"\s+", "", raw_threshold).lower()
                        match = re.fullmatch(r"(?:price|close)(>|<)(ema|sma)\((\d+)\)", expression)
                        if not match:
                            rule_active = False
                            break
                        expression_operator, average_type, period_text = match.groups()
                        period = int(period_text)
                        if average_type == "ema" and period in {9, 21}:
                            feature_name = f"dist_ema_{period}"
                        elif average_type == "sma" and period == 50:
                            feature_name = "dist_sma_50"
                        else:
                            # Never substitute another moving-average period.
                            rule_active = False
                            break
                        operator = expression_operator
                        raw_threshold = 0.0
                    elif indicator == "SMA":
                        feature_name = "dist_sma_50"
                else:
                    # Unknown indicators must not silently become true conditions.
                    rule_active = False
                    break

                try:
                    threshold = float(raw_threshold)
                    value = float(latest_bar_features.get(feature_name, float("nan")))
                except (KeyError, TypeError, ValueError, OverflowError):
                    rule_active = False
                    break

                if not np.isfinite(threshold) or not np.isfinite(value):
                    rule_active = False
                    break

                try:
                    condition_passes = supported_operators[operator](value, threshold)
                except Exception:
                    condition_passes = False
                if not condition_passes:
                    rule_active = False
                    break

            if not rule_active:
                continue

            triggered.append(rule.rule_id)
            if str(getattr(rule, "category", "")).lower() != "entry":
                continue

            # M4: direction comes ONLY from the explicit rule.direction field.
            # A missing/unknown direction casts no vote (fail-closed); the rule
            # name is never inspected for direction keywords.
            rule_direction = str(getattr(rule, "direction", "") or "").strip().upper()
            if rule_direction == "BUY":
                buy_votes += 1
            elif rule_direction == "SELL":
                sell_votes += 1

        if buy_votes > sell_votes:
            return "BUY", 1.0, triggered
        if sell_votes > buy_votes:
            return "SELL", 1.0, triggered
        return "HOLD", 0.0, triggered

    @staticmethod
    def resolve_consensus(
        rule_direction: str,
        rule_strength: float,
        model_direction: Optional[str],
        model_strength: float,
        approved_rule_count: int,
    ) -> Tuple[str, float]:
        """Resolve rules and the current one-sided BUY-success model safely.

        BUY rules require model BUY agreement when a model produces a directional
        result. SELL may use an approved SELL rule when the BUY-only model is
        neutral, because low BUY-success probability is not evidence of a SELL.
        A direct model/rule conflict always remains HOLD.
        """
        if rule_direction in {"BUY", "SELL"}:
            if model_direction == rule_direction:
                return rule_direction, float(model_strength)
            if model_direction is None:
                return rule_direction, float(rule_strength)
            if rule_direction == "SELL" and model_direction == "HOLD":
                return "SELL", float(rule_strength)
            return "HOLD", 0.0

        if (
            approved_rule_count == 0
            and model_direction == "BUY"
        ):
            return "BUY", float(model_strength)
        return "HOLD", 0.0

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

        if not np.isfinite(prob_threshold) or not 0.0 <= prob_threshold <= 1.0:
            return None, 0.0
        try:
            raw_values = []
            for feature_name in feat_names:
                value = float(latest_bar_features.get(feature_name, float("nan")))
                if not np.isfinite(value):
                    return None, 0.0
                raw_values.append(value)
            raw_vec = np.asarray(raw_values, dtype=float).reshape(1, -1)
            scaled_vec = scaler.transform(raw_vec) if scaler else raw_vec

            # Training class 1 means BUY reaches TP before SL; class 0 means that
            # BUY setup fails or expires. A low class-1 probability is not a SELL label.
            proba = model.predict_proba(scaled_vec)
            if getattr(proba, "ndim", 0) != 2 or proba.shape[0] != 1:
                return None, 0.0

            classes = getattr(model, "classes_", None)
            if classes is None:
                classes = getattr(getattr(model, "model", None), "classes_", None)
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
