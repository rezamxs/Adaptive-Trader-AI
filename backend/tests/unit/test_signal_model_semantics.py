"""Regression tests for binary BUY-success model interpretation."""
import numpy as np
import pandas as pd

from backend.app.services.signal.generator import SignalGenerator


class FakeBinaryModel:
    def __init__(self, probabilities, classes=(0, 1)):
        self.probabilities = np.asarray([probabilities], dtype=float)
        self.classes_ = np.asarray(classes)

    def predict_proba(self, features):
        return self.probabilities


def _artifact(probabilities, classes=(0, 1)):
    return {
        "model": FakeBinaryModel(probabilities, classes),
        "scaler": None,
        "feature_names": ["rsi_14"],
    }


def test_high_buy_success_probability_emits_buy():
    direction, confidence = SignalGenerator.evaluate_model(
        pd.Series({"rsi_14": 30.0}), _artifact([0.2, 0.8]), prob_threshold=0.55,
    )
    assert direction == "BUY"
    assert confidence == 0.8


def test_low_buy_success_probability_does_not_emit_sell():
    direction, confidence = SignalGenerator.evaluate_model(
        pd.Series({"rsi_14": 70.0}), _artifact([0.8, 0.2]), prob_threshold=0.55,
    )
    assert direction == "HOLD"
    assert confidence == 0.2


def test_positive_probability_uses_class_label_not_fixed_column():
    direction, confidence = SignalGenerator.evaluate_model(
        pd.Series({"rsi_14": 30.0}), _artifact([0.85, 0.15], classes=(1, 0)),
        prob_threshold=0.55,
    )
    assert direction == "BUY"
    assert confidence == 0.85


def test_missing_positive_class_fails_closed():
    direction, confidence = SignalGenerator.evaluate_model(
        pd.Series({"rsi_14": 30.0}), _artifact([1.0], classes=(0,)),
        prob_threshold=0.55,
    )
    assert direction is None
    assert confidence == 0.0



def test_approved_sell_rule_can_act_when_buy_only_model_is_neutral():
    direction, confidence = SignalGenerator.resolve_consensus(
        rule_direction="SELL",
        rule_strength=1.0,
        model_direction="HOLD",
        model_strength=0.20,
        approved_rule_count=2,
    )
    assert direction == "SELL"
    assert confidence == 1.0


def test_model_buy_conflict_blocks_approved_sell_rule():
    direction, confidence = SignalGenerator.resolve_consensus(
        rule_direction="SELL",
        rule_strength=1.0,
        model_direction="BUY",
        model_strength=0.80,
        approved_rule_count=2,
    )
    assert direction == "HOLD"
    assert confidence == 0.0


def test_low_buy_probability_never_creates_sell_without_sell_rule():
    direction, confidence = SignalGenerator.resolve_consensus(
        rule_direction="HOLD",
        rule_strength=0.0,
        model_direction="HOLD",
        model_strength=0.20,
        approved_rule_count=0,
    )
    assert direction == "HOLD"
    assert confidence == 0.0
