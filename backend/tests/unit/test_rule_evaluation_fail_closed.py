"""Approved-rule evaluation must reject unsupported logic and out-of-scope rules."""
from types import SimpleNamespace

import pandas as pd

from backend.app.services.signal.generator import SignalGenerator


def _rule(name, conditions, *, symbol_scope=None, timeframe_scope=None, rule_id="r1"):
    return SimpleNamespace(
        rule_id=rule_id,
        name=name,
        category="entry",
        conditions=conditions,
        symbol_scope=symbol_scope or [],
        timeframe_scope=timeframe_scope or [],
    )


def test_supported_rsi_rule_can_vote_buy():
    rule = _rule("RSI Buy", [{"indicator": "RSI", "operator": "<", "threshold": 30}])
    direction, strength, triggered = SignalGenerator.evaluate_rules(
        pd.Series({"rsi_14": 25.0}), [rule], symbol="XAUUSD", timeframe="M1"
    )
    assert direction == "BUY"
    assert strength == 1.0
    assert triggered == ["r1"]


def test_unknown_indicator_does_not_trigger_rule():
    rule = _rule("Buy", [{"indicator": "STOCHASTIC", "operator": "<", "threshold": 20}])
    direction, _, triggered = SignalGenerator.evaluate_rules(
        pd.Series({"rsi_14": 10.0}), [rule], symbol="XAUUSD", timeframe="M1"
    )
    assert direction == "HOLD"
    assert triggered == []


def test_unknown_operator_does_not_trigger_rule():
    rule = _rule("Buy", [{"indicator": "RSI", "operator": "CROSSES", "threshold": 30}])
    direction, _, triggered = SignalGenerator.evaluate_rules(
        pd.Series({"rsi_14": 10.0}), [rule], symbol="XAUUSD", timeframe="M1"
    )
    assert direction == "HOLD"
    assert triggered == []


def test_rule_scope_mismatch_does_not_trigger():
    rule = _rule(
        "RSI Buy", [{"indicator": "RSI", "operator": "<", "threshold": 30}],
        symbol_scope=["EURUSD"], timeframe_scope=["M15"],
    )
    direction, _, triggered = SignalGenerator.evaluate_rules(
        pd.Series({"rsi_14": 10.0}), [rule], symbol="XAUUSD", timeframe="M1"
    )
    assert direction == "HOLD"
    assert triggered == []


def test_ema_condition_uses_threshold_not_only_sign():
    rule = _rule("EMA Buy", [{"indicator": "EMA", "operator": ">", "threshold": 0.1}])
    direction, _, triggered = SignalGenerator.evaluate_rules(
        pd.Series({"dist_ema_21": 0.01}), [rule], symbol="XAUUSD", timeframe="M1"
    )
    assert direction == "HOLD"
    assert triggered == []



def test_documented_ema_expression_uses_the_matching_feature():
    rule = _rule("EMA Buy", [{"indicator": "EMA", "operator": ">", "threshold": "Price > EMA(21)"}])
    direction, _, triggered = SignalGenerator.evaluate_rules(
        pd.Series({"dist_ema_21": 0.01}), [rule], symbol="XAUUSD", timeframe="M1"
    )
    assert direction == "BUY"
    assert triggered == ["r1"]


def test_unsupported_ema_period_fails_closed_instead_of_using_ema21():
    rule = _rule("EMA Buy", [{"indicator": "EMA", "operator": ">", "threshold": "Price > EMA(50)"}])
    direction, _, triggered = SignalGenerator.evaluate_rules(
        pd.Series({"dist_ema_21": 0.50}), [rule], symbol="XAUUSD", timeframe="M1"
    )
    assert direction == "HOLD"
    assert triggered == []


def test_documented_sma50_expression_uses_sma50_feature():
    rule = _rule("SMA Buy", [{"indicator": "SMA", "operator": ">", "threshold": "Price > SMA(50)"}])
    direction, _, triggered = SignalGenerator.evaluate_rules(
        pd.Series({"dist_sma_50": 0.02}), [rule], symbol="XAUUSD", timeframe="M1"
    )
    assert direction == "BUY"
    assert triggered == ["r1"]
