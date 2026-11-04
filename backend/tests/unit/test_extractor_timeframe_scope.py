"""Regression for the timeframe-scope extraction substring bug (audit M-4).

The extractor matched timeframes with a plain substring test, so text that
mentions "M15" also detected "M1" (and "H1" could match inside other
tokens). A wrong timeframe_scope on an extracted rule would silently widen
the rule's runtime evaluation scope the moment the user approves it.
"""
import pytest

from backend.app.services.knowledge.extractor import RuleExtractor


def _extract(text: str):
    pages = [{"page": 1, "text": text}]
    creates = RuleExtractor.extract_from_pages("doc_tf_test", pages)
    return creates


def test_m15_mention_does_not_also_match_m1():
    creates = _extract("Entry confirmation on the M15 chart close only.")
    assert creates, "a rule must still be extracted"
    for rule in creates:
        assert "M15" in rule.timeframe_scope, rule.timeframe_scope
        assert "M1" not in rule.timeframe_scope, (
            f"substring bug: M15 text must not grant M1 scope (got {rule.timeframe_scope})")


def test_explicit_m1_mention_still_matches_m1():
    creates = _extract("Scalp entries are validated on the M1 candle close.")
    assert creates
    scopes = [tuple(r.timeframe_scope) for r in creates]
    assert any("M1" in s for s in scopes)


def test_m1_and_m15_mentioned_together_match_both():
    creates = _extract("Use M1 for entry trigger and M15 for trend filter.")
    assert creates
    scopes = [tuple(r.timeframe_scope) for r in creates]
    combined = set().union(*scopes) if scopes else set()
    assert {"M1", "M15"} <= combined
