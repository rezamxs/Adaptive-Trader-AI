"""Step 3.3: the system SUGGESTS a direction, the human batch-review decides.

The extractor's suggestion lives in suggested_direction and is informational
only — the generator votes on the human-decided `direction` field exclusively
(M4 preserved). The batch review endpoint approves/rejects many rules in one
human action, optionally adopting the suggestion or overriding it.
"""
import pandas as pd
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.app.models  # noqa: F401 - register all ORM tables on Base
from alembic import command
from alembic.config import Config
from backend.app.core.config import settings
from backend.app.db.base import Base
from backend.app.main import app
from backend.app.models.rule import TradingRule
from backend.app.services.knowledge.extractor import RuleExtractor
from backend.app.services.knowledge.manager import KnowledgeManager
from backend.app.services.signal.generator import SignalGenerator

V1 = "/api/v1"


# ---------------------------------------------------------------------------
# Extractor suggestions (from document TEXT, never the rule name)
# ---------------------------------------------------------------------------

def test_suggestion_buy_from_document_text():
    rule = RuleExtractor._parse_block(
        "Buy when RSI drops below 30 and price is above the daily support.",
        "doc_1", [12],
    )
    assert rule.suggested_direction == "BUY"


def test_suggestion_sell_from_persian_text():
    rule = RuleExtractor._parse_block(
        "فروش زمانی که قیمت به مقاومت رسید و RSI بالای 70 باشد.",
        "doc_1", [13],
    )
    assert rule.suggested_direction == "SELL"


def test_suggestion_empty_when_both_or_neither_sides_mentioned():
    both = RuleExtractor._parse_block(
        "Buy on breakout; sell on breakdown of the range.", "doc_1", [14],
    )
    assert both.suggested_direction is None, "both sides mentioned -> no suggestion"
    neither = RuleExtractor._parse_block(
        "Trail the stop below the 20-period EMA once price moves 1 ATR.",
        "doc_1", [15],
    )
    assert neither.suggested_direction is None


def test_suggestion_is_not_executable_direction():
    """A suggested direction must never vote: the generator reads `direction`."""
    rule = TradingRule(
        rule_id="rule_sug", name="setup", category="entry",
        direction=None, suggested_direction="BUY",
        description="d",
        conditions=[{"indicator": "RSI", "operator": "<", "threshold": 30}],
        parameters={}, symbol_scope=[], timeframe_scope=[],
        invalidation_conditions=[], source_pages=[], assumptions=[],
        ambiguities=[], approval_status="approved",
    )
    direction, _, triggered = SignalGenerator.evaluate_rules(
        pd.Series({"rsi_14": 25.0}), [rule], symbol="XAUUSD", timeframe="M1"
    )
    assert direction == "HOLD", "a suggestion alone must stay fail-closed"
    assert triggered == ["rule_sug"]


# ---------------------------------------------------------------------------
# Batch review endpoint
# ---------------------------------------------------------------------------

def _seed_pending(session, rule_id, suggested=None):
    session.add(TradingRule(
        rule_id=rule_id, name="pending rule", category="entry",
        direction=None, suggested_direction=suggested,
        description="d",
        conditions=[{"indicator": "RSI", "operator": "<", "threshold": 70.0}],
        parameters={}, symbol_scope=["XAUUSD"], timeframe_scope=["M15"],
        invalidation_conditions=[], source_pages=[], assumptions=[],
        ambiguities=[], approval_status="pending",
    ))


def _client():
    return TestClient(app, raise_server_exceptions=False)


def _headers():
    return {"X-API-TOKEN": settings.API_TOKEN}


def test_batch_review_adopts_suggestion_explicit_and_reject(isolated_app_db):
    factory = isolated_app_db
    with factory() as session:
        _seed_pending(session, "r_sug", suggested="SELL")
        _seed_pending(session, "r_explicit", suggested="BUY")
        _seed_pending(session, "r_reject", suggested="BUY")
        _seed_pending(session, "r_nodir")
        session.commit()

    resp = _client().post(f"{V1}/rules/batch-review", json={"items": [
        {"rule_id": "r_sug", "action": "approve", "direction": "SUGGESTED"},
        {"rule_id": "r_explicit", "action": "approve", "direction": "BUY"},
        {"rule_id": "r_reject", "action": "reject", "reason": "conflicts with rule A"},
        {"rule_id": "r_nodir", "action": "approve"},  # approved directionless: never votes
    ]}, headers=_headers())
    assert resp.status_code == 200
    results = {r["rule_id"]: r for r in resp.json()["results"]}

    assert results["r_sug"]["ok"] and results["r_sug"]["direction"] == "SELL"
    assert results["r_explicit"]["ok"] and results["r_explicit"]["direction"] == "BUY"
    assert results["r_reject"]["ok"] and results["r_reject"]["rejected"]
    assert results["r_nodir"]["ok"] and results["r_nodir"]["direction"] is None

    with factory() as session:
        rows = {r.rule_id: r for r in session.query(TradingRule).all()}
    assert rows["r_sug"].approval_status == "approved" and rows["r_sug"].direction == "SELL"
    assert rows["r_explicit"].approval_status == "approved"
    assert rows["r_reject"].approval_status == "rejected"
    assert rows["r_reject"].rejection_reason == "conflicts with rule A"
    assert rows["r_nodir"].approval_status == "approved" and rows["r_nodir"].direction is None


def test_batch_review_suggested_without_suggestion_is_per_item_error(isolated_app_db):
    factory = isolated_app_db
    with factory() as session:
        _seed_pending(session, "r_nosug", suggested=None)
        session.commit()
    resp = _client().post(f"{V1}/rules/batch-review", json={"items": [
        {"rule_id": "r_nosug", "action": "approve", "direction": "SUGGESTED"},
    ]}, headers=_headers())
    results = resp.json()["results"]
    assert results[0]["ok"] is False
    assert "suggested_direction" in results[0]["error"]


def test_batch_review_unknown_rule_and_bad_action_never_abort_batch(isolated_app_db):
    factory = isolated_app_db
    with factory() as session:
        _seed_pending(session, "r_ok", suggested=None)
        session.commit()
    resp = _client().post(f"{V1}/rules/batch-review", json={"items": [
        {"rule_id": "nope", "action": "approve", "direction": "BUY"},
        {"rule_id": "r_ok", "action": "frobnicate"},
        {"rule_id": "r_ok", "action": "reject"},  # reject without reason
        {"rule_id": "r_ok", "action": "approve", "direction": "SELL"},
    ]}, headers=_headers())
    results = resp.json()["results"]
    assert [r["ok"] for r in results] == [False, False, False, True]
    assert results[3]["direction"] == "SELL"


def test_batch_review_reject_without_reason_is_item_error(isolated_app_db):
    factory = isolated_app_db
    with factory() as session:
        _seed_pending(session, "r_x", suggested=None)
        session.commit()
    resp = _client().post(f"{V1}/rules/batch-review", json={"items": [
        {"rule_id": "r_x", "action": "reject", "reason": "   "},
    ]}, headers=_headers())
    assert resp.json()["results"][0]["ok"] is False


def test_batch_review_is_token_gated(isolated_app_db):
    factory = isolated_app_db
    resp = _client().post(f"{V1}/rules/batch-review", json={"items": [
        {"rule_id": "r", "action": "approve"},
    ]})
    assert resp.status_code in (401, 403)


def test_rules_list_exposes_suggested_direction(isolated_app_db):
    factory = isolated_app_db
    with factory() as session:
        _seed_pending(session, "r_vis", suggested="BUY")
        session.commit()
    resp = _client().get(f"{V1}/rules/", headers=_headers())
    rule = next(r for r in resp.json() if r["rule_id"] == "r_vis")
    assert rule["suggested_direction"] == "BUY"
    assert rule["direction"] is None


# ---------------------------------------------------------------------------
# Migration: suggested_direction column, inspector-guarded, reversible
# ---------------------------------------------------------------------------

@pytest.fixture
def alembic_cfg():
    cfg = Config(str(PROJECT_ROOT_FOR_MIGRATIONS / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT_FOR_MIGRATIONS / "migrations"))
    return cfg


PROJECT_ROOT_FOR_MIGRATIONS = Path(__file__).resolve().parents[3]


def test_sdir_migration_adds_and_reverses(tmp_path, alembic_cfg, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'sdir.db').as_posix()}"
    monkeypatch.setenv("ALEMBIC_DATABASE_URL", url)

    command.upgrade(alembic_cfg, "20261101_rdir")
    command.upgrade(alembic_cfg, "head")
    engine = create_engine(url)
    inspector = inspect(engine)
    columns = {c["name"] for c in inspector.get_columns("trading_rules")}
    assert "suggested_direction" in columns
    engine.dispose()

    command.downgrade(alembic_cfg, "20261101_rdir")
    engine = create_engine(url)
    inspector = inspect(engine)
    columns = {c["name"] for c in inspector.get_columns("trading_rules")}
    assert "suggested_direction" not in columns
    engine.dispose()
