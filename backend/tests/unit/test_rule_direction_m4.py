"""M4 regression tests: rule direction is explicit, never name-derived.

The generator and the conflict detector must use the explicit
``TradingRule.direction`` field ('BUY' | 'SELL' | NULL) and NEVER inspect the
rule name for direction keywords. A rule without an explicit direction
triggers but casts NO direction vote (fail-closed). The direction migration
must not backfill from names either.
"""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.app.models  # noqa: F401 - register all ORM tables on Base
from backend.app.core.config import settings
from backend.app.db.base import Base
from backend.app.main import app
from backend.app.models.rule import TradingRule
from backend.app.services.knowledge.manager import KnowledgeManager
from backend.app.services.signal.generator import SignalGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[3]

V1 = "/api/v1"


def _rule(name, direction, *, rule_id="r1"):
    return SimpleNamespace(
        rule_id=rule_id,
        name=name,
        direction=direction,
        category="entry",
        conditions=[{"indicator": "RSI", "operator": "<", "threshold": 30}],
        symbol_scope=[],
        timeframe_scope=[],
    )


# ---------------------------------------------------------------------------
# Generator: direction comes only from the explicit field
# ---------------------------------------------------------------------------

def test_misleading_name_uses_explicit_direction():
    """A rule NAMED 'buy breakout' with explicit direction=SELL must vote SELL."""
    rule = _rule("buy breakout setup", "SELL")
    direction, strength, triggered = SignalGenerator.evaluate_rules(
        pd.Series({"rsi_14": 25.0}), [rule], symbol="XAUUSD", timeframe="M1"
    )
    assert direction == "SELL"
    assert strength == 1.0
    assert triggered == ["r1"]


def test_misleading_name_uses_explicit_direction_buy():
    """A rule NAMED 'sell the top' with explicit direction=BUY must vote BUY."""
    rule = _rule("sell the top strategy", "BUY")
    direction, _, triggered = SignalGenerator.evaluate_rules(
        pd.Series({"rsi_14": 25.0}), [rule], symbol="XAUUSD", timeframe="M1"
    )
    assert direction == "BUY"
    assert triggered == ["r1"]


def test_rule_without_direction_triggers_but_never_votes():
    """direction=NULL (undecided): the rule triggers but casts no vote — fail-closed."""
    rule = _rule("Any neutral name", None)
    direction, strength, triggered = SignalGenerator.evaluate_rules(
        pd.Series({"rsi_14": 25.0}), [rule], symbol="XAUUSD", timeframe="M1"
    )
    assert direction == "HOLD"
    assert strength == 0.0
    assert triggered == ["r1"]  # triggered is recorded, but no direction vote happened


def test_unknown_direction_value_fails_closed():
    """A legacy/garbage direction value ('LONG') must never become a vote."""
    rule = _rule("x", "LONG")
    direction, _, triggered = SignalGenerator.evaluate_rules(
        pd.Series({"rsi_14": 25.0}), [rule], symbol="XAUUSD", timeframe="M1"
    )
    assert direction == "HOLD"
    assert triggered == ["r1"]


def test_directional_name_without_direction_field_does_not_vote():
    """Even a name that literally says BUY casts no vote when direction is NULL."""
    rule = _rule("BUY everything now خرید", None)
    direction, _, triggered = SignalGenerator.evaluate_rules(
        pd.Series({"rsi_14": 25.0}), [rule], symbol="XAUUSD", timeframe="M1"
    )
    assert direction == "HOLD"
    assert triggered == ["r1"]


def test_opposite_directions_resolve_by_majority():
    buy = _rule("setup alpha", "BUY", rule_id="r_buy")
    sell = _rule("setup beta", "SELL", rule_id="r_sell_1")
    sell2 = _rule("setup gamma", "SELL", rule_id="r_sell_2")
    direction, _, triggered = SignalGenerator.evaluate_rules(
        pd.Series({"rsi_14": 25.0}), [buy, sell, sell2], symbol="XAUUSD", timeframe="M1"
    )
    assert direction == "SELL"
    assert triggered == ["r_buy", "r_sell_1", "r_sell_2"]


# ---------------------------------------------------------------------------
# Conflict detection: explicit direction only, names are irrelevant
# ---------------------------------------------------------------------------

def _persist_rule(session, rule_id, name, direction):
    session.add(TradingRule(
        rule_id=rule_id, name=name, category="entry", direction=direction,
        description="d", conditions=[{"indicator": "RSI", "operator": "<", "threshold": 70.0}],
        parameters={}, symbol_scope=["XAUUSD"], timeframe_scope=["M15"],
        invalidation_conditions=[], source_pages=[], assumptions=[],
        ambiguities=[], approval_status="approved",
    ))


def _session_with_rules(*specs):
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    for rule_id, name, direction in specs:
        _persist_rule(session, rule_id, name, direction)
    session.commit()
    return session


def test_conflict_detected_from_explicit_directions():
    with _session_with_rules(
        ("r1", "setup one", "BUY"),
        ("r2", "setup two", "SELL"),
    ) as session:
        conflicts = KnowledgeManager(session).check_rule_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0]["conflict_type"] == "CONTRADICTORY_DIRECTION"


def test_directional_names_without_direction_field_do_not_conflict():
    """Names saying buy/sell mean nothing: NULL direction on both sides → no conflict."""
    with _session_with_rules(
        ("r1", "Buy on signal خرید", None),
        ("r2", "Sell on signal فروش", None),
    ) as session:
        assert KnowledgeManager(session).check_rule_conflicts() == []


# ---------------------------------------------------------------------------
# API: the only conscious, reviewable direction assignment path
# ---------------------------------------------------------------------------

def _seed_rule_via_api(factory):
    with factory() as session:
        _persist_rule(session, "rule_m4_api", "Neutral legacy rule", None)
        session.commit()


def test_direction_endpoint_sets_and_clears_direction(isolated_app_db):
    factory = isolated_app_db
    _seed_rule_via_api(factory)
    client = TestClient(app, raise_server_exceptions=False)
    headers = {"X-API-TOKEN": settings.API_TOKEN}

    resp = client.post(
        f"{V1}/rules/rule_m4_api/direction",
        json={"direction": "SELL", "actor": "OPERATOR", "notes": "reviewed page 12"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["direction"] == "SELL"

    resp = client.post(
        f"{V1}/rules/rule_m4_api/direction",
        json={"direction": None, "notes": "clearing back to undecided"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["direction"] is None


def test_direction_endpoint_rejects_invalid_value_and_unknown_rule(isolated_app_db):
    factory = isolated_app_db
    _seed_rule_via_api(factory)
    client = TestClient(app, raise_server_exceptions=False)
    headers = {"X-API-TOKEN": settings.API_TOKEN}

    # 'LONG' is not a legal direction: pydantic validation must reject it.
    resp = client.post(
        f"{V1}/rules/rule_m4_api/direction",
        json={"direction": "LONG"},
        headers=headers,
    )
    assert resp.status_code == 422

    resp = client.post(
        f"{V1}/rules/no_such_rule/direction",
        json={"direction": "BUY"},
        headers=headers,
    )
    assert resp.status_code == 404

    # And the endpoint is token-gated like every other state-changing route.
    resp = client.post(
        f"{V1}/rules/rule_m4_api/direction",
        json={"direction": "BUY"},
    )
    assert resp.status_code in (401, 403)


def test_rules_list_response_exposes_direction(isolated_app_db):
    factory = isolated_app_db
    _seed_rule_via_api(factory)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get(
        f"{V1}/rules/", headers={"X-API-TOKEN": settings.API_TOKEN}
    )
    assert resp.status_code == 200
    rule = next(r for r in resp.json() if r["rule_id"] == "rule_m4_api")
    assert rule["direction"] is None


# ---------------------------------------------------------------------------
# Migration: adds the column, never backfills from names, reversible
# ---------------------------------------------------------------------------

@pytest.fixture
def alembic_cfg():
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    return cfg


def test_direction_migration_adds_nullable_column_without_name_backfill(tmp_path, alembic_cfg, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'rdir.db').as_posix()}"
    monkeypatch.setenv("ALEMBIC_DATABASE_URL", url)

    # Build the chain up to the revision BEFORE direction, then plant legacy
    # rules exactly as they exist today (no direction column yet).
    command.upgrade(alembic_cfg, "20261031_auidx")
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO trading_rules (rule_id, name, category, description, conditions, "
            "parameters, symbol_scope, timeframe_scope, invalidation_conditions, source_pages, "
            "assumptions, ambiguities, approval_status, version, created_at, updated_at) "
            "VALUES ('legacy_buy_name', 'Buy on RSI Oversold', 'entry', 'd', '[]', '{}', '[]', "
            "'[]', '[]', '[]', '[]', '[]', 'pending', 1, '2026-10-30 00:00:00', '2026-10-30 00:00:00')"))
    engine.dispose()

    command.upgrade(alembic_cfg, "head")

    engine = create_engine(url)
    try:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("trading_rules")}
        assert "direction" in columns
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT direction FROM trading_rules WHERE rule_id = 'legacy_buy_name'")
            ).fetchone()
        # The migration must NOT guess from the name: a rule named 'Buy...'
        # still has direction NULL until an operator assigns it consciously.
        assert row[0] is None
    finally:
        engine.dispose()


def test_direction_migration_is_reversible(tmp_path, alembic_cfg, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'rdir_rev.db').as_posix()}"
    monkeypatch.setenv("ALEMBIC_DATABASE_URL", url)

    command.upgrade(alembic_cfg, "head")
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO trading_rules (rule_id, name, category, direction, description, conditions, "
            "parameters, symbol_scope, timeframe_scope, invalidation_conditions, source_pages, "
            "assumptions, ambiguities, approval_status, version, created_at, updated_at) "
            "VALUES ('r1', 'n', 'entry', 'SELL', 'd', '[]', '{}', '[]', '[]', '[]', '[]', "
            "'[]', '[]', 'pending', 1, '2026-10-30 00:00:00', '2026-10-30 00:00:00')"))
    engine.dispose()

    command.downgrade(alembic_cfg, "20261031_auidx")
    engine = create_engine(url)
    try:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("trading_rules")}
        assert "direction" not in columns
    finally:
        engine.dispose()
