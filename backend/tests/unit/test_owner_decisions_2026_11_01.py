"""Owner decisions of 2026-11-01, pinned as executable contracts.

These tests prevent an accidental revert of explicit trading-policy and
retention decisions the owner made on 2026-11-01:

1. Daily loss cap: 5.0% (resolves the historical 3.0-vs-5.0 discrepancy).
2. Retention: 30 days for worker_cycle_records, audit_logs, trade_signals
   and demo_orders (a monthly summary is archived before pruning).
3. METRIC_SIGNING_KEY: the .env.example must contain ONLY the placeholder —
   the owner generates and stores the real key privately; no agent ever
   writes a real key into the repo.
"""
import re
from pathlib import Path

from backend.app.core.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_daily_loss_cap_is_owner_decided_five_percent():
    assert settings.MAX_DAILY_LOSS_PERCENT == 5.0, (
        "owner decision 2026-11-01: daily loss cap is 5.0% — reverting to 3.0% "
        "requires a new explicit owner decision")


def test_retention_defaults_are_thirty_days():
    for key in (
        "WORKER_CYCLE_RETENTION_DAYS",
        "AUDIT_LOG_RETENTION_DAYS",
        "TRADE_SIGNAL_RETENTION_DAYS",
        "DEMO_ORDER_RETENTION_DAYS",
    ):
        assert getattr(settings, key) == 30, (
            f"owner decision 2026-11-01: {key} defaults to 30 days")


def test_signing_key_placeholder_never_carries_a_real_value():
    env = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    # The only METRIC_SIGNING_KEY occurrence must be the commented placeholder.
    occurrences = [line for line in env.splitlines() if "METRIC_SIGNING_KEY=" in line]
    assert len(occurrences) == 1
    assert occurrences[0].lstrip().startswith("#"), (
        "the signing key line must stay a comment")
    value = occurrences[0].split("=", 1)[1].strip()
    assert "run:" in value, "placeholder must instruct the owner to generate the key"
    assert not re.fullmatch(r"[A-Za-z0-9_\-]{30,}", value), (
        "a real-looking key must never be committed")


def test_strategy_spec_documents_the_decided_cap():
    spec = (PROJECT_ROOT / "docs" / "strategy-specification.md").read_text(encoding="utf-8")
    assert "۵.۰٪" in spec
    assert "تصمیم قطعی مالک در ۲۰۲۶-۱۱-۰۱" in spec, (
        "the spec must record the owner decision and its date")
