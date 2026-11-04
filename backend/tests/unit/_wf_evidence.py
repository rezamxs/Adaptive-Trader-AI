"""Shared walk-forward evidence fixture builder (M8 round 2).

Governance tests need a wf run + payload pair that is internally consistent
the way backend/app/services/backtest/walk_forward.py produces them: the
persisted run's top-level fields re-aggregate from the fold records, the
payload mirrors the persisted provenance exactly, and the run is labelled
strategy_name="walk_forward". This simulates the evaluator's OUTPUT for tests
whose subject is a different gate (digest, status, promotion flow) — it does
NOT replace the real evaluator tests in test_walk_forward_provenance.py.
"""
from datetime import datetime
from typing import Any, Dict

from backend.app.models.backtest import BacktestRun

# Numbers shared by the governance fixtures (fold_count divides trade_count
# and net_profit exactly so the aggregates match the historical claims).
WF_NUMBERS = {
    "net_profit": 150.0,
    "trade_count": 50,
    "profit_factor": 1.25,
    "max_drawdown_percent": 4.0,
    "fold_count": 5,
}


def build_walk_forward_payload(run_id: str = "bt_wf_fixture") -> Dict[str, Any]:
    """Builds a wf payload whose per-fold records re-aggregate to the claimed
    top-level numbers (net = gross_profit + gross_loss; pf = gp/|gl|)."""
    net = WF_NUMBERS["net_profit"]
    pf = WF_NUMBERS["profit_factor"]
    fold_count = WF_NUMBERS["fold_count"]
    gross_loss_total = -round(net / (pf - 1.0), 6)   # |gl| = net / (pf - 1)
    gross_profit_total = round(net + abs(gross_loss_total), 6)
    per_net = round(net / fold_count, 6)
    per_gp = round(gross_profit_total / fold_count, 6)
    per_gl = round(gross_loss_total / fold_count, 6)
    per_trades = WF_NUMBERS["trade_count"] // fold_count
    dd = WF_NUMBERS["max_drawdown_percent"]

    folds = [{
        "fold_index": i + 1,
        "run_id": f"{run_id}_f{i + 1}",
        "test_start": f"2026-01-{i + 1:02d}T00:00:00+00:00",
        "test_end": f"2026-01-{i + 2:02d}T00:00:00+00:00",
        "bars": 100,
        "net_profit": per_net,
        "total_trades": per_trades,
        "gross_profit": per_gp,
        "gross_loss": per_gl,
        "max_drawdown_percent": dd,
    } for i in range(fold_count)]

    agg_net = round(sum(f["net_profit"] for f in folds), 6)
    agg_trades = int(sum(f["total_trades"] for f in folds))
    agg_gp = round(sum(f["gross_profit"] for f in folds), 6)
    agg_gl = round(sum(f["gross_loss"] for f in folds), 6)
    fingerprint = {
        "rows": 1000,
        "start": "2026-01-01 00:00:00+00:00",
        "end": "2026-02-01 00:00:00+00:00",
        "sha256": "f" * 64,
    }
    return {
        "backtest_run_id": run_id,
        "fold_count": fold_count,
        "trade_count": agg_trades,
        "net_profit": agg_net,
        "profit_factor": round(agg_gp / abs(agg_gl), 6),
        "max_drawdown_percent": dd,
        "spread_costs_included": True,
        "slippage_costs_included": True,
        "commission_costs_included": True,
        "folds": folds,
        "dataset_fingerprint": fingerprint,
        "engine_config": {
            "initial_balance_per_fold": 10000.0,
            "spread_points": 20, "slippage_points": 2, "commission_per_lot": 7.0,
            "point_value": 0.01, "contract_size": 100.0,
        },
    }


def seed_walk_forward_run(session, run_id: str = "bt_wf_fixture") -> BacktestRun:
    """Persists the aggregate BacktestRun for `build_walk_forward_payload` —
    strategy_name='walk_forward' with the provenance block in assumptions."""
    payload = build_walk_forward_payload(run_id)
    session.add(BacktestRun(
        run_id=run_id,
        strategy_name="walk_forward",
        model_id=None,
        symbol="XAUUSD",
        timeframe="M15",
        start_date=datetime(2026, 1, 1),
        end_date=datetime(2026, 2, 1),
        initial_balance=10000.0,
        final_balance=round(10000.0 + payload["net_profit"], 2),
        net_profit=payload["net_profit"],
        gross_profit=round(sum(f["gross_profit"] for f in payload["folds"]), 6),
        gross_loss=round(sum(f["gross_loss"] for f in payload["folds"]), 6),
        max_drawdown_percent=payload["max_drawdown_percent"],
        max_drawdown_amount=0.0,
        total_trades=payload["trade_count"],
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        profit_factor=payload["profit_factor"],
        expectancy=0.0,
        average_trade=0.0,
        assumptions={
            "walk_forward": {
                "evaluator": "test fixture simulating walk_forward.py output",
                "fold_count": payload["fold_count"],
                "folds": payload["folds"],
                "dataset_fingerprint": payload["dataset_fingerprint"],
                "engine_config": payload["engine_config"],
                "costs_included": {
                    # Must mirror the real evaluator's honest cost block AND
                    # stay consistent with engine_config above (spread 20,
                    # slippage 2, commission 7.0 — all actually applied).
                    "spread": True, "slippage": True, "commission": True,
                },
                "fold_scheme": "chronological contiguous out-of-sample windows",
            },
        },
        limitations=[],
        equity_curve=[],
        trades_summary=[],
    ))
    session.commit()
    return session.query(BacktestRun).filter_by(run_id=run_id).first()
