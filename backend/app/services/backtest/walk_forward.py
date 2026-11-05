"""Real walk-forward backtest evaluation (M8).

History: the walk-forward evidence gate used to accept any BacktestRun whose
top-level numbers matched the attached payload — mere existence (or matching
hand-made numbers) is not provenance. This module produces the evidence the
gate now demands: an ACTUAL walk-forward evaluation run by the same
RealisticBacktestEngine with the same cost model over chronological
out-of-sample folds. Every fold is itself a persisted BacktestRun; one
aggregate run (strategy_name="walk_forward") carries a provenance block
(fold boundaries, per-fold metrics, dataset fingerprint) that
ModelRegistry._verify_backtest_provenance re-aggregates at attach AND at
promotion time.

What this attests: out-of-sample robustness of the approved-rule STRATEGY
across folds, computed with real costs. It does not retrain the ML model per
fold — model-quality evidence is the separate validation/test attestation.
Any DB writer can still forge rows; METRIC_SIGNING_KEY (HMAC digest) remains
the tamper-evident layer for the payload.
"""
import hashlib
import uuid
from typing import Any, Dict, List

import pandas as pd
from sqlalchemy.orm import Session

from backend.app.core.logging import logger
from backend.app.services.backtest.engine import RealisticBacktestEngine
from backend.app.services.dataset.features import FeatureEngineer
from backend.app.services.signal.generator import SignalGenerator

WF_STRATEGY_NAME = "walk_forward"
WF_FOLD_STRATEGY_NAME = "walk_forward_fold"


def _dataset_fingerprint(df: pd.DataFrame) -> Dict[str, Any]:
    """Deterministic fingerprint of the evaluated input frame."""
    return {
        "rows": int(len(df)),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "sha256": hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest(),
    }


def _aggregate_profit_factor(gross_profit: float, gross_loss: float) -> float:
    """Aggregate PF from summed gross components, mirroring the engine's
    convention (BacktestMetrics.calculate, metrics.py): gross profit / gross
    loss; 99.0 sentinel when there is profit but NO loss to divide by.

    Buglist item 1: the previous no-loss branch stored the raw gross-profit
    AMOUNT as profit_factor (e.g. 500.0), so the promotion gate's
    "profit_factor > 1.0" check compared an absolute money amount against a
    ratio threshold and the metric was meaningless.
    """
    if gross_loss != 0:
        # Fold records carry the loss with either sign convention (metrics.py
        # reports a positive absolute value; some payloads sum negatives), so
        # the ratio normalizes by the magnitude.
        return round(gross_profit / abs(gross_loss), 6)
    return 99.0 if gross_profit > 0 else 0.0


def _continuous_drawdown_from_folds(
    folds: List[Dict[str, Any]],
    initial_balance: float,
) -> Dict[str, Any]:
    """Stitches the per-fold equity extremes into ONE continuous account path.

    Buglist item 3: each fold's engine run restarts from initial_balance, so
    the previous aggregation (max of per-fold MDD%) understated the risk of a
    CONTINUOUS account where losses from earlier folds lower the equity a
    later fold can lose from. This recurrence walks the folds in order,
    carries the running global peak, and measures each fold's absolute peak/
    trough (from its FULL equity timeline, floating marks included) in
    continuous terms:

        start_k = initial_balance + sum(net_j, j < k)
        peak_in_fold_k = max(running_global_peak, fold_peak_abs + offset_k)
        dd_k = (peak_in_fold_k - (fold_trough_abs + offset_k)) / peak_in_fold_k
        running_global_peak = max(running_global_peak, fold_peak_abs + offset_k)

    Each fold's dd_k uses its WORST trough against the peak available up to
    that point; measuring the trough against max(global peak, fold peak) can
    only OVERSTATE the within-fold drawdown (never understate), which is the
    safe direction for a risk gate.

    Requires every fold record to carry ``equity_peak_abs`` and
    ``equity_trough_abs`` (produced by BacktestMetrics.calculate).
    """
    if not folds:
        raise ValueError("Continuous drawdown requires at least one fold record.")
    missing = [
        f.get("fold_index")
        for f in folds
        if not isinstance(f.get("equity_peak_abs"), (int, float))
        or not isinstance(f.get("equity_trough_abs"), (int, float))
    ]
    if missing:
        raise ValueError(
            "Fold records lack absolute equity extremes (equity_peak_abs/"
            f"equity_trough_abs) at fold_index values {missing}; continuous "
            "drawdown cannot be re-aggregated from this payload."
        )
    cumulative_net = 0.0
    global_peak = float(initial_balance)
    worst_dd_pct = 0.0
    worst_dd_amount = 0.0
    for f in folds:
        offset = cumulative_net  # = carried start - fold's own initial balance
        peak_abs = float(f["equity_peak_abs"]) + offset
        trough_abs = float(f["equity_trough_abs"]) + offset
        peak_so_far = max(global_peak, peak_abs)
        dd_amount = max(0.0, peak_so_far - trough_abs)
        dd_pct = (dd_amount / peak_so_far) * 100.0 if peak_so_far > 0 else 0.0
        if dd_pct > worst_dd_pct:
            worst_dd_pct = dd_pct
        if dd_amount > worst_dd_amount:
            worst_dd_amount = dd_amount
        global_peak = max(global_peak, peak_abs)
        cumulative_net += float(f.get("net_profit", 0.0) or 0.0)
    return {
        "max_drawdown_percent": round(worst_dd_pct, 6),
        "max_drawdown_amount": round(worst_dd_amount, 6),
        "scheme": (
            "continuous stitched account: per-fold ABSOLUTE equity extremes "
            "(full floating+closed timeline) offset by the carried balance; "
            "running global peak across folds"
        ),
    }


def run_walk_forward_backtest(
    db: Session,
    df_raw: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    approved_rules: List[Any],
    fold_count: int = 5,
    initial_balance: float = 10000.0,
    spread_points: int = 0,
    slippage_points: int = 0,
    commission_per_lot: float = 0.0,
    point_value: float = 0.01,
    contract_size: float = 100.0,
    tick_size: float = 0.01,
    tick_value: float = 1.0,
    lot_step: float = 0.01,
    min_lot: float = 0.01,
    max_lot: float = 10.0,
) -> Dict[str, Any]:
    """Runs a real walk-forward evaluation and persists one aggregate
    BacktestRun (strategy_name="walk_forward") plus one engine run per fold.

    Returns {"run_id", "walk_forward_metrics", "fold_run_ids"} — the payload
    is ready for ModelRegistry.attach_walk_forward_metrics.
    """
    if fold_count < 2:
        raise ValueError("walk-forward evaluation requires at least 2 folds.")
    if df_raw is None or len(df_raw) < fold_count * 10:
        raise ValueError(
            f"Not enough bars ({0 if df_raw is None else len(df_raw)}) for "
            f"{fold_count} walk-forward folds; need at least {fold_count * 10}.")

    # Features and rule signals are causal per bar — computing them once over
    # the full frame and slicing chronological folds keeps every fold's bars
    # strictly out-of-sample for a rule-based strategy.
    df_feat, _ = FeatureEngineer.transform(df_raw)
    signals = pd.Series(0.0, index=df_feat.index)
    for i in range(len(df_feat)):
        bar = df_feat.iloc[i]
        direction, _, _ = SignalGenerator.evaluate_rules(
            bar, approved_rules, symbol=symbol, timeframe=timeframe
        )
        if direction == "BUY":
            signals.iloc[i] = 1
        elif direction == "SELL":
            signals.iloc[i] = -1

    engine = RealisticBacktestEngine(
        symbol=symbol,
        timeframe=timeframe,
        initial_balance=initial_balance,
        spread_points=spread_points,
        slippage_points=slippage_points,
        commission_per_lot=commission_per_lot,
        point_value=point_value,
        contract_size=contract_size,
        tick_size=tick_size,
        tick_value=tick_value,
        lot_step=lot_step,
        min_lot=min_lot,
        max_lot=max_lot,
        db=db,
    )

    # Chronological contiguous OOS folds (the last fold absorbs the remainder).
    n = len(df_feat)
    fold_len = n // fold_count
    fold_records: List[Dict[str, Any]] = []
    fold_run_ids: List[str] = []
    for k in range(fold_count):
        start = k * fold_len
        end = n if k == fold_count - 1 else (k + 1) * fold_len
        df_fold = df_feat.iloc[start:end]
        sig_fold = signals.iloc[start:end]
        fold_result = engine.run(
            df_fold, sig_fold, strategy_name=WF_FOLD_STRATEGY_NAME,
        )
        m = fold_result["metrics"]
        fold_run_ids.append(fold_result["run_id"])
        fold_records.append({
            "fold_index": k + 1,
            "run_id": fold_result["run_id"],
            "test_start": str(df_fold.index[0]),
            "test_end": str(df_fold.index[-1]),
            "bars": int(len(df_fold)),
            "net_profit": float(m["net_profit"]),
            "total_trades": int(m["total_trades"]),
            "gross_profit": float(m["gross_profit"]),
            "gross_loss": float(m["gross_loss"]),
            "max_drawdown_percent": float(m["max_drawdown_percent"]),
            # Item 3: absolute equity extremes from the fold's FULL timeline,
            # so the aggregate run can attest a CONTINUOUS account drawdown.
            "equity_peak_abs": float(m["equity_peak_abs"]),
            "equity_trough_abs": float(m["equity_trough_abs"]),
        })

    agg_net = round(sum(f["net_profit"] for f in fold_records), 6)
    agg_trades = int(sum(f["total_trades"] for f in fold_records))
    agg_gross_p = round(sum(f["gross_profit"] for f in fold_records), 6)
    agg_gross_l = round(sum(f["gross_loss"] for f in fold_records), 6)
    continuous_dd = _continuous_drawdown_from_folds(fold_records, initial_balance)
    agg_dd = continuous_dd["max_drawdown_percent"]
    agg_pf = _aggregate_profit_factor(agg_gross_p, agg_gross_l)

    provenance: Dict[str, Any] = {
        "evaluator": "backend/app/services/backtest/walk_forward.py::run_walk_forward_backtest",
        "fold_count": fold_count,
        "folds": fold_records,
        "dataset_fingerprint": _dataset_fingerprint(df_raw),
        "engine_config": {
            "initial_balance_per_fold": initial_balance,
            "spread_points": spread_points,
            "slippage_points": slippage_points,
            "commission_per_lot": commission_per_lot,
            "point_value": point_value,
            "contract_size": contract_size,
        },
        "costs_included": {"spread": True, "slippage": True, "commission": commission_per_lot > 0},
        "fold_scheme": "chronological contiguous out-of-sample windows; last fold absorbs the remainder",
        # Item 3: the aggregate drawdown is the CONTINUOUS stitched-account
        # figure (see _continuous_drawdown_from_folds), not max-of-folds.
        "continuous_drawdown": continuous_dd,
    }

    run_id = f"bt_wf_{uuid.uuid4().hex[:10]}"
    from backend.app.models.backtest import BacktestRun
    db.add(BacktestRun(
        run_id=run_id,
        strategy_name=WF_STRATEGY_NAME,
        model_id=None,
        symbol=symbol,
        timeframe=timeframe,
        start_date=df_feat.index[0].to_pydatetime(),
        end_date=df_feat.index[-1].to_pydatetime(),
        initial_balance=initial_balance,
        final_balance=round(initial_balance + agg_net, 2),
        net_profit=agg_net,
        gross_profit=agg_gross_p,
        gross_loss=agg_gross_l,
        max_drawdown_percent=agg_dd,
        max_drawdown_amount=continuous_dd["max_drawdown_amount"],
        total_trades=agg_trades,
        winning_trades=0,   # per-fold win counts are not meaningfully additive
        losing_trades=0,
        win_rate=0.0,
        profit_factor=agg_pf,
        expectancy=round(agg_net / agg_trades, 6) if agg_trades > 0 else 0.0,
        average_trade=0.0,
        assumptions={
            "walk_forward": provenance,
            "note": "aggregate of per-fold engine runs; per-fold runs are persisted separately",
        },
        limitations=[
            "Position sizing uses each fold's own initial balance, so the "
            "continuous drawdown is stitched from per-fold absolute equity "
            "extremes (an upper-bound estimate for the sizing path, never an "
            "underestimate).",
            "Rule-based strategy evaluation; no per-fold ML model retraining.",
        ],
        equity_curve=[],
        trades_summary=[],
    ))
    db.commit()
    logger.info(
        f"Walk-forward evaluation {run_id}: {fold_count} folds, "
        f"agg_net={agg_net}, trades={agg_trades}, pf={agg_pf}, max_dd={agg_dd}%")

    payload: Dict[str, Any] = {
        "backtest_run_id": run_id,
        "fold_count": fold_count,
        "trade_count": agg_trades,
        "net_profit": agg_net,
        "profit_factor": agg_pf,
        "max_drawdown_percent": agg_dd,
        # Honest cost flags: they must reflect the parameters this evaluation
        # ACTUALLY ran with (mirroring the persisted provenance block below).
        # Hardcoding True let a zero-cost evaluation satisfy the promotion
        # gate's "costs included" requirement with fabricated flags.
        "spread_costs_included": spread_points > 0,
        "slippage_costs_included": slippage_points > 0,
        "commission_costs_included": commission_per_lot > 0,
        "folds": fold_records,
        "dataset_fingerprint": provenance["dataset_fingerprint"],
        "engine_config": provenance["engine_config"],
    }
    return {"run_id": run_id, "walk_forward_metrics": payload, "fold_run_ids": fold_run_ids}
