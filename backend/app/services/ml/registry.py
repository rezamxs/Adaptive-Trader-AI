"""Model Governance, Versioning, and Controlled Promotion Workflow."""
from pathlib import Path
from typing import Any, Dict, List, Optional
import math
import joblib
from sqlalchemy.orm import Session

from backend.app.core.logging import logger, log_security_event
from backend.app.core.config import settings
from backend.app.core.errors import ModelGovernanceError
from backend.app.core.metrics_integrity import (
    metric_digest,
    verify_metric_digest,
    verify_ml_walk_forward_digest,
)
from backend.app.models.ml_model import MLModelRecord
from backend.app.models.backtest import BacktestRun
from backend.app.models.audit_log import AuditLog
from backend.app.services.ml.walk_forward import (
    MLWF_EVIDENCE_TYPE,
    MLWF_THRESHOLD,
    verify_ml_walk_forward_payload,
)


class ModelRegistry:
    """Oversees model promotions, audits, artifact loading, and lifecycle state."""

    # Keep these gates aligned with the minimum evidence accepted by compare_models.
    MIN_PROMOTION_ROC_AUC = 0.52
    MAX_PROMOTION_BRIER_SCORE = 0.25
    MIN_PROMOTION_EVALUATION_SAMPLES = 30
    MIN_WALK_FORWARD_FOLDS = 3
    MIN_WALK_FORWARD_TRADES = 30
    # M8: promotion-grade per-fold-retrained ML walk-forward evidence.
    MIN_ML_WALK_FORWARD_FOLDS = 3

    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def _metric_value(metrics: Any, key: str) -> Optional[float]:
        """Return a finite probability/score in [0, 1], or None when invalid."""
        if not isinstance(metrics, dict) or key not in metrics:
            return None
        try:
            value = float(metrics[key])
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return None
        return value

    @classmethod
    def _sample_count(cls, metrics: Any) -> int:
        if not isinstance(metrics, dict):
            return 0
        try:
            count = int(metrics.get("sample_count", 0))
        except (TypeError, ValueError, OverflowError):
            return 0
        return count if count >= 0 else 0

    @classmethod
    def _walk_forward_failure(cls, metrics: Any) -> Optional[str]:
        """Require actual cost-aware, out-of-sample backtest evidence before promotion."""
        if not isinstance(metrics, dict) or not metrics:
            return "Walk-forward backtest evidence is missing."

        fold_count = metrics.get("fold_count")
        trade_count = metrics.get("trade_count")
        if (
            not isinstance(fold_count, int) or isinstance(fold_count, bool)
            or not isinstance(trade_count, int) or isinstance(trade_count, bool)
            or fold_count < cls.MIN_WALK_FORWARD_FOLDS
            or trade_count < cls.MIN_WALK_FORWARD_TRADES
        ):
            return (
                "Walk-forward evidence must contain at least "
                f"{cls.MIN_WALK_FORWARD_FOLDS} folds and {cls.MIN_WALK_FORWARD_TRADES} trades."
            )

        try:
            net_profit = float(metrics["net_profit"])
            profit_factor = float(metrics["profit_factor"])
            max_drawdown_percent = float(metrics["max_drawdown_percent"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return "Walk-forward evidence is missing net profit, profit factor, or drawdown."

        if not all(math.isfinite(value) for value in (net_profit, profit_factor, max_drawdown_percent)):
            return "Walk-forward metrics must contain finite numeric values."
        if net_profit <= 0:
            return "Walk-forward net profit must be positive after costs."
        if profit_factor <= 1.0:
            return "Walk-forward profit factor must be greater than 1 after costs."
        if max_drawdown_percent < 0 or max_drawdown_percent > settings.MAX_DRAWDOWN_PERCENT:
            return (
                "Walk-forward maximum drawdown must be between 0 and "
                f"{settings.MAX_DRAWDOWN_PERCENT:.2f}%."
            )

        cost_fields = ("spread_costs_included", "slippage_costs_included", "commission_costs_included")
        if any(metrics.get(field) is not True for field in cost_fields):
            return "Walk-forward backtest must include spread, slippage, and commission costs."
        return None

    def get_active_model_record(self) -> Optional[MLModelRecord]:
        """Returns the currently active MLModelRecord, if any."""
        return self.db.query(MLModelRecord).filter(MLModelRecord.status == "active").first()

    def load_active_model_artifact(self) -> Optional[Dict[str, Any]]:
        """Loads and unpickles the active model artifact from disk."""
        active_rec = self.get_active_model_record()
        if not active_rec:
            return None

        path = Path(active_rec.artifact_path).resolve()
        artifact_root = Path(settings.ARTIFACTS_MODELS_DIR).resolve()
        try:
            path.relative_to(artifact_root)
        except ValueError:
            logger.error("Active model artifact path is outside the configured artifact directory.")
            return None
        if not path.is_file() or path.suffix.lower() != ".joblib":
            logger.error(f"Active model artifact file is missing or has an unexpected extension: {path}")
            return None

        try:
            payload = joblib.load(path)
        except Exception as exc:
            logger.error(f"Active model artifact could not be loaded: {exc}")
            return None
        if (
            not isinstance(payload, dict)
            or not callable(getattr(payload.get("model"), "predict_proba", None))
            or not isinstance(payload.get("feature_names"), (list, tuple))
            or not payload.get("feature_names")
        ):
            logger.error("Active model artifact is missing required inference components.")
            return None
        # H1 at inference time: the artifact's embedded digest must match the
        # attested store digest. A swapped/edited joblib file otherwise executes
        # on load with no integrity check (promote verifies this contract, but
        # load is the runtime path — it must honor it too). A digest-less
        # artifact (legacy H1 rows) fails closed, consistent with promotion.
        artifact_digest = payload.get("metrics_digest")
        if not artifact_digest or artifact_digest != active_rec.metrics_digest:
            logger.error(
                "Active model artifact integrity check failed: artifact digest "
                f"{'missing' if not artifact_digest else 'mismatch'} vs attested store digest."
            )
            return None
        return payload

    def list_models(self, status: Optional[str] = None) -> List[MLModelRecord]:
        """Lists models optionally filtered by status."""
        query = self.db.query(MLModelRecord)
        if status:
            query = query.filter(MLModelRecord.status == status)
        return query.order_by(MLModelRecord.id.desc()).all()

    def compare_models(self, candidate_id: str) -> Dict[str, Any]:
        """Compares candidate metrics against currently active model or baseline."""
        candidate = self.db.query(MLModelRecord).filter_by(model_id=candidate_id).first()
        if not candidate:
            raise ValueError(f"Candidate model {candidate_id} not found.")

        active = self.get_active_model_record()

        comparison = {
            "candidate_id": candidate.model_id,
            "candidate_algorithm": candidate.algorithm,
            "candidate_validation_metrics": candidate.validation_metrics,
            "candidate_test_metrics": candidate.test_metrics,
            "active_id": active.model_id if active else None,
            "active_algorithm": active.algorithm if active else None,
            "active_validation_metrics": active.validation_metrics if active else None,
            "active_test_metrics": active.test_metrics if active else None,
            "recommendation": "UNKNOWN",
        }

        cand_val_auc = self._metric_value(candidate.validation_metrics, "roc_auc")
        cand_val_brier = self._metric_value(candidate.validation_metrics, "brier_score")
        cand_test_auc = self._metric_value(candidate.test_metrics, "roc_auc")
        cand_test_brier = self._metric_value(candidate.test_metrics, "brier_score")
        cand_val_n = self._sample_count(candidate.validation_metrics)
        cand_test_n = self._sample_count(candidate.test_metrics)

        if (
            cand_val_auc is None or cand_val_brier is None
            or cand_test_auc is None or cand_test_brier is None
            or cand_val_n < self.MIN_PROMOTION_EVALUATION_SAMPLES
            or cand_test_n < self.MIN_PROMOTION_EVALUATION_SAMPLES
        ):
            comparison["recommendation"] = (
                "INSUFFICIENT_EVIDENCE: Valid validation/test metrics and at least "
                f"{self.MIN_PROMOTION_EVALUATION_SAMPLES} samples per partition are required."
            )
            return comparison

        absolute_gate_passes = (
            cand_val_auc > self.MIN_PROMOTION_ROC_AUC
            and cand_test_auc > self.MIN_PROMOTION_ROC_AUC
            and cand_val_brier <= self.MAX_PROMOTION_BRIER_SCORE
            and cand_test_brier <= self.MAX_PROMOTION_BRIER_SCORE
        )
        if not absolute_gate_passes:
            comparison["recommendation"] = (
                "INSUFFICIENT_EVIDENCE: Validation and independent test metrics must "
                f"both pass ROC-AUC > {self.MIN_PROMOTION_ROC_AUC:.2f} and Brier <= "
                f"{self.MAX_PROMOTION_BRIER_SCORE:.2f}."
            )
            return comparison

        if active:
            active_val_auc = self._metric_value(active.validation_metrics, "roc_auc")
            active_val_n = self._sample_count(active.validation_metrics)
            if active_val_auc is None or active_val_n < self.MIN_PROMOTION_EVALUATION_SAMPLES:
                comparison["recommendation"] = (
                    "INSUFFICIENT_EVIDENCE: Active model has no valid temporal validation metrics; "
                    "re-evaluate it before replacement."
                )
            elif cand_val_auc > active_val_auc:
                comparison["recommendation"] = (
                    "PROCEED_TO_BACKTEST: Candidate validation AUC is superior and independent test gates pass."
                )
            else:
                comparison["recommendation"] = (
                    "RETAIN_ACTIVE: Candidate does not improve the active model on temporal validation."
                )
        else:
            comparison["recommendation"] = (
                "PROCEED_TO_BACKTEST: Validation and independent test metric gates pass."
            )
        return comparison

    @classmethod
    def _verify_backtest_provenance(cls, walk_forward_metrics: Any, backtest_run: BacktestRun) -> Optional[str]:
        """M8: the run existing is not provenance — the walk-forward numbers
        must BE the numbers the stored BacktestRun row actually produced,
        and the run must BE a real walk-forward evaluation. Returns a failure
        description, or None when the evidence matches."""
        try:
            wf_net = float(walk_forward_metrics["net_profit"])
            wf_pf = float(walk_forward_metrics["profit_factor"])
            wf_dd = float(walk_forward_metrics["max_drawdown_percent"])
            wf_trades_raw = walk_forward_metrics["trade_count"]
        except (KeyError, TypeError, ValueError, OverflowError):
            return "Walk-forward evidence is missing net profit, profit factor, drawdown, or trade count."
        # trade_count must be a real integer: float(...) silent truncation
        # (e.g. 120.7 -> 120) would let non-integer claims through.
        if isinstance(wf_trades_raw, bool) or not isinstance(wf_trades_raw, int):
            return "Walk-forward trade_count must be an integer."
        wf_trades = wf_trades_raw

        for name, claimed, stored in (
            ("net_profit", wf_net, float(backtest_run.net_profit or 0.0)),
            ("profit_factor", wf_pf, float(backtest_run.profit_factor or 0.0)),
            ("max_drawdown_percent", wf_dd, float(backtest_run.max_drawdown_percent or 0.0)),
        ):
            if not math.isclose(claimed, stored, rel_tol=1e-9, abs_tol=1e-9):
                return (
                    f"Walk-forward {name} ({claimed}) does not match the stored "
                    f"backtest run ({stored}); the evidence numbers are not from that run."
                )
        if wf_trades != int(backtest_run.total_trades or 0):
            return (
                f"Walk-forward trade_count ({wf_trades}) does not match the stored "
                f"backtest run total_trades ({backtest_run.total_trades}); "
                "the evidence numbers are not from that run."
            )
        return cls._verify_walk_forward_provenance(walk_forward_metrics, backtest_run)

    @classmethod
    def _verify_walk_forward_provenance(cls, walk_forward_metrics: Any, backtest_run: BacktestRun) -> Optional[str]:
        """M8 round 2: the referenced run must carry the provenance of a REAL
        walk-forward evaluation — strategy marker, per-fold records with time
        boundaries, a dataset fingerprint, and per-fold numbers that re-
        aggregate to BOTH the stored run and the payload. A plain hand-made
        BacktestRun with coincidentally matching top-level numbers fails here."""
        if (backtest_run.strategy_name or "") != "walk_forward":
            return (
                "referenced run is not a walk-forward evaluation "
                f"(strategy_name={backtest_run.strategy_name!r}); a plain "
                "backtest run cannot attest walk-forward evidence"
            )
        assumptions = backtest_run.assumptions or {}
        provenance = assumptions.get("walk_forward") if isinstance(assumptions, dict) else None
        if not isinstance(provenance, dict):
            return "referenced run lacks the 'walk_forward' provenance block in its assumptions"

        folds = provenance.get("folds")
        if not isinstance(folds, list) or not folds:
            return "walk-forward provenance carries no fold records"

        claimed_fold_count = walk_forward_metrics.get("fold_count")
        if claimed_fold_count != len(folds):
            return (
                f"fold_count mismatch: payload claims {claimed_fold_count}, "
                f"run provenance records {len(folds)} folds"
            )
        if len(folds) < cls.MIN_WALK_FORWARD_FOLDS:
            return (
                f"walk-forward provenance must record at least "
                f"{cls.MIN_WALK_FORWARD_FOLDS} folds, got {len(folds)}"
            )
        if walk_forward_metrics.get("folds") != folds:
            return "payload fold records do not match the persisted run provenance"

        # Fold windows must be chronological and each must carry a time range.
        starts = [str(f.get("test_start", "")) for f in folds]
        if any(not s for s in starts) or starts != sorted(starts):
            return "fold test windows are missing or not chronological"

        # Re-aggregate from the PERSISTED fold records and compare against
        # BOTH the stored run fields and the payload claims.
        agg_net = sum(float(f.get("net_profit", 0.0) or 0.0) for f in folds)
        agg_trades = sum(int(f.get("total_trades", 0) or 0) for f in folds)
        agg_gross_p = sum(float(f.get("gross_profit", 0.0) or 0.0) for f in folds)
        agg_gross_l = sum(float(f.get("gross_loss", 0.0) or 0.0) for f in folds)
        agg_dd = max(float(f.get("max_drawdown_percent", 0.0) or 0.0) for f in folds)

        def _close(a: float, b: float) -> bool:
            return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)

        if not _close(agg_net, float(backtest_run.net_profit or 0.0)):
            return (
                f"per-fold net_profit re-aggregates to {agg_net} but the stored run "
                f"records {backtest_run.net_profit}; the fold records are not the "
                "origin of this run"
            )
        if agg_trades != int(backtest_run.total_trades or 0):
            return (
                f"per-fold total_trades re-aggregate to {agg_trades} but the stored run "
                f"records {backtest_run.total_trades}"
            )
        if not _close(agg_gross_p, float(backtest_run.gross_profit or 0.0)) or not _close(
            agg_gross_l, float(backtest_run.gross_loss or 0.0)
        ):
            return "per-fold gross profit/loss do not re-aggregate to the stored run"
        if not _close(agg_dd, float(backtest_run.max_drawdown_percent or 0.0)):
            return (
                f"per-fold max drawdown re-aggregates to {agg_dd} but the stored run "
                f"records {backtest_run.max_drawdown_percent}"
            )
        try:
            payload_net = float(walk_forward_metrics["net_profit"])
            payload_trades = int(walk_forward_metrics["trade_count"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return "payload is missing net_profit or trade_count for re-aggregation"
        if not _close(agg_net, payload_net) or agg_trades != payload_trades:
            return "payload net_profit/trade_count do not match the per-fold re-aggregation"

        # Dataset fingerprint binds the evaluation to its input data.
        fingerprint = provenance.get("dataset_fingerprint")
        if not isinstance(fingerprint, dict) or not fingerprint.get("sha256"):
            return "walk-forward provenance lacks the dataset fingerprint"
        if walk_forward_metrics.get("dataset_fingerprint") != fingerprint:
            return "payload dataset fingerprint does not match the persisted run provenance"

        # M8 round 3: the cost flags must match what the producing run
        # ACTUALLY ran with. The payload's *_costs_included booleans gate
        # promotion, so a fabricated True on a zero-cost run would satisfy
        # the costs gate — the persisted provenance (engine_config values and
        # the evaluator's own costs_included block) is the ground truth.
        costs_block = provenance.get("costs_included")
        engine_cfg = provenance.get("engine_config")
        if not isinstance(costs_block, dict) or not isinstance(engine_cfg, dict):
            return "walk-forward provenance lacks the costs_included/engine_config blocks"
        try:
            spread_points = float(engine_cfg["spread_points"])
            slippage_points = float(engine_cfg["slippage_points"])
            commission_per_lot = float(engine_cfg["commission_per_lot"])
        except (KeyError, TypeError, ValueError):
            return "walk-forward provenance engine_config carries missing or non-numeric cost parameters"
        expected_flags = {
            "spread_costs_included": spread_points > 0 and costs_block.get("spread") is True,
            "slippage_costs_included": slippage_points > 0 and costs_block.get("slippage") is True,
            "commission_costs_included": commission_per_lot > 0 and costs_block.get("commission") is True,
        }
        for flag_key, expected in expected_flags.items():
            claimed = walk_forward_metrics.get(flag_key)
            if claimed is not expected:
                return (
                    f"payload {flag_key}={claimed!r} contradicts the producing run "
                    f"(run evidence: {expected}); cost claims must match the persisted evaluation"
                )
        return None

    def attach_walk_forward_metrics(
        self,
        model_id: str,
        walk_forward_metrics: dict,
        validation_metrics: Optional[dict] = None,
        test_metrics: Optional[dict] = None,
    ) -> MLModelRecord:
        """Record walk-forward backtest evidence on a candidate (Finding 5).

        This is the only supported way to change a candidate's evaluation
        payload after registration: it re-attests the digest (H1) and rewrites
        the artifact payload so the store and the artifact stay in agreement.
        Both must move together, otherwise promotion rejects the model as
        tampered. Optional validation_metrics/test_metrics replace the stored
        values in the same re-attested write.
        """
        record = self.db.query(MLModelRecord).filter_by(model_id=model_id).first()
        if not record:
            raise ValueError(f"Model {model_id} not found.")
        if record.status != "candidate":
            raise ModelGovernanceError(
                f"Cannot attach walk-forward metrics to model {model_id} with status "
                f"'{record.status}': the evaluation payload of a promoted model is immutable."
            )
        if not isinstance(walk_forward_metrics, dict) or not walk_forward_metrics:
            raise ValueError("walk_forward_metrics must be a non-empty dict.")

        # Provenance gate: walk-forward evidence must be traceable to a REAL,
        # persisted backtest run. Self-declared numbers with no producing run
        # are rejected before any state is touched.
        backtest_run_id = walk_forward_metrics.get("backtest_run_id")
        if not isinstance(backtest_run_id, str) or not backtest_run_id.strip():
            raise ModelGovernanceError(
                "walk_forward_metrics must carry provenance: a non-empty "
                "'backtest_run_id' identifying the producing backtest run."
            )
        backtest_run = self.db.query(BacktestRun).filter(BacktestRun.run_id == backtest_run_id).first()
        if backtest_run is None:
            raise ModelGovernanceError(
                f"walk_forward_metrics provenance failed: backtest run "
                f"'{backtest_run_id}' does not exist in backtest_runs."
            )
        # M8: the wf numbers must BE the stored run's numbers, not merely
        # point at a real run row.
        provenance_failure = self._verify_backtest_provenance(walk_forward_metrics, backtest_run)
        if provenance_failure:
            raise ModelGovernanceError(
                f"walk_forward_metrics provenance failed: {provenance_failure}"
            )

        record.walk_forward_metrics = walk_forward_metrics
        if validation_metrics is not None:
            if not isinstance(validation_metrics, dict) or not validation_metrics:
                raise ValueError("validation_metrics must be a non-empty dict when provided.")
            record.validation_metrics = validation_metrics
        if test_metrics is not None:
            if not isinstance(test_metrics, dict) or not test_metrics:
                raise ValueError("test_metrics must be a non-empty dict when provided.")
            record.test_metrics = test_metrics

        record.metrics_digest = metric_digest(
            record.model_id,
            record.validation_metrics,
            record.test_metrics,
            record.walk_forward_metrics,
        )

        # Same containment contract as promote/load: attach rewrites the
        # artifact file, so the stored path must stay inside the configured
        # artifact directory (a tampered record.artifact_path must not turn
        # attach into an arbitrary-file write).
        artifact_path = Path(record.artifact_path).resolve()
        artifact_root = Path(settings.ARTIFACTS_MODELS_DIR).resolve()
        try:
            artifact_path.relative_to(artifact_root)
        except ValueError as exc:
            raise ModelGovernanceError(
                f"Cannot attach walk-forward metrics for model {model_id}: "
                "artifact path is outside the configured artifact directory."
            ) from exc
        if artifact_path.is_file():
            payload = joblib.load(artifact_path)
            if isinstance(payload, dict):
                payload["walk_forward_metrics"] = walk_forward_metrics
                payload["validation_metrics"] = record.validation_metrics
                payload["test_metrics"] = record.test_metrics
                payload["metrics_digest"] = record.metrics_digest
                joblib.dump(payload, artifact_path)

        self.db.commit()
        self.db.refresh(record)
        logger.info(f"Attached walk-forward metrics and re-attested digest for {model_id}.")
        return record

    def attach_ml_walk_forward_metrics(self, model_id: str, payload: dict) -> MLModelRecord:
        """Record per-fold-retrained ML walk-forward evidence on a candidate (M8).

        The payload must be the output of run_ml_walk_forward (or at least pass
        the same independent re-verification), and it is committed together
        with a dedicated tamper-evident digest. A payload that fails
        structural verification (fold overlap, purge/embargo violation, forged
        aggregate, missing provenance) or whose digest does not match is
        rejected before any state changes.
        """
        record = self.db.query(MLModelRecord).filter_by(model_id=model_id).first()
        if not record:
            raise ValueError(f"Model {model_id} not found.")
        if record.status != "candidate":
            raise ModelGovernanceError(
                f"Cannot attach ML walk-forward metrics to model {model_id} with status "
                f"'{record.status}': the evaluation payload of a promoted model is immutable."
            )
        if not isinstance(payload, dict) or not payload:
            raise ValueError("ml_walk_forward payload must be a non-empty dict.")
        if payload.get("evidence_type") != MLWF_EVIDENCE_TYPE:
            raise ModelGovernanceError(
                "ml_walk_forward payload must carry evidence_type "
                f"'{MLWF_EVIDENCE_TYPE}'."
            )
        fold_count = payload.get("fold_count")
        if not isinstance(fold_count, int) or isinstance(fold_count, bool) or (
            fold_count < self.MIN_ML_WALK_FORWARD_FOLDS
        ):
            raise ModelGovernanceError(
                f"ML walk-forward evidence must contain at least "
                f"{self.MIN_ML_WALK_FORWARD_FOLDS} retrained folds."
            )

        structural_failure = verify_ml_walk_forward_payload(payload)
        if structural_failure:
            raise ModelGovernanceError(
                f"ml_walk_forward payload verification failed: {structural_failure}"
            )

        # Replay guard: a payload that ALREADY carries a digest signed for a
        # DIFFERENT model must be refused, never silently re-signed for this
        # one. Fresh evaluator output carries no digest and is signed here;
        # re-attaching a model's OWN payload stays idempotent (the digest is
        # recomputed over the digest-stripped payload below).
        incoming_digest = payload.get("digest")
        if incoming_digest is not None:
            if not isinstance(incoming_digest, str):
                raise ModelGovernanceError(
                    "ml_walk_forward payload digest must be a hex string "
                    f"(got {type(incoming_digest).__name__})"
                )
            from backend.app.core.metrics_integrity import verify_ml_walk_forward_digest

            payload_without_digest = {k: v for k, v in payload.items() if k != "digest"}
            if not verify_ml_walk_forward_digest(
                record.model_id, payload_without_digest, incoming_digest
            ):
                raise ModelGovernanceError(
                    f"ml_walk_forward payload carries a digest that does not belong to "
                    f"model {model_id} (tampered or replayed evidence); refusing to re-sign"
                )

        from backend.app.core.metrics_integrity import ml_walk_forward_digest

        record.ml_walk_forward_metrics = payload
        record.metrics_digest = metric_digest(
            record.model_id,
            record.validation_metrics,
            record.test_metrics,
            record.walk_forward_metrics,
        )
        # Dedicated digest over the ML WF payload itself — computed over the
        # digest-STRIPPED payload (a stale digest key must never participate
        # in its own replacement's message); stored alongside so promotion
        # (and any later reviewer) can re-verify independently.
        payload.pop("digest", None)
        payload["digest"] = ml_walk_forward_digest(record.model_id, payload)

        # Same containment contract as the other attach path.
        artifact_path = Path(record.artifact_path).resolve()
        artifact_root = Path(settings.ARTIFACTS_MODELS_DIR).resolve()
        try:
            artifact_path.relative_to(artifact_root)
        except ValueError as exc:
            raise ModelGovernanceError(
                f"Cannot attach ML walk-forward metrics for model {model_id}: "
                "artifact path is outside the configured artifact directory."
            ) from exc
        if artifact_path.is_file():
            artifact_payload = joblib.load(artifact_path)
            if isinstance(artifact_payload, dict):
                artifact_payload["ml_walk_forward_metrics"] = payload
                joblib.dump(artifact_payload, artifact_path)

        self.db.commit()
        self.db.refresh(record)
        log_security_event(
            "ML_WALK_FORWARD_EVIDENCE_ATTACHED",
            {"model_id": model_id, "fold_count": fold_count},
            "INFO",
        )
        return record

    @classmethod
    def _ml_walk_forward_failure(cls, model_id: str, payload: Any) -> Optional[str]:
        """Independent re-verification of attached ML walk-forward evidence."""
        if not isinstance(payload, dict) or not payload:
            return None  # absent evidence is handled by the strategy WF gate
        structural_failure = verify_ml_walk_forward_payload(payload)
        if structural_failure:
            return structural_failure
        digest = payload.get("digest")
        if not digest or not isinstance(digest, str) or not verify_ml_walk_forward_digest(
            model_id, {k: v for k, v in payload.items() if k != "digest"}, digest
        ):
            return "ml_walk_forward payload digest missing or mismatched (tampered evidence)"
        fold_count = payload.get("fold_count")
        if not isinstance(fold_count, int) or fold_count < cls.MIN_ML_WALK_FORWARD_FOLDS:
            return (
                f"ml_walk_forward evidence must contain at least "
                f"{cls.MIN_ML_WALK_FORWARD_FOLDS} retrained folds"
            )
        return None

    def promote_candidate(self, candidate_id: str, actor: str = "USER", notes: Optional[str] = None) -> MLModelRecord:
        """Controlled promotion of a candidate model to ACTIVE status.
        Archives previously active model.
        """
        candidate = self.db.query(MLModelRecord).filter_by(model_id=candidate_id).first()
        if not candidate:
            raise ValueError(f"Candidate model {candidate_id} not found.")

        if candidate.status == "rejected":
            raise ModelGovernanceError(f"Cannot promote model {candidate_id} because it was marked REJECTED.")
        if candidate.status != "candidate":
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id} from status {candidate.status!r}; "
                "only candidates can be promoted."
            )

        # H1: promotion reads attested metrics only. The digest committed at
        # registration must still verify against the store, and the artifact's
        # embedded metrics must match the store — otherwise the gates below
        # could be satisfied by tampered numbers.
        if not candidate.metrics_digest:
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: stored metrics are not attested "
                "(missing digest); re-register the candidate."
            )
        if not verify_metric_digest(
            candidate.model_id,
            candidate.validation_metrics,
            candidate.test_metrics,
            candidate.walk_forward_metrics,
            candidate.metrics_digest,
        ):
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: stored metrics failed integrity "
                "attestation (tamper detected)."
            )

        # A model must have a readable artifact before the active model is touched.
        artifact_path = Path(candidate.artifact_path).resolve()
        artifact_root = Path(settings.ARTIFACTS_MODELS_DIR).resolve()
        try:
            artifact_path.relative_to(artifact_root)
        except ValueError as exc:
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: artifact path is outside the configured artifact directory."
            ) from exc
        if not artifact_path.is_file() or artifact_path.suffix.lower() != ".joblib":
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: artifact file is missing or invalid."
            )

        val_metrics = candidate.validation_metrics
        test_metrics = candidate.test_metrics
        val_auc = self._metric_value(val_metrics, "roc_auc")
        val_brier = self._metric_value(val_metrics, "brier_score")
        test_auc = self._metric_value(test_metrics, "roc_auc")
        test_brier = self._metric_value(test_metrics, "brier_score")
        if (
            val_auc is None or val_brier is None or test_auc is None or test_brier is None
            or self._sample_count(val_metrics) < self.MIN_PROMOTION_EVALUATION_SAMPLES
            or self._sample_count(test_metrics) < self.MIN_PROMOTION_EVALUATION_SAMPLES
        ):
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: validation and test partitions must each have "
                f"valid ROC-AUC/Brier metrics and at least {self.MIN_PROMOTION_EVALUATION_SAMPLES} samples."
            )
        if (
            val_auc <= self.MIN_PROMOTION_ROC_AUC
            or test_auc <= self.MIN_PROMOTION_ROC_AUC
            or val_brier > self.MAX_PROMOTION_BRIER_SCORE
            or test_brier > self.MAX_PROMOTION_BRIER_SCORE
        ):
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: validation and independent test metrics do not pass "
                f"(ROC-AUC must be > {self.MIN_PROMOTION_ROC_AUC:.2f}; "
                f"Brier score must be <= {self.MAX_PROMOTION_BRIER_SCORE:.2f})."
            )

        # Classification metrics alone do not prove that this model is profitable
        # when used as a trading strategy. Do not activate it without a separate,
        # cost-aware walk-forward backtest on unseen periods.
        walk_forward_failure = self._walk_forward_failure(candidate.walk_forward_metrics)
        if walk_forward_failure:
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: {walk_forward_failure}"
            )

        # M8: walk-forward evidence must still be traceable to its persisted
        # producing run at promotion time, and its numbers must match that run.
        # A payload that bypassed attach (registered directly) without
        # backtest_run_id, or whose producing run was deleted afterwards, is a
        # provenance failure — the numbers alone prove nothing.
        wf_metrics = candidate.walk_forward_metrics or {}
        backtest_run_id = wf_metrics.get("backtest_run_id")
        if not isinstance(backtest_run_id, str) or not backtest_run_id.strip():
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: walk-forward evidence lacks "
                "'backtest_run_id' provenance (re-attach it via attach_walk_forward_metrics)."
            )
        backtest_run = self.db.query(BacktestRun).filter(BacktestRun.run_id == backtest_run_id).first()
        if backtest_run is None:
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: referenced backtest run "
                f"'{backtest_run_id}' no longer exists in backtest_runs."
            )
        provenance_failure = self._verify_backtest_provenance(wf_metrics, backtest_run)
        if provenance_failure:
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: {provenance_failure}"
            )

        # M8: per-fold-retrained ML walk-forward evidence, when attached, must
        # still verify independently at promotion time — a payload tampered
        # after attach (fold geometry edited, aggregate forged, digest
        # stripped) must block the promotion instead of riding along.
        ml_wf_failure = self._ml_walk_forward_failure(
            candidate.model_id, candidate.ml_walk_forward_metrics
        )
        if ml_wf_failure:
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: ml_walk_forward evidence failed: {ml_wf_failure}"
            )

        # Test-set values are reserved for absolute gates. Compare candidates to
        # the active model on validation only to avoid repeatedly tuning against test.
        active_current = self.get_active_model_record()
        if active_current and active_current.model_id != candidate_id:
            active_val_auc = self._metric_value(active_current.validation_metrics, "roc_auc")
            if (
                active_val_auc is None
                or self._sample_count(active_current.validation_metrics) < self.MIN_PROMOTION_EVALUATION_SAMPLES
            ):
                raise ModelGovernanceError(
                    "Cannot promote candidate: the active model must be re-evaluated on a valid temporal validation partition."
                )
            if val_auc <= active_val_auc:
                raise ModelGovernanceError(
                    f"Cannot promote model {candidate_id}: validation ROC-AUC {val_auc:.4f} "
                    f"does not exceed active model validation ROC-AUC {active_val_auc:.4f}."
                )

        # Verify that the persisted artifact is loadable and has the required inference contract.
        try:
            artifact = joblib.load(artifact_path)
        except Exception as exc:
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: artifact cannot be loaded ({type(exc).__name__})."
            ) from exc
        if (
            not isinstance(artifact, dict)
            or not callable(getattr(artifact.get("model"), "predict_proba", None))
            or not isinstance(artifact.get("feature_names"), (list, tuple))
            or not artifact.get("feature_names")
        ):
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: artifact is missing inference components."
            )

        # H1: cross-check the artifact's embedded metric payload against the
        # attested store values so neither side can be edited independently.
        artifact_metrics_digest = artifact.get("metrics_digest")
        # A digest-less artifact is a provenance FAILURE, not a pass: without
        # an attested digest the payload cannot be verified against the store.
        if artifact_metrics_digest is None:
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: the artifact carries no attested "
                "metrics digest (H1 provenance required)."
            )
        if artifact_metrics_digest != candidate.metrics_digest:
            raise ModelGovernanceError(
                f"Cannot promote model {candidate_id}: artifact metrics digest does not match the attested store digest."
            )
        for metric_key in ("validation_metrics", "test_metrics"):
            artifact_metrics = artifact.get(metric_key)
            if artifact_metrics is not None and artifact_metrics != getattr(candidate, metric_key):
                raise ModelGovernanceError(
                    f"Cannot promote model {candidate_id}: artifact {metric_key} do not match "
                    "the attested store metrics (artifact tamper detected)."
                )

        # Mutate model states only after every promotion gate has passed.
        if active_current and active_current.model_id != candidate_id:
            active_current.status = "archived"
            logger.info(f"Archived previous active model {active_current.model_id}.")

        candidate.status = "active"
        candidate.promotion_notes = notes or "Promoted after validation, independent-test, and walk-forward gates passed."
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        self.db.refresh(candidate)

        log_security_event(
            "MODEL_PROMOTED",
            {
                "model_id": candidate_id,
                "actor": actor,
                "previous_active": active_current.model_id if active_current else None,
            },
            "INFO"
        )
        return candidate
