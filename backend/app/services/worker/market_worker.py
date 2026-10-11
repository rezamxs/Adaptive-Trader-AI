"""Market Monitoring Background Worker for continuous market scanning and signal generation.
Strictly Read-Only and Fail-Closed: Operates strictly in SIGNAL_ONLY mode and NEVER executes orders.
"""
import asyncio
import uuid
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.core.config import settings, ExecutionMode
from backend.app.core.logging import logger, log_security_event
from backend.app.db.session import SessionLocal
from backend.app.models.worker_cycle import WorkerCycleRecord
from backend.app.models.signal import TradeSignal
from backend.app.services.market_data.base import IMarketDataProvider, TimeFrame, Candle
from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider
from backend.app.services.knowledge.manager import KnowledgeManager
from backend.app.services.ml.registry import ModelRegistry
from backend.app.services.dataset.features import FeatureEngineer
from backend.app.services.signal.generator import SignalGenerator
from backend.app.services.signal.manager import SignalManager
from backend.app.services.risk.circuit_breaker import EmergencyStopManager


class MarketMonitoringWorker:
    """Continuous asynchronous market scanner running in FastAPI background lifespan."""

    def __init__(
        self,
        data_provider: Optional[IMarketDataProvider] = None,
        db_factory = SessionLocal,
        symbol: Optional[str] = None,
        timeframe: Optional[TimeFrame] = None,
        poll_interval: Optional[float] = None,
    ):
        self.data_provider = data_provider or MT5MarketDataProvider()
        self.db_factory = db_factory
        self.symbol = symbol or settings.WORKER_SYMBOL
        raw_tf = timeframe or settings.WORKER_TIMEFRAME
        if isinstance(raw_tf, TimeFrame):
            self.timeframe = raw_tf
        else:
            self.timeframe = TimeFrame(raw_tf) if raw_tf in [t.value for t in TimeFrame] else TimeFrame.M1

        self.poll_interval = poll_interval or settings.WORKER_POLL_INTERVAL_SECONDS
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Telemetry & Diagnostics State
        self.started_at: Optional[datetime] = None
        self.last_heartbeat: Optional[datetime] = None
        self.last_cycle_time: Optional[datetime] = None
        self.last_successful_cycle_time: Optional[datetime] = None
        self.last_processed_bar_time: Optional[datetime] = None
        self.last_processed_bar_key: Optional[str] = None
        self.last_status_code: str = "INITIALIZING"
        self.last_decision: str = "NONE"
        self.last_reason: str = "Worker initialized"
        self.last_signal_id: Optional[str] = None
        self.last_error_message: Optional[str] = None
        
        self.total_cycles: int = 0
        self.successful_cycles: int = 0
        self.consecutive_errors: int = 0
        self.total_errors: int = 0

    @property
    def liveness_status(self) -> str:
        """Process liveness: LIVE if running and heartbeat fresh within 30s, otherwise STOPPED/STALE."""
        if not self.is_running:
            return "STOPPED"
        if not self.last_heartbeat:
            return "STARTING"
        age = (datetime.now(timezone.utc) - self.last_heartbeat).total_seconds()
        return "LIVE" if age <= 30.0 else "STALE"

    @property
    def readiness_status(self) -> str:
        """Pipeline readiness: READY, DEGRADED, STOPPED, or ERROR."""
        if not self.is_running:
            return "STOPPED"
        if self.consecutive_errors > 3:
            return "ERROR"
        if self.last_status_code in ["MT5_DISCONNECTED", "MARKET_CLOSED", "MARKET_DATA_STALE", "EMERGENCY_STOP_ACTIVE"]:
            return "DEGRADED"
        return "READY"

    def get_diagnostics(self) -> Dict[str, Any]:
        """Provides full structured telemetry for API inspection."""
        now_utc = datetime.now(timezone.utc)
        heartbeat_age = (now_utc - self.last_heartbeat).total_seconds() if self.last_heartbeat else None
        last_cycle_age = (now_utc - self.last_cycle_time).total_seconds() if self.last_cycle_time else None

        return {
            "liveness": self.liveness_status,
            "readiness": self.readiness_status,
            "is_running": self.is_running,
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "poll_interval_seconds": self.poll_interval,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "heartbeat_age_seconds": round(heartbeat_age, 1) if heartbeat_age is not None else None,
            "last_cycle_time": self.last_cycle_time.isoformat() if self.last_cycle_time else None,
            "last_cycle_age_seconds": round(last_cycle_age, 1) if last_cycle_age is not None else None,
            "last_successful_cycle_time": self.last_successful_cycle_time.isoformat() if self.last_successful_cycle_time else None,
            "last_processed_bar_time": self.last_processed_bar_time.isoformat() if self.last_processed_bar_time else None,
            "last_processed_bar_key": self.last_processed_bar_key,
            "last_status_code": self.last_status_code,
            "last_decision": self.last_decision,
            "last_reason": self.last_reason,
            "last_signal_id": self.last_signal_id,
            "last_error_message": self.last_error_message,
            "total_cycles": self.total_cycles,
            "successful_cycles": self.successful_cycles,
            "consecutive_errors": self.consecutive_errors,
            "total_errors": self.total_errors,
            "execution_mode": settings.EXECUTION_MODE.value,
            "demo_trading_enabled": settings.DEMO_TRADING_ENABLED,
        }

    async def start(self) -> None:
        """Starts background worker task safely."""
        async with self._lock:
            if self.is_running:
                logger.warning("MarketMonitoringWorker start called but worker is already running.")
                return

            # Safety assertion
            if settings.EXECUTION_MODE != ExecutionMode.SIGNAL_ONLY or settings.DEMO_TRADING_ENABLED:
                raise RuntimeError(
                    f"CRITICAL SAFETY VIOLATION: Cannot start Worker in mode '{settings.EXECUTION_MODE.value}' "
                    f"with DEMO_TRADING_ENABLED={settings.DEMO_TRADING_ENABLED}. Must be strictly SIGNAL_ONLY."
                )

            self.is_running = True
            self.started_at = datetime.now(timezone.utc)
            self.last_heartbeat = self.started_at
            self.consecutive_errors = 0
            self.last_status_code = "STARTED"
            self.last_reason = "Worker started successfully in SIGNAL_ONLY mode."
            self._task = asyncio.create_task(self._run_loop())
            logger.info(f"MarketMonitoringWorker started monitoring {self.symbol} ({self.timeframe.value}) every {self.poll_interval}s.")

    async def stop(self) -> None:
        """Stops background worker task and disconnects data provider."""
        async with self._lock:
            if not self.is_running:
                return

            self.is_running = False
            self.last_status_code = "STOPPING"
            if self._task:
                self._task.cancel()
                try:
                    await asyncio.wait_for(self._task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                self._task = None

            # Disconnect data provider in thread
            await asyncio.to_thread(self.data_provider.disconnect)
            self.last_status_code = "STOPPED"
            self.last_reason = "Worker stopped cleanly on application shutdown."
            logger.info("MarketMonitoringWorker stopped cleanly.")

    async def _run_loop(self) -> None:
        """Continuous execution loop with exponential backoff on errors."""
        while self.is_running:
            self.last_heartbeat = datetime.now(timezone.utc)
            cycle_start = time.perf_counter()

            try:
                # Run the single cycle in worker thread to prevent event loop blocking
                await asyncio.to_thread(self.run_single_cycle)
                self.consecutive_errors = 0
                self.last_successful_cycle_time = datetime.now(timezone.utc)
                current_delay = self.poll_interval
            except Exception as e:
                self.consecutive_errors += 1
                self.total_errors += 1
                self.last_error_message = str(e)
                self.last_status_code = "WORKER_EXCEPTION"
                self.last_reason = f"Unhandled error in cycle: {str(e)}"
                logger.error(f"MarketMonitoringWorker cycle exception ({self.consecutive_errors} consecutive): {e}")

                # Exponential backoff with ceiling
                backoff_multiplier = settings.WORKER_RETRY_BACKOFF_BASE ** min(self.consecutive_errors, 5)
                current_delay = min(settings.WORKER_MAX_RETRY_DELAY_SECONDS, self.poll_interval * backoff_multiplier)

            cycle_duration = time.perf_counter() - cycle_start
            sleep_time = max(0.1, current_delay - cycle_duration)

            try:
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                break

    def run_single_cycle(self) -> Dict[str, Any]:
        """Executes a single synchronous market monitoring pass.
        Strictly Read-Only and Idempotent.
        """
        cycle_id = f"cyc_{uuid.uuid4().hex[:10]}"
        t0 = time.perf_counter()
        now_utc = datetime.now(timezone.utc)
        self.total_cycles += 1
        self.last_cycle_time = now_utc

        # 1. Strict Security Gate Assertion (Fail-Closed)
        if settings.EXECUTION_MODE != ExecutionMode.SIGNAL_ONLY or settings.DEMO_TRADING_ENABLED:
            self.last_status_code = "SECURITY_VIOLATION_BLOCKED"
            self.last_reason = "Execution mode is not strictly SIGNAL_ONLY or DEMO_TRADING_ENABLED is True."
            return {"cycle_id": cycle_id, "status_code": self.last_status_code, "reason": self.last_reason}

        db: Session = self.db_factory()
        try:
            # 2. Check Emergency Stop Circuit Breaker
            stop_mgr = EmergencyStopManager(db)
            if stop_mgr.is_active():
                self.last_status_code = "EMERGENCY_STOP_ACTIVE"
                self.last_reason = "Emergency Stop Circuit Breaker is active. Pipeline suspended."
                self.last_decision = "HOLD"
                return {"cycle_id": cycle_id, "status_code": self.last_status_code, "reason": self.last_reason}

            # 3. Check / Establish MT5 Connection
            if not self.data_provider.is_connected():
                ok = self.data_provider.connect()
                if not ok:
                    self.last_status_code = "MT5_DISCONNECTED"
                    self.last_reason = "Failed to connect to MetaTrader 5 terminal."
                    self.last_decision = "NONE"
                    return {"cycle_id": cycle_id, "status_code": self.last_status_code, "reason": self.last_reason}

            # 4. Market Status and Quote Freshness
            quote = self.data_provider.get_current_quote(self.symbol, max_staleness_seconds=settings.WORKER_STALE_TICK_SECONDS)
            if not quote or quote.is_stale:
                tick = getattr(self.data_provider, "get_last_tick", lambda s: None)(self.symbol) if hasattr(self.data_provider, "get_last_tick") else None
                # If quote is very stale (> 2 hours), market is closed
                if quote and (now_utc - quote.time).total_seconds() > 7200:
                    self.last_status_code = "MARKET_CLOSED"
                    self.last_reason = f"Market is currently closed (Last tick age: {(now_utc - quote.time).total_seconds() / 3600:.1f} hours)."
                else:
                    self.last_status_code = "MARKET_DATA_STALE"
                    self.last_reason = "Market quote is stale or unavailable."
                self.last_decision = "HOLD"
                # Even if market is closed, we proceed to check if there is an unhandled closed bar from prior session

            # 5. Retrieve Historical Completed Candles (start_pos=1 guarantees closed bars only)
            candles = self.data_provider.get_historical_candles(
                symbol=self.symbol,
                timeframe=self.timeframe,
                count=100,
                start_pos=1
            )

            if not candles or len(candles) < 50:
                self.last_status_code = "DATA_VALIDATION_FAILED"
                self.last_reason = f"Insufficient closed bars ({len(candles) if candles else 0} < 50) retrieved for {self.symbol}."
                self.last_decision = "NONE"
                return {"cycle_id": cycle_id, "status_code": self.last_status_code, "reason": self.last_reason}

            # Identify latest completed candle
            latest_bar: Candle = candles[-1]
            bar_time_utc = latest_bar.time
            bar_key = f"{self.symbol}_{self.timeframe.value}_{bar_time_utc.strftime('%Y%m%d%H%M')}"

            # 6. Check Bar Deduplication in Database
            existing_record = db.query(WorkerCycleRecord).filter_by(bar_key=bar_key).first()
            if existing_record:
                # We already processed this exact closed bar! Prevent duplicate signal generation.
                self.last_status_code = "NO_NEW_CLOSED_BAR"
                self.last_reason = f"Latest closed bar {bar_key} has already been analyzed in cycle {existing_record.cycle_id}."
                self.last_decision = existing_record.decision or "HOLD"
                self.successful_cycles += 1
                return {
                    "cycle_id": cycle_id,
                    "bar_key": bar_key,
                    "status_code": self.last_status_code,
                    "reason": self.last_reason,
                    "bar_time": bar_time_utc.isoformat(),
                }

            # 7. A NEW CLOSED BAR IS AVAILABLE! Process through Analysis Pipeline
            self.last_processed_bar_time = bar_time_utc
            self.last_processed_bar_key = bar_key

            # Convert candles to DataFrame for Feature Engineering
            records = [{
                "time": c.time,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "tick_volume": c.tick_volume,
                "spread": c.spread,
            } for c in candles]
            df_raw = pd.DataFrame(records).set_index("time").sort_index(ascending=True)

            try:
                df_feats, _ = FeatureEngineer.transform(df_raw)
            except Exception as fe_err:
                self.last_status_code = "FEATURE_GENERATION_FAILED"
                self.last_reason = f"Feature generation error: {str(fe_err)}"
                self._record_cycle_in_db(db, cycle_id, bar_key, bar_time_utc, self.last_status_code, "HOLD", None, t0, self.last_reason)
                return {"cycle_id": cycle_id, "status_code": self.last_status_code, "reason": self.last_reason}

            latest_feat_series = df_feats.iloc[-1]

            # 8. Check Governance: Approved Rules and Active ML Model
            knowledge_mgr = KnowledgeManager(db)
            approved_rules = knowledge_mgr.get_active_strategy_rules()
            model_reg = ModelRegistry(db)
            active_model_artifact = model_reg.load_active_model_artifact()

            # Evaluate rules & model
            rule_dir, rule_str, triggered_rules = SignalGenerator.evaluate_rules(latest_feat_series, approved_rules)
            model_dir, model_prob = SignalGenerator.evaluate_model(latest_feat_series, active_model_artifact)

            # Determine direction & status code
            final_direction = "HOLD"
            status_code = "NO_VALID_SIGNAL"
            reason_details = []

            if len(approved_rules) == 0:
                reason_details.append("NO_APPROVED_RULES")
            if active_model_artifact is None:
                reason_details.append("NO_ACTIVE_MODEL")

            if rule_dir != "HOLD" and model_dir == rule_dir:
                final_direction = rule_dir
            elif rule_dir != "HOLD" and model_dir is None:
                final_direction = rule_dir
            elif model_dir is not None and model_dir != "HOLD" and len(approved_rules) == 0:
                final_direction = model_dir

            signal_id_created = None
            if final_direction == "HOLD":
                if reason_details:
                    status_code = "_AND_".join(reason_details)
                    reason_msg = f"Hold: System lacks active components ({status_code})."
                else:
                    status_code = "NO_VALID_SIGNAL"
                    reason_msg = "Hold: Neither rules nor ML model produced an entry trigger on latest closed bar."
            else:
                # Valid directional trigger found!
                status_code = "SIGNAL_DETECTED_EXECUTION_DISABLED"
                reason_msg = f"Valid {final_direction} signal detected on {bar_key}. Execution disabled in SIGNAL_ONLY mode."

                # Persist TradeSignal record using SignalManager
                sig_mgr = SignalManager(db=db, data_provider=self.data_provider)
                sig_record = sig_mgr.process_latest_bar(symbol=self.symbol, timeframe=self.timeframe)
                if sig_record:
                    signal_id_created = sig_record.signal_id
                    self.last_signal_id = signal_id_created

            # 9. Persist Cycle Record in Database
            self.last_status_code = status_code
            self.last_decision = final_direction
            self.last_reason = reason_msg
            self.successful_cycles += 1

            self._record_cycle_in_db(
                db=db,
                cycle_id=cycle_id,
                bar_key=bar_key,
                bar_time=bar_time_utc,
                status_code=status_code,
                decision=final_direction,
                signal_id=signal_id_created,
                t0=t0,
                details=reason_msg
            )

            return {
                "cycle_id": cycle_id,
                "bar_key": bar_key,
                "bar_time": bar_time_utc.isoformat(),
                "status_code": status_code,
                "decision": final_direction,
                "signal_id": signal_id_created,
                "reason": reason_msg,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        finally:
            db.close()

    def _record_cycle_in_db(
        self,
        db: Session,
        cycle_id: str,
        bar_key: Optional[str],
        bar_time: Optional[datetime],
        status_code: str,
        decision: str,
        signal_id: Optional[str],
        t0: float,
        details: str,
    ) -> None:
        """Saves a WorkerCycleRecord to database, safely catching any write errors."""
        try:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            rec = WorkerCycleRecord(
                cycle_id=cycle_id,
                symbol=self.symbol,
                timeframe=self.timeframe.value,
                bar_key=bar_key,
                bar_time=bar_time,
                status_code=status_code,
                decision=decision,
                signal_id=signal_id,
                duration_ms=dur_ms,
                details=details,
                error_message=self.last_error_message,
            )
            db.add(rec)
            db.commit()
        except Exception as db_err:
            db.rollback()
            logger.error(f"Failed to record WorkerCycleRecord in DB: {db_err}")
            self.last_status_code = "DATABASE_WRITE_FAILED"
            self.last_reason = f"Database commit failed: {db_err}"


# Global worker instance holder
_global_worker: Optional[MarketMonitoringWorker] = None


def get_market_worker() -> MarketMonitoringWorker:
    """Returns the singleton MarketMonitoringWorker instance."""
    global _global_worker
    if _global_worker is None:
        _global_worker = MarketMonitoringWorker()
    return _global_worker
