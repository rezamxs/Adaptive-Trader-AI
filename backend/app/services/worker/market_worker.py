"""Market Monitoring Background Worker for continuous market scanning and signal generation.
Strictly Read-Only and Fail-Closed: Operates strictly in SIGNAL_ONLY mode and NEVER executes orders.
"""
import asyncio
import uuid
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.core.config import settings, ExecutionMode
from backend.app.core.logging import logger, log_security_event
from backend.app.db.session import SessionLocal
from backend.app.db.base import utc_now_naive
from backend.app.models.worker_cycle import WorkerCycleRecord
from backend.app.models.signal import TradeSignal
from backend.app.models.audit_log import AuditLog
from backend.app.services.market_data.base import IMarketDataProvider, TimeFrame, Candle, Quote
from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider
from backend.app.services.knowledge.manager import KnowledgeManager
from backend.app.services.ml.registry import ModelRegistry
from backend.app.services.execution.demo_executor import DemoTradingExecutor
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
        
        # Taxonomy & Freshness State
        self.mt5_status: str = "MT5_DISCONNECTED"
        self.market_status: str = "INITIALIZING"
        self.market_staleness_reason: Optional[str] = None
        self.last_tick_time: Optional[datetime] = None
        self.last_lifecycle_phase: str = "IDLE"
        
        # Explicit Event Timestamps
        self.last_poll_time: Optional[datetime] = None
        self.last_successful_analysis_time: Optional[datetime] = None
        self.last_error_time: Optional[datetime] = None
        
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
        if self.mt5_status == "MT5_DISCONNECTED" or self.last_status_code == "EMERGENCY_STOP_ACTIVE":
            return "DEGRADED"
        return "READY"

    @property
    def service_ready(self) -> bool:
        """Service infrastructure readiness: True if process is alive, heartbeat fresh, and no breaker trip."""
        return (
            self.is_running
            and self.liveness_status == "LIVE"
            and self.consecutive_errors <= 3
            and self.last_status_code != "EMERGENCY_STOP_ACTIVE"
        )

    @property
    def service_ready_reason(self) -> str:
        if not self.is_running:
            return "SERVICE_COMPONENT_UNAVAILABLE (Worker stopped)"
        if self.liveness_status != "LIVE":
            return "SERVICE_HEARTBEAT_STALE"
        if self.consecutive_errors > 3:
            return f"EXCESSIVE_CONSECUTIVE_ERRORS ({self.consecutive_errors})"
        if self.last_status_code == "EMERGENCY_STOP_ACTIVE":
            return "SAFETY_GATE_BLOCKED (Emergency stop circuit breaker active)"
        return "SERVICE_HEALTHY_AND_OPERATIONAL"

    @property
    def market_data_ready(self) -> bool:
        """Market feed readiness: True ONLY if MT5 is connected and live quotes are fresh."""
        return (
            self.mt5_status == "MT5_CONNECTED"
            and self.market_status == "MARKET_OPEN_DATA_FRESH"
        )

    @property
    def market_data_reason(self) -> str:
        if self.mt5_status != "MT5_CONNECTED":
            return "MT5_DISCONNECTED"
        if self.market_status == "MARKET_CLOSED":
            return self.market_staleness_reason or "MARKET_CLOSED"
        if self.market_status in ["NO_RECENT_TICK", "MARKET_DATA_STALE", "MARKET_CLOSED_OR_STALE"]:
            return self.market_staleness_reason or self.market_status
        if self.market_status == "MARKET_OPEN_DATA_FRESH":
            return "MARKET_DATA_FRESH_AND_STREAMING"
        return self.market_status

    def trading_governance_probe(self) -> Tuple[bool, str]:
        """Single governance probe behind both trading-readiness views: it
        short-circuits on service/market readiness, then checks approved rules
        and the active ML artifact with ONE artifact load. A diagnostics or
        /health poll that reads both views previously deserialized the pickle
        artifact twice from disk (O(artifact size) per poll); this method
        keeps the two views consistent from one shared evaluation.

        The rules query mirrors KnowledgeManager.get_active_strategy_rules,
        including its CRITICAL invariant: approval_status == 'approved' ONLY.
        Constructing KnowledgeManager here would eagerly build the whole BM25
        corpus index, which a probe evaluated around every worker cycle must
        not pay for (item 8) — hence the direct query.
        """
        if not self.service_ready:
            return False, f"SERVICE_NOT_READY ({self.service_ready_reason})"
        if not self.market_data_ready:
            return False, f"MARKET_DATA_NOT_READY ({self.market_data_reason})"

        db = self.db_factory()
        try:
            from backend.app.models.rule import TradingRule
            from backend.app.schemas.rule_schema import ApprovalStatus
            from backend.app.services.ml.registry import ModelRegistry
            rules = db.query(TradingRule).filter(
                TradingRule.approval_status == ApprovalStatus.APPROVED.value
            ).all()
            model = ModelRegistry(db).load_active_model_artifact()
            missing = []
            if len(rules) == 0:
                missing.append("NO_APPROVED_RULES")
            if model is None:
                missing.append("NO_ACTIVE_MODEL")
            if missing:
                return False, "_AND_".join(missing)
            return True, "READY_FOR_SIGNAL_GENERATION"
        except Exception as e:
            return False, f"GOVERNANCE_CHECK_FAILED: {e}"
        finally:
            db.close()

    @property
    def trading_ready(self) -> bool:
        """Signal generation readiness: True ONLY if service is ready, market data is fresh,
        and approved strategy rules and active ML model artifact exist in database.
        NOTE: Never enables trade execution; strictly governs signal generation validity.
        """
        return self.trading_governance_probe()[0]

    @property
    def trading_readiness_reason(self) -> str:
        return self.trading_governance_probe()[1]

    def evaluate_market_freshness(self, quote: Optional[Quote], now_utc: datetime) -> Tuple[str, str]:
        """Evaluates market state independently:
        Returns (market_status, reason)
        Statuses: MARKET_OPEN_DATA_FRESH, NO_RECENT_TICK, MARKET_CLOSED, MARKET_CLOSED_OR_STALE
        """
        if not quote:
            return "NO_RECENT_TICK", "No quote or tick data received from broker terminal."
        if not isinstance(quote.time, datetime) or quote.time.tzinfo is None or quote.time.utcoffset() is None:
            return "INVALID_TICK_TIMESTAMP", "Quote timestamp must be timezone-aware UTC."
        if not isinstance(now_utc, datetime) or now_utc.tzinfo is None or now_utc.utcoffset() is None:
            return "INVALID_WORKER_CLOCK", "Worker evaluation time must be timezone-aware UTC."

        quote_time_utc = quote.time.astimezone(timezone.utc)
        now_utc = now_utc.astimezone(timezone.utc)
        age_seconds = (now_utc - quote_time_utc).total_seconds()
        if age_seconds < -5.0:
            return "FUTURE_TICK_TIMESTAMP", (
                f"Quote timestamp is {-age_seconds:.1f}s in the future; freshness cannot be verified."
            )
        if age_seconds <= settings.WORKER_STALE_TICK_SECONDS:
            return "MARKET_OPEN_DATA_FRESH", f"Market is open with live quotes (tick age {age_seconds:.1f}s)."
        
        # Check if UTC weekday falls in weekend session closure
        weekday = now_utc.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
        hour = now_utc.hour
        is_weekend = (
            weekday == 5  # Saturday
            or (weekday == 6 and hour < 22)  # Sunday before 22:00 UTC
            or (weekday == 4 and hour >= 23)  # Friday after 23:00 UTC
        )
        
        if is_weekend:
            return "MARKET_CLOSED", f"Market is closed for the weekend (UTC day={now_utc.strftime('%A')}, tick age={age_seconds/3600:.1f}h)."
        elif age_seconds > 7200:  # Over 2 hours outside weekend
            return "MARKET_CLOSED_OR_STALE", f"Market data stale for {age_seconds/3600:.1f} hours (potential holiday or session closure)."
        else:
            return "NO_RECENT_TICK", f"Quote is stale ({age_seconds:.1f}s > {settings.WORKER_STALE_TICK_SECONDS}s threshold)."

    def get_diagnostics(self) -> Dict[str, Any]:
        """Provides full structured telemetry for API inspection."""
        now_utc = datetime.now(timezone.utc)
        heartbeat_age = (now_utc - self.last_heartbeat).total_seconds() if self.last_heartbeat else None
        last_cycle_age = (now_utc - self.last_cycle_time).total_seconds() if self.last_cycle_time else None
        # One shared governance probe for both readiness views: a single
        # artifact load per diagnostics poll instead of two.
        probe_ready, probe_reason = self.trading_governance_probe()

        return {
            "liveness": self.liveness_status,
            "readiness": self.readiness_status,
            "service_ready": self.service_ready,
            "service_ready_reason": self.service_ready_reason,
            "market_data_ready": self.market_data_ready,
            "market_data_reason": self.market_data_reason,
            "trading_ready": probe_ready,
            "trading_readiness_reason": probe_reason,
            "is_running": self.is_running,
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "poll_interval_seconds": self.poll_interval,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "heartbeat_age_seconds": round(heartbeat_age, 1) if heartbeat_age is not None else None,
            "last_cycle_time": self.last_cycle_time.isoformat() if self.last_cycle_time else None,
            "last_cycle_age_seconds": round(last_cycle_age, 1) if last_cycle_age is not None else None,
            "last_poll_time": self.last_poll_time.isoformat() if self.last_poll_time else None,
            "last_successful_analysis_time": self.last_successful_analysis_time.isoformat() if self.last_successful_analysis_time else None,
            "last_successful_cycle_time": self.last_successful_cycle_time.isoformat() if self.last_successful_cycle_time else None,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "last_processed_bar_time": self.last_processed_bar_time.isoformat() if self.last_processed_bar_time else None,
            "last_processed_bar_key": self.last_processed_bar_key,
            "last_status_code": self.last_status_code,
            "last_decision": self.last_decision,
            "last_reason": self.last_reason,
            "last_signal_id": self.last_signal_id,
            "last_error_message": self.last_error_message,
            "mt5_status": self.mt5_status,
            "market_status": self.market_status,
            "market_staleness_reason": self.market_staleness_reason,
            "last_tick_time": self.last_tick_time.isoformat() if self.last_tick_time else None,
            "last_lifecycle_phase": self.last_lifecycle_phase,
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
                cycle_res = await asyncio.to_thread(self.run_single_cycle)
                self.consecutive_errors = 0
                self.last_poll_time = datetime.now(timezone.utc)
                if cycle_res.get("analysis_executed"):
                    self.last_successful_analysis_time = self.last_poll_time
                    self.last_successful_cycle_time = self.last_poll_time
                current_delay = self.poll_interval
            except Exception as e:
                self.consecutive_errors += 1
                self.total_errors += 1
                self.last_error_time = datetime.now(timezone.utc)
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
            return {"cycle_id": cycle_id, "status_code": self.last_status_code, "reason": self.last_reason, "analysis_executed": False}

        db: Session = self.db_factory()
        try:
            # 2. Check Emergency Stop Circuit Breaker
            stop_mgr = EmergencyStopManager(db)
            if stop_mgr.is_active():
                self.last_status_code = "EMERGENCY_STOP_ACTIVE"
                self.last_reason = "Emergency Stop Circuit Breaker is active. Pipeline suspended."
                self.last_decision = "HOLD"
                return {"cycle_id": cycle_id, "status_code": self.last_status_code, "reason": self.last_reason, "analysis_executed": False}

            # 3. Check / Establish MT5 Connection
            if not self.data_provider.is_connected():
                ok = self.data_provider.connect()
                if not ok:
                    self.mt5_status = "MT5_DISCONNECTED"
                    self.market_status = "UNKNOWN"
                    self.last_status_code = "MT5_DISCONNECTED"
                    self.last_reason = "Failed to connect to MetaTrader 5 terminal."
                    self.last_decision = "NONE"
                    return {"cycle_id": cycle_id, "status_code": self.last_status_code, "reason": self.last_reason, "analysis_executed": False}
            self.mt5_status = "MT5_CONNECTED"

            # 4. Market Status and Quote Freshness
            quote = self.data_provider.get_current_quote(self.symbol, max_staleness_seconds=settings.WORKER_STALE_TICK_SECONDS)
            m_status, m_reason = self.evaluate_market_freshness(quote, now_utc)
            self.market_status = m_status
            self.market_staleness_reason = m_reason
            if quote:
                self.last_tick_time = quote.time

            # 5. Retrieve Historical Completed Candles (start_pos=1 guarantees closed bars only)
            candles = self.data_provider.get_historical_candles(
                symbol=self.symbol,
                timeframe=self.timeframe,
                count=100,
                start_pos=1
            )

            if not candles or len(candles) < 50:
                self.last_status_code = "DATA_INVALID"
                self.last_reason = f"Insufficient or invalid closed bars ({len(candles) if candles else 0} < 50) retrieved for {self.symbol}."
                self.last_decision = "NONE"
                return {"cycle_id": cycle_id, "status_code": self.last_status_code, "reason": self.last_reason, "analysis_executed": False}

            # Identify latest completed candle
            latest_bar: Candle = candles[-1]
            bar_time_utc = latest_bar.time
            bar_key = f"{self.symbol}_{self.timeframe.value}_{bar_time_utc.strftime('%Y%m%d%H%M')}"

            # 6. Check Bar Deduplication & Atomic Reservation
            existing_record = db.query(WorkerCycleRecord).filter_by(bar_key=bar_key).first()
            rec: Optional[WorkerCycleRecord] = None

            if existing_record:
                if existing_record.lifecycle_phase == "COMPLETED":
                    self.last_status_code = "NO_NEW_CLOSED_BAR"
                    self.last_reason = f"Latest closed bar {bar_key} has already been analyzed in reference cycle {existing_record.cycle_id}."
                    self.last_decision = existing_record.decision or "HOLD"
                    self.last_lifecycle_phase = "DUPLICATE_SKIPPED"
                    self.successful_cycles += 1
                    return {
                        "cycle_id": cycle_id,
                        "bar_key": bar_key,
                        "status_code": self.last_status_code,
                        "lifecycle_phase": "DUPLICATE_SKIPPED",
                        "reference_cycle_id": existing_record.cycle_id,
                        "reference_status": existing_record.status_code,
                        "decision": self.last_decision,
                        "reason": self.last_reason,
                        "bar_time": bar_time_utc.isoformat(),
                        "analysis_executed": False,
                    }
                elif existing_record.lifecycle_phase == "PROCESSING":
                    # Check age of processing record
                    age_proc = (utc_now_naive() - (existing_record.created_at or utc_now_naive())).total_seconds()
                    if age_proc < 120.0:
                        self.last_status_code = "BAR_UNDER_PROCESSING"
                        self.last_reason = f"Bar {bar_key} is currently being processed by cycle {existing_record.cycle_id}."
                        return {
                            "cycle_id": cycle_id,
                            "bar_key": bar_key,
                            "status_code": self.last_status_code,
                            "lifecycle_phase": "PROCESSING",
                            "reference_cycle_id": existing_record.cycle_id,
                            "reference_status": existing_record.status_code,
                            "reason": self.last_reason,
                            "analysis_executed": False,
                        }
                    else:
                        # Orphaned from previous crash/restart -> recover and complete
                        rec = existing_record
                        rec.cycle_id = cycle_id
                        rec.details = "Recovered orphaned processing cycle on restart"
                        db.commit()
                else:
                    rec = existing_record
            else:
                # Atomically reserve candle in database before running heavy pipeline
                from sqlalchemy.exc import IntegrityError
                try:
                    rec = WorkerCycleRecord(
                        cycle_id=cycle_id,
                        symbol=self.symbol,
                        timeframe=self.timeframe.value,
                        bar_key=bar_key,
                        bar_time=bar_time_utc,
                        status_code="PROCESSING",
                        lifecycle_phase="PROCESSING",
                        decision="PENDING",
                        duration_ms=0.0,
                        details="Bar discovered, analysis in progress",
                        created_at=utc_now_naive(),
                        updated_at=utc_now_naive(),
                    )
                    db.add(rec)
                    db.commit()
                    self.last_lifecycle_phase = "PROCESSING"
                except IntegrityError:
                    db.rollback()
                    existing = db.query(WorkerCycleRecord).filter_by(bar_key=bar_key).first()
                    self.last_status_code = "NO_NEW_CLOSED_BAR"
                    self.last_reason = f"Bar {bar_key} claimed concurrently by cycle {existing.cycle_id if existing else 'parallel'}."
                    self.last_decision = existing.decision if existing else "HOLD"
                    self.last_lifecycle_phase = "DUPLICATE_SKIPPED"
                    return {
                        "cycle_id": cycle_id,
                        "bar_key": bar_key,
                        "status_code": self.last_status_code,
                        "lifecycle_phase": "DUPLICATE_SKIPPED",
                        "reference_cycle_id": existing.cycle_id if existing else None,
                        "reference_status": existing.status_code if existing else None,
                        "decision": self.last_decision,
                        "reason": self.last_reason,
                        "analysis_executed": False,
                    }
                except Exception as db_res_err:
                    db.rollback()
                    logger.warning(f"Failed to reserve bar {bar_key} in DB: {db_res_err}")
                    # Fail-closed: continuing here would run the analysis
                    # UNRESERVED, and the finalizer's insert would then collide
                    # with the unique bar_key — guaranteed DATABASE_WRITE_FAILED
                    # churn on every cycle. Skip instead; the same bar is
                    # retried on the next poll once the DB recovers.
                    self.last_status_code = "RESERVATION_FAILED"
                    self.last_reason = f"Bar reservation failed for {bar_key}; cycle skipped: {db_res_err}"
                    self.last_decision = "NONE"
                    self.last_lifecycle_phase = "RESERVATION_FAILED"
                    return {
                        "cycle_id": cycle_id,
                        "bar_key": bar_key,
                        "status_code": self.last_status_code,
                        "lifecycle_phase": self.last_lifecycle_phase,
                        "reason": self.last_reason,
                        "analysis_executed": False,
                    }

            # 7. A NEW CLOSED BAR IS RESERVED! Process through Analysis Pipeline
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
                self.last_lifecycle_phase = "FAILED"
                if rec:
                    try:
                        rec.status_code = self.last_status_code
                        rec.lifecycle_phase = "FAILED"
                        rec.error_message = str(fe_err)
                        rec.details = self.last_reason
                        rec.updated_at = utc_now_naive()
                        db.commit()
                    except Exception:
                        db.rollback()
                return {"cycle_id": cycle_id, "status_code": self.last_status_code, "reason": self.last_reason}

            latest_feat_series = df_feats.iloc[-1]

            # 8. Check Governance: Approved Rules and Active ML Model
            knowledge_mgr = KnowledgeManager(db)
            approved_rules = knowledge_mgr.get_active_strategy_rules()
            model_reg = ModelRegistry(db)
            active_model_artifact = model_reg.load_active_model_artifact()

            # Evaluate rules & model
            rule_dir, rule_str, triggered_rules = SignalGenerator.evaluate_rules(
                latest_feat_series, approved_rules, symbol=self.symbol, timeframe=self.timeframe.value
            )
            model_dir, model_prob = SignalGenerator.evaluate_model(latest_feat_series, active_model_artifact)

            # Determine direction through the same consensus policy as SignalManager.
            final_direction, _ = SignalGenerator.resolve_consensus(
                rule_direction=rule_dir,
                rule_strength=rule_str,
                model_direction=model_dir,
                model_strength=model_prob,
                approved_rule_count=len(approved_rules),
            )
            status_code = "NO_VALID_SIGNAL"
            reason_details = []

            if len(approved_rules) == 0:
                reason_details.append("NO_APPROVED_RULES")
            if active_model_artifact is None:
                reason_details.append("NO_ACTIVE_MODEL")

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

                # Persist TradeSignal record using SignalManager. The worker
                # passes the feature frame it ALREADY computed for the reserved
                # bar: without it the manager re-fetches candles, and a new M1
                # bar closing in between records bar N+1's signal under cycle N
                # (item 40 race).
                sig_mgr = SignalManager(db=db, data_provider=self.data_provider)
                sig_record = sig_mgr.process_latest_bar(
                    symbol=self.symbol, timeframe=self.timeframe,
                    precomputed_features=df_feats,
                )
                if sig_record:
                    signal_id_created = sig_record.signal_id
                    self.last_signal_id = signal_id_created

            # 9. Finalize Persistent Cycle Record in Database
            self.last_status_code = status_code
            self.last_decision = final_direction
            self.last_reason = reason_msg
            self.last_lifecycle_phase = "COMPLETED"
            self.successful_cycles += 1

            # Item 6 / M5: recovery sweep for demo orders whose broker outcome
            # was never confirmed (PENDING stuck by a failed post-fill commit,
            # or EXECUTION_UNKNOWN). Read-only toward the broker (history check
            # only) and never dispatches, so it runs regardless of the demo
            # gate: stale rows from before demo trading was disabled still
            # deserve reconciliation.
            try:
                DemoTradingExecutor(db, self.data_provider).reconcile_stale_pending_orders()
            except Exception as sweep_err:
                logger.warning(f"PENDING reconciliation sweep failed: {sweep_err}")

            dur_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            # rec is guaranteed non-None here: every reservation-failure path
            # returns early, so analysis always finalizes against a reserved row.
            try:
                rec.status_code = status_code
                rec.lifecycle_phase = "COMPLETED"
                rec.decision = final_direction
                rec.signal_id = signal_id_created
                rec.duration_ms = dur_ms
                rec.details = reason_msg
                rec.updated_at = utc_now_naive()
                db.commit()
            except Exception as commit_err:
                db.rollback()
                logger.error(f"Failed to commit completed cycle record: {commit_err}")
                self.last_status_code = "DATABASE_WRITE_FAILED"
                self.last_reason = f"Database commit failed: {commit_err}"

            return {
                "cycle_id": cycle_id,
                "bar_key": bar_key,
                "bar_time": bar_time_utc.isoformat(),
                "status_code": status_code,
                "lifecycle_phase": "COMPLETED",
                "decision": final_direction,
                "signal_id": signal_id_created,
                "reason": reason_msg,
                "duration_ms": dur_ms,
                "analysis_executed": True,
            }

        finally:
            db.close()


# Global worker instance holder
_global_worker: Optional[MarketMonitoringWorker] = None


def get_market_worker() -> MarketMonitoringWorker:
    """Returns the singleton MarketMonitoringWorker instance."""
    global _global_worker
    if _global_worker is None:
        _global_worker = MarketMonitoringWorker()
    return _global_worker


def prune_operational_records(
    db,
    worker_cycle_days: int = 0,
    audit_log_days: int = 0,
    trade_signal_days: int = 0,
    demo_order_days: int = 0,
) -> Dict[str, int]:
    """Configurable retention for operational tables (item 40).

    Deletes worker_cycle_records / audit_logs / trade_signals / demo_orders
    rows older than the respective window (in days). A window of 0 (the
    default) DISABLES pruning for that table — nothing is ever deleted without
    the operator explicitly setting a retention window in settings. Returns
    the per-table delete counts.
    """
    deleted = {"worker_cycle_records": 0, "audit_logs": 0, "trade_signals": 0, "demo_orders": 0}
    now = utc_now_naive()
    if worker_cycle_days and worker_cycle_days > 0:
        cutoff = now - timedelta(days=worker_cycle_days)
        removed = (
            db.query(WorkerCycleRecord)
            .filter(WorkerCycleRecord.created_at < cutoff)
            .delete(synchronize_session=False)
        )
        deleted["worker_cycle_records"] = int(removed)
    if audit_log_days and audit_log_days > 0:
        cutoff = now - timedelta(days=audit_log_days)
        removed = (
            db.query(AuditLog)
            .filter(AuditLog.created_at < cutoff)
            .delete(synchronize_session=False)
        )
        deleted["audit_logs"] = int(removed)
    if trade_signal_days and trade_signal_days > 0:
        cutoff = now - timedelta(days=trade_signal_days)
        removed = (
            db.query(TradeSignal)
            .filter(TradeSignal.created_at < cutoff)
            .delete(synchronize_session=False)
        )
        deleted["trade_signals"] = int(removed)
    if demo_order_days and demo_order_days > 0:
        from backend.app.models.order import DemoOrder

        cutoff = now - timedelta(days=demo_order_days)
        removed = (
            db.query(DemoOrder)
            .filter(DemoOrder.created_at < cutoff)
            .delete(synchronize_session=False)
        )
        deleted["demo_orders"] = int(removed)
    if any(deleted.values()):
        db.commit()
        logger.info(
            "Operational retention applied: %d worker_cycle_records, %d audit_logs, "
            "%d trade_signals, %d demo_orders pruned.",
            deleted["worker_cycle_records"], deleted["audit_logs"],
            deleted["trade_signals"], deleted["demo_orders"],
        )
    return deleted
