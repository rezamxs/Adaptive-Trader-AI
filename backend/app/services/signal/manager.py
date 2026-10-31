"""Signal Lifecycle Manager with strict idempotency and audit trail."""
import uuid
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from backend.app.core.config import settings, ExecutionMode
from backend.app.core.logging import logger, log_security_event
from backend.app.models.signal import TradeSignal
from backend.app.models.audit_log import AuditLog
from backend.app.services.market_data.base import IMarketDataProvider, TimeFrame
from backend.app.services.knowledge.manager import KnowledgeManager
from backend.app.services.ml.registry import ModelRegistry
from backend.app.services.dataset.features import FeatureEngineer
from backend.app.services.risk.risk_engine import RiskEngine
from backend.app.services.signal.generator import SignalGenerator


class SignalManager:
    """Orchestrates signal generation, idempotency checks, risk assessment, and DB persistence."""

    def __init__(self, db: Session, data_provider: IMarketDataProvider):
        self.db = db
        self.data_provider = data_provider
        self.knowledge_mgr = KnowledgeManager(db)
        self.model_registry = ModelRegistry(db)
        self.risk_engine = RiskEngine(
            db, margin_estimator=getattr(data_provider, "calculate_order_margin", None)
        )

    def process_latest_bar(
        self,
        symbol: str = "EURUSD",
        timeframe: TimeFrame = TimeFrame.M15,
        tp_atr_mult: float = 1.5,
        sl_atr_mult: float = 1.0,
        precomputed_features: Optional[pd.DataFrame] = None,
    ) -> Optional[TradeSignal]:
        """Analyzes latest completed bar, runs rules and model, assesses risk, and records signal.

        precomputed_features: the caller (worker) may pass the feature frame it
        ALREADY computed for the reserved bar. Without it this method re-fetches
        candles — and if a new M1 bar closes between the worker's fetch and this
        refetch, the signal would be recorded against bar N+1 under cycle N
        (item 40 race). Passing the precomputed frame pins the signal to the
        reserved bar.
        """
        # 1. Fetch historical candles and compute features (or reuse the
        # caller's precomputed frame for the reserved bar).
        if precomputed_features is not None and len(precomputed_features) >= 50:
            df_feats = precomputed_features
        else:
            df_raw = self.data_provider.get_historical_dataframe(symbol, timeframe, count=100)
            if len(df_raw) < 50:
                logger.warning(f"Insufficient bars ({len(df_raw)}) to generate signal for {symbol}.")
                return None
            df_feats, _ = FeatureEngineer.transform(df_raw)
        latest_bar = df_feats.iloc[-1]
        bar_time = df_feats.index[-1]
        if isinstance(bar_time, pd.Timestamp):
            bar_time_dt = bar_time.to_pydatetime()
        else:
            bar_time_dt = bar_time

        # 2. Get approved rules & active ML model
        approved_rules = self.knowledge_mgr.get_active_strategy_rules()
        active_model_artifact = self.model_registry.load_active_model_artifact()

        # 3. Evaluate Rule logic
        rule_dir, rule_str, triggered_rules = SignalGenerator.evaluate_rules(
            latest_bar, approved_rules, symbol=symbol, timeframe=timeframe.value
        )

        # 4. Evaluate Model logic
        model_dir, model_prob = SignalGenerator.evaluate_model(latest_bar, active_model_artifact)

        # Resolve consensus consistently with the BUY-only training target.
        final_direction, confidence = SignalGenerator.resolve_consensus(
            rule_direction=rule_dir,
            rule_strength=rule_str,
            model_direction=model_dir,
            model_strength=model_prob,
            approved_rule_count=len(approved_rules),
        )

        if final_direction == "HOLD":
            return None

        # 5. Idempotency Check
        idempotency_key = f"{symbol}_{timeframe.value}_{bar_time_dt.strftime('%Y%m%d%H%M')}"
        # Older records included direction in the key; check both formats during transition.
        existing = self.db.query(TradeSignal).filter(
            TradeSignal.idempotency_key.in_(
                (idempotency_key, f"{idempotency_key}_BUY", f"{idempotency_key}_SELL")
            )
        ).first()
        if existing:
            logger.info(f"Signal with key {idempotency_key} already recorded. Skipping duplicate.")
            return existing

        # 6. Query current market context before setting executable entry and barriers.
        account = self.data_provider.get_account_info()
        symbol_info = self.data_provider.get_symbol_info(symbol)
        quote = self.data_provider.get_current_quote(symbol)

        # ATR must be valid even for a rejected signal; otherwise there is no
        # meaningful basis for its stop-loss and take-profit levels.
        try:
            atr = float(latest_bar.get("atr_14", float("nan")))
            bar_close = float(latest_bar["close"])
        except (TypeError, ValueError, OverflowError, KeyError):
            logger.warning(f"Missing numeric price/ATR for {symbol}; signal not persisted.")
            return None
        if not math.isfinite(atr) or atr <= 0 or not math.isfinite(bar_close) or bar_close <= 0:
            logger.warning(f"Invalid price/ATR for {symbol}; signal not persisted.")
            return None

        # A signal based on a closed bar still needs a current executable price.
        # Use Ask for BUY and Bid for SELL when a usable live quote exists.
        quote_is_usable = False
        quote_bid = quote_ask = float("nan")
        if quote is not None:
            try:
                quote_bid = float(quote.bid)
                quote_ask = float(quote.ask)
                quote_is_usable = (
                    not quote.is_stale
                    and str(quote.symbol).upper() == symbol.upper()
                    and isinstance(quote.time, datetime)
                    and quote.time.tzinfo is not None
                    and quote.time.utcoffset() is not None
                    and math.isfinite(quote_bid)
                    and math.isfinite(quote_ask)
                    and 0 < quote_bid <= quote_ask
                )
            except (TypeError, ValueError, OverflowError):
                quote_is_usable = False

        entry_price = (
            quote_ask if final_direction == "BUY" else quote_bid
        ) if quote_is_usable else bar_close
        if final_direction == "BUY":
            stop_loss = entry_price - (sl_atr_mult * atr)
            take_profit = entry_price + (tp_atr_mult * atr)
        else:
            stop_loss = entry_price + (sl_atr_mult * atr)
            take_profit = entry_price - (tp_atr_mult * atr)

        is_approved, calculated_lots, risk_reason = False, 0.0, "Risk evaluation unfulfilled"
        if not quote_is_usable:
            risk_reason = "REJECTED_RISK_CONTEXT_UNAVAILABLE: A current, valid quote is required."

        open_trades_count: Optional[int] = None
        daily_loss_percent: Optional[float] = None
        try:
            positions_reader = getattr(self.data_provider, "get_open_positions_count", None)
            loss_reader = getattr(self.data_provider, "get_daily_loss_percent", None)
            if callable(positions_reader):
                open_trades_count = positions_reader()
            if callable(loss_reader):
                daily_loss_percent = loss_reader()
        except Exception as context_error:
            logger.warning(f"Unable to verify account risk context for {symbol}: {context_error}")
            open_trades_count = None
            daily_loss_percent = None

        # Strategy Rules (Pages 14-15): the session window is enforced on
        # every production risk assessment. The martingale ceiling uses the
        # floating loss of OUR demo-grid positions only (comment-prefixed),
        # computed from live broker positions. Unverifiable position data is
        # fail-closed (REJECTED), never replaced by a zero default (item 4).
        try:
            positions_reader = getattr(self.data_provider, "get_open_positions", None)
            open_positions = positions_reader() if callable(positions_reader) else None
        except Exception as positions_error:
            logger.warning(f"Unable to read open positions for {symbol}: {positions_error}")
            open_positions = None
        grid_drawdown = RiskEngine.compute_grid_martingale_drawdown(open_positions)

        if not (account and symbol_info and quote and quote_is_usable):
            risk_reason = "REJECTED_RISK_CONTEXT_UNAVAILABLE: Account, symbol specification, or quote is unavailable."
        elif grid_drawdown is None:
            risk_reason = "REJECTED_MARTINGALE_DATA_UNAVAILABLE: Open-position data could not be verified."
        else:
            is_approved, calculated_lots, risk_reason = self.risk_engine.assess_signal(
                symbol=symbol,
                direction=final_direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                account=account,
                symbol_info=symbol_info,
                quote=quote,
                open_trades_count=open_trades_count,
                daily_loss_percent=daily_loss_percent,
                current_time_utc=datetime.now(timezone.utc),
                active_martingale_drawdown=grid_drawdown,
                enforce_session_filter=settings.ENFORCE_SESSION_FILTER,
            )

        # 8. Create and persist TradeSignal
        signal_id = f"sig_{uuid.uuid4().hex[:10]}"
        current_mode = settings.EXECUTION_MODE.value

        signal_record = TradeSignal(
            signal_id=signal_id,
            idempotency_key=idempotency_key,
            symbol=symbol,
            timeframe=timeframe.value,
            direction=final_direction,
            strength=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            rules_triggered=triggered_rules,
            model_version_id=active_model_artifact.get("model_id") if active_model_artifact else None,
            features_snapshot={"rsi_14": float(latest_bar.get("rsi_14", 50.0)), "atr_14": atr},
            risk_status="APPROVED" if is_approved else "REJECTED",
            risk_rejection_reason=risk_reason if not is_approved else None,
            calculated_volume=calculated_lots if is_approved else None,
            execution_mode=current_mode,
            status="GENERATED",
            bar_time=bar_time_dt,
        )
        self.db.add(signal_record)
        try:
            self.db.commit()
        except IntegrityError:
            # Lost the idempotency race against a concurrent worker/API insert
            # on the same symbol/timeframe/bar. Surface the existing record
            # instead of a 500 / WORKER_EXCEPTION.
            self.db.rollback()
            winner = self.db.query(TradeSignal).filter(
                TradeSignal.idempotency_key.in_(
                    (idempotency_key, f"{idempotency_key}_BUY", f"{idempotency_key}_SELL")
                )
            ).first()
            if winner is not None:
                logger.info(
                    f"Concurrent signal for key {idempotency_key} already recorded "
                    f"as {winner.signal_id}; returning the existing record."
                )
                return winner
            raise
        self.db.refresh(signal_record)

        log_security_event(
            "SIGNAL_GENERATED",
            {
                "signal_id": signal_id,
                "symbol": symbol,
                "direction": final_direction,
                "mode": current_mode,
                "risk_status": signal_record.risk_status,
            },
            "INFO"
        )
        logger.info(f"Generated signal {signal_id} ({final_direction} on {symbol}) under mode {current_mode}.")
        return signal_record
