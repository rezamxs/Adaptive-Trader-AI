"""Restricted Demo Trading Execution Engine.
Strictly isolated to Demo accounts with fail-closed safety assertions.
NO LIVE TRADING LOGIC EXISTS IN THIS MODULE OR APPLICATION.
"""
import uuid
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

from backend.app.core.config import settings, ExecutionMode
from backend.app.core.logging import logger, log_security_event
from backend.app.core.security import (
    assert_safe_to_execute_demo_order,
    verify_account_is_demo,
    MT5_ACCOUNT_TRADE_MODE_DEMO,
)
from backend.app.core.errors import (
    SecurityViolationError,
    AccountAuthenticationError,
    EmergencyStopActiveError,
)
from backend.app.models.signal import TradeSignal
from backend.app.models.order import DemoOrder
from backend.app.models.audit_log import AuditLog
from backend.app.db.base import utc_now_naive
from backend.app.services.market_data.base import IMarketDataProvider
from backend.app.services.risk.circuit_breaker import EmergencyStopManager
from backend.app.services.risk.risk_engine import RiskEngine

# Transport-level retcodes whose broker outcome cannot be known locally: the
# request may or may not have reached the trade server. Definitive refusals
# (invalid price, no money, market closed, ...) stay REJECTED.
_AMBIGUOUS_RETCODES = frozenset(
    code for code in (
        getattr(mt5, "TRADE_RETCODE_TIMEOUT", None),
        getattr(mt5, "TRADE_RETCODE_ERROR", None),
        getattr(mt5, "TRADE_RETCODE_CONNECTION", None),
        getattr(mt5, "TRADE_RETCODE_NO_MEMORY", None),
        # 10008 "order placed": the broker ACCEPTED the request (it applies to
        # DEAL orders on some servers when execution is not instantaneous).
        # Marking it REJECTED would abandon a possibly-executing position while
        # the idempotency key stays reserved; reconciliation is the only safe
        # resolution, same as the transport-ambiguous codes above.
        getattr(mt5, "TRADE_RETCODE_PLACED", None),
    ) if code is not None
)

# M5: retcodes that CONFIRM execution. DONE_PARTIAL (10010) means part of the
# request executed (IOC); the executed volume is on result.volume. Treating it
# as a rejection would abandon a live partial position at the broker.
_CONFIRMED_FILL_RETCODES = frozenset(
    code for code in (
        getattr(mt5, "TRADE_RETCODE_DONE", None),
        getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", None),
    ) if code is not None
)


# M22: canonical failure-message fragments. signals.py maps HTTP statuses
# from THESE constants (single source of truth) — rewording a message here
# cannot silently break the API failure mapping. Volume/record-validity
# rejections intentionally have no fragment (they map to 400 by default).
EXEC_MSG_DUPLICATE = "Duplicate order prevented"
EXEC_MSG_UNRESOLVED = "UNKNOWN"  # EXECUTION_UNKNOWN reconciliation required
EXEC_MSG_SAFETY_GATE = "Safety Gate Blocked"
EXEC_MSG_RISK_CHECKS = "Execution rejected by current risk checks"
EXEC_MSG_RISK_STATUS = "Signal risk status is"
EXEC_MSG_STALE_QUOTE = "Market quote is stale or unavailable"
EXEC_MSG_NO_SPECS = "broker symbol specifications are unavailable"


class DemoTradingExecutor:
    """Isolated, fail-closed executor for demo trading orders only."""

    def __init__(self, db: Session, data_provider: IMarketDataProvider):
        self.db = db
        self.data_provider = data_provider
        self.emergency_stop = EmergencyStopManager(db)

    def reconcile_stale_pending_orders(self, max_age_minutes: float = 30.0) -> int:
        """Recovery sweep for orders whose broker outcome was never confirmed.

        Re-checks broker history for every PENDING (stuck post-fill commit) or
        EXECUTION_UNKNOWN (dispatch outcome unresolved) order older than the
        threshold and resolves confirmed fills to FILLED (deals-first evidence
        ladder, see _reconcile_unknown_order). Read-only toward the broker
        (history_orders_get / history_deals_get only); returns the number of
        orders resolved to FILLED.
        """
        cutoff = utc_now_naive() - timedelta(minutes=max_age_minutes)
        stale_orders = self.db.query(DemoOrder).filter(
            DemoOrder.status.in_(("PENDING", "EXECUTION_UNKNOWN")),
            DemoOrder.created_at < cutoff,
        ).all()
        reconciled = 0
        for order in stale_orders:
            # Per-order isolation: a failed commit inside the reconciliation
            # leaves the session pending-rollback; without a rollback here the
            # next refresh/commit raises PendingRollbackError and aborts the
            # whole sweep (and the worker cycle) on the FIRST bad order.
            try:
                self._reconcile_unknown_order(order)
                self.db.refresh(order)
            except Exception as sweep_error:
                logger.warning(
                    f"Reconciliation sweep for order {order.order_id} failed: "
                    f"{sweep_error}; rolling back and continuing with remaining orders."
                )
                self.db.rollback()
                try:
                    self.db.refresh(order)
                except Exception:
                    pass
                continue
            if order.status == "FILLED":
                reconciled += 1
        return reconciled

    def _reconcile_unknown_order(self, order_rec: DemoOrder) -> None:
        """Best-effort broker history check for an outcome-unknown order (M5).

        Evidence ladder, most reliable source first:
        1. Deals — history_deals_get(position=<order record's position_id>)
           returns the immutable execution records of the position our order
           belongs to. The executed volume of THIS order is the sum of
           DEAL_ENTRY_IN deal volumes whose `order` field equals our ticket
           (exit deals and other orders' deals on the same position must not
           be summed). Any executed volume > 0 is DEFINITIVE: FILLED.
        2. Order row — history_orders_get(date range) TradeOrder fields, with
           mql5 semantics: volume_current is the volume REMAINING (unfilled),
           so executed = volume_initial - volume_current. Used when the deals
           call is unavailable. Terminal-negative states (CANCELED/EXPIRED/
           REJECTED) with zero executed volume resolve to CANCELLED/REJECTED.
        3. Nothing found / terminal unreachable / API error → the row keeps
           EXECUTION_UNKNOWN (or PENDING) for a later sweep — never guessed.

        MT5 exposes terminal history only when the terminal is reachable, so
        any failure here leaves the row for operator review. The request
        carried the order_id in its comment, which is what the history lookup
        matches against. Repeated sweeps are idempotent: terminal rows leave
        the sweep filter and deals are immutable broker records.
        """
        if not MT5_AVAILABLE or mt5 is None:
            return
        try:
            from_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
            to_date = datetime.now(timezone.utc) + timedelta(days=1)
            # H-1 (2026-10-31 audit): every MetaTrader5 C-extension call must
            # hold the process-global _mt5_lock — the reconciliation sweep runs
            # on its own worker thread while API threads call order_send and
            # the provider's locked read methods. The MetaTrader5 package is
            # not thread-safe; interleaved IPC can corrupt responses (a
            # transient None here would wrongly park the order in
            # EXECUTION_UNKNOWN and reserve its idempotency key).
            from backend.app.services.market_data.mt5_provider import _mt5_lock
            with _mt5_lock:
                history = mt5.history_orders_get(from_date, to_date)
            # Documented date-range return: a tuple of TradeOrder objects
            # directly; None on error or when nothing is found. The old
            # history[0] unwrap grabbed the first order instead of the
            # sequence and silently never matched.
            if history is None:
                return
            entries = history
            if not entries:
                return
            for entry in entries:
                comment = str(getattr(entry, "comment", "") or "")
                if f"Demo AI: {order_rec.order_id}" not in comment:
                    continue
                state = getattr(entry, "state", None)
                ticket = getattr(entry, "ticket", None)
                # The POSITION identifier is read from the ORDER RECORD itself
                # (TradeOrder.position_id), never assumed to equal the order
                # ticket: on netting accounts an add-on order attaches to an
                # EXISTING position whose id is the original opening order's
                # ticket, so passing our own ticket as the position id would
                # silently query the wrong position. A zero/absent position_id
                # means the order is unrelated to any position — deals evidence
                # is then unavailable and the order-row fallback decides.
                try:
                    position_id = int(getattr(entry, "position_id", 0) or 0)
                except (TypeError, ValueError):
                    position_id = 0

                # --- Evidence 1: deals (authoritative executed volume) ---
                executed_volume = None  # None = deals evidence unavailable
                deals_evidence = False  # True only when deals records were read
                cost_weighted_price = 0.0  # Σ(volume × price) over ENTRY_IN deals
                deal_commission = 0.0
                deal_swap = 0.0
                try:
                    if ticket and position_id > 0:
                        # Official MT5 semantics: `position=` filters deals by
                        # DEAL_POSITION_ID, `ticket=` filters by DEAL_ORDER.
                        # We query by the order record's position_id and then
                        # re-filter deals by the EXACT order identity
                        # (deal.order == ticket) so only THIS order's
                        # DEAL_ENTRY_IN deals are summed — partial fills of our
                        # order aggregate, exit deals and other orders' deals
                        # on the same position are excluded.
                        with _mt5_lock:
                            deals = mt5.history_deals_get(position=position_id)
                        if deals is not None:
                            deals_evidence = True
                            executed_volume = 0.0
                            entry_in = int(getattr(mt5, "DEAL_ENTRY_IN", 0))
                            for deal in deals:
                                # NOTE: 0 is a valid field value (DEAL_ENTRY_IN
                                # == 0) — never coerce fields with `or`, which
                                # would discard every legitimate entry deal.
                                try:
                                    deal_order = int(getattr(deal, "order", 0))
                                except (TypeError, ValueError):
                                    continue
                                if deal_order != int(ticket):
                                    continue
                                try:
                                    deal_entry = int(getattr(deal, "entry", -1))
                                except (TypeError, ValueError):
                                    continue
                                if deal_entry != entry_in:
                                    continue
                                try:
                                    deal_volume = float(getattr(deal, "volume", 0.0) or 0.0)
                                    deal_price = float(getattr(deal, "price", 0.0) or 0.0)
                                    executed_volume += deal_volume
                                    cost_weighted_price += deal_volume * deal_price
                                    deal_commission += float(getattr(deal, "commission", 0.0) or 0.0)
                                    deal_swap += float(getattr(deal, "swap", 0.0) or 0.0)
                                except (TypeError, ValueError):
                                    continue
                except Exception as deals_error:
                    logger.warning(
                        f"history_deals_get for order {order_rec.order_id} failed: {deals_error}")
                    executed_volume = None

                # --- Evidence 2: order-row arithmetic (fallback) ---
                if executed_volume is None:
                    try:
                        requested = float(getattr(entry, "volume_initial", 0.0) or 0.0)
                        remaining = float(getattr(entry, "volume_current", 0.0) or 0.0)
                        executed_volume = max(0.0, requested - remaining)
                    except (TypeError, ValueError):
                        executed_volume = None  # stays None: no usable evidence

                # --- Decision (None evidence ≠ zero execution) ---
                # Terminal-negative states: the broker will never fill this
                # order anymore. STARTED/PLACED are NOT terminal — an order
                # still working with zero fills stays unresolved.
                terminal_no_fill_states = {
                    getattr(mt5, "ORDER_STATE_CANCELED", 2),
                    getattr(mt5, "ORDER_STATE_REJECTED", 5),
                    getattr(mt5, "ORDER_STATE_EXPIRED", 6),
                }
                rejected_state = getattr(mt5, "ORDER_STATE_REJECTED", 5)
                if executed_volume is None:
                    # Neither deals nor order-row arithmetic produced usable
                    # evidence: keep the row unresolved — never guess.
                    marker = f"broker history state={state!r}"
                    if marker not in (order_rec.error_message or ""):
                        order_rec.error_message = (
                            f"{order_rec.error_message or ''} | {marker}"
                        )
                        self.db.commit()
                elif executed_volume > 0.0:
                    # A recorded deal (or order arithmetic) proves real
                    # execution — record the ACTUAL executed volume, never
                    # the requested size. Deal-derived costs (VWAP fill
                    # price, commission, swap) are only recorded when the
                    # evidence came from deals; the order-row fallback has
                    # no price/cost fields and must not fabricate zeros.
                    order_rec.mt5_ticket = ticket or order_rec.mt5_ticket
                    order_rec.volume = executed_volume
                    if deals_evidence:
                        order_rec.executed_price = cost_weighted_price / executed_volume
                        order_rec.commission = deal_commission
                        order_rec.swap = deal_swap
                    order_rec.status = "FILLED"
                    order_rec.executed_at = utc_now_naive()
                    self.db.commit()
                    self.db.refresh(order_rec)
                    log_security_event(
                        "DEMO_ORDER_RECONCILED_FILLED",
                        {"order_id": order_rec.order_id, "mt5_ticket": order_rec.mt5_ticket},
                        "INFO",
                    )
                elif state in terminal_no_fill_states:
                    # Definitive no-execution: the broker canceled/expired/
                    # refused the order without any fill.
                    order_rec.mt5_ticket = ticket or order_rec.mt5_ticket
                    order_rec.status = (
                        "REJECTED" if state == rejected_state else "CANCELLED")
                    self.db.commit()
                    self.db.refresh(order_rec)
                else:
                    # Not terminal (STARTED/PLACED/PARTIAL with no fill yet):
                    # leave the row for a later sweep. Record the state once —
                    # appending on every sweep pass would grow the row
                    # unboundedly.
                    marker = f"broker history state={state!r}"
                    if marker not in (order_rec.error_message or ""):
                        order_rec.error_message = (
                            f"{order_rec.error_message or ''} | {marker}"
                        )
                        self.db.commit()
                return
        except Exception as reconcile_error:
            # A failed commit (e.g. SQLite lock) leaves the session pending-
            # rollback; release it so the caller's session stays usable.
            self.db.rollback()
            logger.warning(f"Reconciliation for order {order_rec.order_id} failed: {reconcile_error}")

    def _mark_unknown_outcome(self, order_rec: DemoOrder, err_msg: str, detail: str = ""):
        """Shared UNKNOWN-outcome handler for raised exceptions, None results,
        and transport-ambiguous retcodes: the idempotency key stays reserved,
        the row is reconciled against broker history, and the operator gets a
        CRITICAL audit event."""
        logger.error(err_msg)
        order_rec.status = "EXECUTION_UNKNOWN"
        order_rec.error_message = err_msg
        self.db.commit()
        self._reconcile_unknown_order(order_rec)
        log_security_event(
            "DEMO_ORDER_OUTCOME_UNKNOWN",
            {
                "order_id": order_rec.order_id,
                "signal_id": order_rec.signal_id,
                "error": detail or err_msg,
            },
            "CRITICAL",
        )
        return False, order_rec, (
            f"Order dispatch outcome is {EXEC_MSG_UNRESOLVED} ({err_msg}). The idempotency key "
            "is reserved; reconcile broker history before any retry."
        )

    def execute_demo_signal(self, signal: TradeSignal) -> Tuple[bool, Optional[DemoOrder], str]:
        """Pre-flight verification and dispatch of an approved signal to DEMO account only.

        Returns:
            (success, demo_order_instance, message)
        """
        # 1. Check Signal Approval
        if signal.risk_status != "APPROVED":
            return False, None, f"Execution rejected: {EXEC_MSG_RISK_STATUS} '{signal.risk_status}'."

        if not signal.calculated_volume or signal.calculated_volume <= 0:
            return False, None, "Execution rejected: Calculated volume is missing or non-positive."

        # 2. Check Idempotency (prevent duplicate order)
        existing_order = self.db.query(DemoOrder).filter_by(idempotency_key=signal.idempotency_key).first()
        if existing_order:
            if existing_order.status == "EXECUTION_UNKNOWN":
                # Outcome at the broker was never confirmed: a second dispatch
                # could double the position. Only reconciliation may resolve it.
                return False, existing_order, (
                    f"Previous dispatch outcome is {EXEC_MSG_UNRESOLVED}; broker reconciliation required "
                    f"before retry (idempotency key {signal.idempotency_key})."
                )
            return False, existing_order, f"{EXEC_MSG_DUPLICATE} by idempotency key {signal.idempotency_key}."

        # Finding 6: a concurrent request can slip past the check above and win
        # the unique-key commit first; the loser's PENDING insert must surface
        # as the same duplicate-prevention message instead of an IntegrityError.
        def _commit_pending_safely(record: DemoOrder) -> bool:
            try:
                self.db.add(record)
                self.db.commit()
                return True
            except IntegrityError:
                self.db.rollback()
                return False

        # 3. Read Terminal Account Information
        account_details = self.data_provider.get_account_info()
        is_connected = self.data_provider.is_connected()
        is_demo = account_details.is_demo if account_details else False

        # 4. Strict Safety Assertions (Fails Closed)
        try:
            assert_safe_to_execute_demo_order(
                current_mode=settings.EXECUTION_MODE,
                is_demo_enabled=settings.DEMO_TRADING_ENABLED,
                is_emergency_stop_active=self.emergency_stop.is_active(),
                is_connected=is_connected,
                account_is_demo=is_demo,
            )
        except Exception as e:
            log_security_event("DEMO_ORDER_BLOCKED", {"signal_id": signal.signal_id, "error": str(e)}, "WARNING")
            return False, None, f"{EXEC_MSG_SAFETY_GATE}: {str(e)}"

        # 5. Revalidate the original signal against the live account and quote.
        # Approval at generation time is not enough: price, margin, and drawdown may change.
        quote = self.data_provider.get_current_quote(signal.symbol)
        symbol_info = self.data_provider.get_symbol_info(signal.symbol)
        if not quote or quote.is_stale:
            return False, None, EXEC_MSG_STALE_QUOTE + " at moment of order dispatch."
        if not symbol_info:
            return False, None, "Execution rejected: " + EXEC_MSG_NO_SPECS + "."

        req_price = quote.ask if signal.direction == "BUY" else quote.bid
        try:
            positions_reader = getattr(self.data_provider, "get_open_positions_count", None)
            loss_reader = getattr(self.data_provider, "get_daily_loss_percent", None)
            open_trades_count = positions_reader() if callable(positions_reader) else None
            daily_loss_percent = loss_reader() if callable(loss_reader) else None
        except Exception as context_error:
            logger.warning(f"Could not refresh execution risk context: {context_error}")
            open_trades_count = None
            daily_loss_percent = None

        risk_engine = RiskEngine(
            self.db,
            margin_estimator=getattr(self.data_provider, "calculate_order_margin", None),
        )
        # Item 4: martingale ceiling = floating loss of OUR demo-grid
        # positions (comment-prefixed) from live broker data. Unverifiable
        # position data fails closed instead of defaulting to zero.
        try:
            positions_reader = getattr(self.data_provider, "get_open_positions", None)
            open_positions = positions_reader() if callable(positions_reader) else None
        except Exception as positions_error:
            logger.warning(f"Could not read open positions for grid drawdown: {positions_error}")
            open_positions = None
        grid_drawdown = RiskEngine.compute_grid_martingale_drawdown(open_positions)
        if grid_drawdown is None:
            return False, None, (
                f"{EXEC_MSG_RISK_CHECKS}: "
                "REJECTED_MARTINGALE_DATA_UNAVAILABLE: Open-position data could not be verified."
            )

        risk_ok, current_safe_lots, risk_reason = risk_engine.assess_signal(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=req_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            account=account_details,
            symbol_info=symbol_info,
            quote=quote,
            open_trades_count=open_trades_count,
            daily_loss_percent=daily_loss_percent,
            current_time_utc=datetime.now(timezone.utc),
            active_martingale_drawdown=grid_drawdown,
            enforce_session_filter=settings.ENFORCE_SESSION_FILTER,
        )
        if not risk_ok:
            return False, None, f"{EXEC_MSG_RISK_CHECKS}: {risk_reason}"

        try:
            execution_volume = float(signal.calculated_volume)
            symbol_point = float(symbol_info.point)
            volume_step = float(symbol_info.volume_step)
            volume_min = float(symbol_info.volume_min)
            volume_max = float(symbol_info.volume_max)
        except (TypeError, ValueError, OverflowError):
            return False, None, "Execution rejected: stored volume or broker volume limits are invalid."
        if (
            not all(math.isfinite(value) for value in (execution_volume, symbol_point, volume_step, volume_min, volume_max))
            or symbol_point <= 0 or volume_step <= 0 or volume_min <= 0 or volume_max < volume_min
            or execution_volume < volume_min or execution_volume > volume_max
            or not math.isclose(execution_volume / volume_step, round(execution_volume / volume_step), abs_tol=1e-8)
        ):
            return False, None, "Execution rejected: stored volume no longer matches broker volume limits."
        if execution_volume > current_safe_lots + 1e-9:
            return False, None, (
                f"Execution rejected: current risk permits at most {current_safe_lots:.2f} lots; "
                "generate a new signal before retrying."
            )

        # 6. Build Order Request using the stored, revalidated volume.
        order_id = f"ord_{uuid.uuid4().hex[:10]}"

        # Check terminal algorithmic trading flag if using real MT5 provider
        from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider, _mt5_lock
        if isinstance(self.data_provider, MT5MarketDataProvider) and MT5_AVAILABLE and is_connected:
            with _mt5_lock:
                term = mt5.terminal_info()
            if not getattr(term, "trade_allowed", False):
                return False, None, "MT5 terminal global 'Algo Trading' button is disabled. Turn it ON in MT5 to allow automated demo orders."

            # Construct MT5 order request
            mt5_action = mt5.TRADE_ACTION_DEAL
            mt5_type = mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL

            request = {
                "action": mt5_action,
                "symbol": signal.symbol,
                "volume": execution_volume,
                "type": mt5_type,
                "price": req_price,
                "sl": float(signal.stop_loss),
                "tp": float(signal.take_profit),
                "deviation": int(settings.MAX_SLIPPAGE_POINTS),
                "magic": 20261010,
                "comment": f"Demo AI: {order_id}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # H6: commit the PENDING order record BEFORE touching the broker.
            # A crash between order_send and persistence must never orphan a live
            # fill, and the idempotency key is reserved before dispatch.
            order_rec = DemoOrder(
                order_id=order_id,
                signal_id=signal.signal_id,
                idempotency_key=signal.idempotency_key,
                symbol=signal.symbol,
                order_type=signal.direction,
                volume=execution_volume,
                requested_price=req_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                status="PENDING",
            )
            if not _commit_pending_safely(order_rec):
                winner = self.db.query(DemoOrder).filter_by(
                    idempotency_key=signal.idempotency_key).first()
                return False, winner, (
                    f"{EXEC_MSG_DUPLICATE} by idempotency key {signal.idempotency_key}."
                )

            # Final-safety fix 2: an order_send exception means the outcome at
            # the broker is UNKNOWN — the request may have reached and executed.
            # Never mark that REJECTED and never release the idempotency key.
            # H-1: the dispatch itself also holds the process-global MT5 lock
            # (re-entrant; terminal_info above already acquired and released).
            try:
                with _mt5_lock:
                    result = mt5.order_send(request)
            except Exception as send_error:
                err_msg = (
                    f"MT5 order_send raised {type(send_error).__name__}: {send_error}; "
                    "broker outcome unknown, manual reconciliation required."
                )
                return self._mark_unknown_outcome(order_rec, err_msg, detail=str(send_error))

            if result is None:
                err_msg = (
                    "MT5 order_send returned None (transport-level failure); the broker "
                    "outcome is UNKNOWN and reconciliation is required."
                )
                return self._mark_unknown_outcome(order_rec, err_msg)
            if result.retcode in _AMBIGUOUS_RETCODES:
                err_msg = (
                    f"MT5 retcode {result.retcode} is transport-ambiguous "
                    f"(comment={getattr(result, 'comment', '')!r}); the broker outcome "
                    "is UNKNOWN and reconciliation is required."
                )
                return self._mark_unknown_outcome(order_rec, err_msg)
            if result.retcode not in _CONFIRMED_FILL_RETCODES:
                err_code = getattr(result, "retcode", "N/A")
                err_comment = getattr(result, "comment", "Unknown MT5 error")
                err_msg = f"MT5 order rejected: Retcode={err_code}, Comment={err_comment}"
                logger.error(err_msg)

                order_rec.status = "REJECTED"
                order_rec.error_message = err_msg
                self.db.commit()
                return False, order_rec, err_msg

            # Confirmed fill (DONE or DONE_PARTIAL): result.volume is the
            # broker-confirmed executed volume, which for IOC partial fills is
            # less than the requested size.
            order_rec.mt5_ticket = result.order
            order_rec.volume = result.volume
            order_rec.executed_price = result.price
            order_rec.slippage_points = round(abs(result.price - req_price) / symbol_point, 1)
            order_rec.status = "FILLED"
            order_rec.executed_at = utc_now_naive()
            try:
                self.db.commit()
            except Exception as persist_error:
                # The fill is already live at the broker; a persistence outage must
                # not surface as an unhandled error. Leave the row PENDING so the
                # reconciliation path can recover it. Finding 11: roll the in-memory
                # object back to the store's state so the response cannot claim
                # FILLED while the persisted row says PENDING.
                self.db.rollback()
                log_security_event(
                    "DEMO_ORDER_PERSIST_FAILED",
                    {"order_id": order_id, "mt5_ticket": result.order, "error": str(persist_error)},
                    "CRITICAL",
                )
                stored = self.db.query(DemoOrder).filter_by(order_id=order_id).first()
                return True, stored or order_rec, (
                    f"Demo order filled (Ticket: {result.order}) but post-fill persistence failed; "
                    "record left PENDING for recovery."
                )

            log_security_event(
                "DEMO_ORDER_FILLED",
                {
                    "order_id": order_id,
                    "mt5_ticket": result.order,
                    "symbol": signal.symbol,
                    "volume": result.volume,
                    "price": result.price,
                },
                "INFO"
            )
            return True, order_rec, f"Demo order filled successfully (Ticket: {result.order})."

        else:
            # Mock provider / Simulated environment. Finding 6 applies here
            # too: a concurrent duplicate that slipped past the idempotency
            # check must surface as duplicate-prevention, not a raw 500.
            order_rec = DemoOrder(
                order_id=order_id,
                signal_id=signal.signal_id,
                idempotency_key=signal.idempotency_key,
                mt5_ticket=999999,
                symbol=signal.symbol,
                order_type=signal.direction,
                volume=execution_volume,
                requested_price=req_price,
                executed_price=req_price,
                slippage_points=0.0,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                status="FILLED",
                executed_at=utc_now_naive(),
            )
            if not _commit_pending_safely(order_rec):
                winner = self.db.query(DemoOrder).filter_by(
                    idempotency_key=signal.idempotency_key).first()
                return False, winner, (
                    f"{EXEC_MSG_DUPLICATE} by idempotency key {signal.idempotency_key}."
                )
            return True, order_rec, "Simulated demo order filled."
