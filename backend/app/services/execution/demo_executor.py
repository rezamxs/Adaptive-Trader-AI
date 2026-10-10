"""Restricted Demo Trading Execution Engine.
Strictly isolated to Demo accounts with fail-closed safety assertions.
NO LIVE TRADING LOGIC EXISTS IN THIS MODULE OR APPLICATION.
"""
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from sqlalchemy.orm import Session

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
from backend.app.services.market_data.base import IMarketDataProvider
from backend.app.services.risk.circuit_breaker import EmergencyStopManager


class DemoTradingExecutor:
    """Isolated, fail-closed executor for demo trading orders only."""

    def __init__(self, db: Session, data_provider: IMarketDataProvider):
        self.db = db
        self.data_provider = data_provider
        self.emergency_stop = EmergencyStopManager(db)

    def execute_demo_signal(self, signal: TradeSignal) -> Tuple[bool, Optional[DemoOrder], str]:
        """Pre-flight verification and dispatch of an approved signal to DEMO account only.
        
        Returns:
            (success, demo_order_instance, message)
        """
        # 1. Check Signal Approval
        if signal.risk_status != "APPROVED":
            return False, None, f"Execution rejected: Signal risk status is '{signal.risk_status}'."

        if not signal.calculated_volume or signal.calculated_volume <= 0:
            return False, None, "Execution rejected: Calculated volume is missing or non-positive."

        # 2. Check Idempotency (prevent duplicate order)
        existing_order = self.db.query(DemoOrder).filter_by(idempotency_key=signal.idempotency_key).first()
        if existing_order:
            return False, existing_order, f"Duplicate order prevented by idempotency key {signal.idempotency_key}."

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
            return False, None, f"Safety Gate Blocked: {str(e)}"

        # 5. Live Quote & Spread Check
        quote = self.data_provider.get_current_quote(signal.symbol)
        if not quote or quote.is_stale:
            return False, None, "Market quote is stale or unavailable at moment of order dispatch."

        # 6. Build Order Request
        order_id = f"ord_{uuid.uuid4().hex[:10]}"
        req_price = quote.ask if signal.direction == "BUY" else quote.bid

        # Check terminal algorithmic trading flag if using real MT5 provider
        from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider
        if isinstance(self.data_provider, MT5MarketDataProvider) and MT5_AVAILABLE and is_connected:
            term = mt5.terminal_info()
            if not getattr(term, "trade_allowed", False):
                return False, None, "MT5 terminal global 'Algo Trading' button is disabled. Turn it ON in MT5 to allow automated demo orders."

            # Construct MT5 order request
            mt5_action = mt5.TRADE_ACTION_DEAL
            mt5_type = mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL

            request = {
                "action": mt5_action,
                "symbol": signal.symbol,
                "volume": float(signal.calculated_volume),
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

            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                err_code = getattr(result, "retcode", "N/A")
                err_comment = getattr(result, "comment", "Unknown MT5 error")
                err_msg = f"MT5 order rejected: Retcode={err_code}, Comment={err_comment}"
                logger.error(err_msg)

                order_rec = DemoOrder(
                    order_id=order_id,
                    signal_id=signal.signal_id,
                    idempotency_key=signal.idempotency_key,
                    symbol=signal.symbol,
                    order_type=signal.direction,
                    volume=signal.calculated_volume,
                    requested_price=req_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    status="REJECTED",
                    error_message=err_msg,
                )
                self.db.add(order_rec)
                self.db.commit()
                return False, order_rec, err_msg

            # Successful Fill
            order_rec = DemoOrder(
                order_id=order_id,
                signal_id=signal.signal_id,
                idempotency_key=signal.idempotency_key,
                mt5_ticket=result.order,
                symbol=signal.symbol,
                order_type=signal.direction,
                volume=result.volume,
                requested_price=req_price,
                executed_price=result.price,
                slippage_points=round(abs(result.price - req_price) / (self.data_provider.get_symbol_info(signal.symbol).point or 0.00001), 1),
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                status="FILLED",
                executed_at=datetime.utcnow(),
            )
            self.db.add(order_rec)
            self.db.commit()

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
            # Mock provider / Simulated environment
            order_rec = DemoOrder(
                order_id=order_id,
                signal_id=signal.signal_id,
                idempotency_key=signal.idempotency_key,
                mt5_ticket=999999,
                symbol=signal.symbol,
                order_type=signal.direction,
                volume=signal.calculated_volume,
                requested_price=req_price,
                executed_price=req_price,
                slippage_points=0.0,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                status="FILLED",
                executed_at=datetime.utcnow(),
            )
            self.db.add(order_rec)
            self.db.commit()
            return True, order_rec, "Simulated demo order filled."
