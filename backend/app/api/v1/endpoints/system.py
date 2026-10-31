"""System Status, Worker Telemetry, and Emergency Stop Endpoints."""
from datetime import datetime, timezone
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.auth import require_api_token
from backend.app.db.session import get_db
from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider
from backend.app.services.risk.circuit_breaker import EmergencyStopManager
from backend.app.services.worker.market_worker import get_market_worker
from backend.app.models.audit_log import AuditLog
from backend.app.models.worker_cycle import WorkerCycleRecord

router = APIRouter()


class EmergencyStopRequest(BaseModel):
    reason: str
    actor: str = "USER"


class EmergencyStopDeactivateRequest(BaseModel):
    notes: str = "Manual resumption"
    actor: str = "USER"


@router.get("/status")
def get_system_status(db: Session = Depends(get_db), _token: str = Depends(require_api_token)) -> Dict[str, Any]:
    """Returns real-time status of system, database, MT5, worker, demo mode, and circuit breaker.

    M15: this payload carries live account context (MT5 connection, demo
    verification) — token-gated, not protected by the 127.0.0.1 default alone.
    """
    stop_mgr = EmergencyStopManager(db)
    is_stop_active, stop_reason, stop_at = stop_mgr.get_status()

    # Query Worker diagnostics
    worker = get_market_worker()
    worker_diag = worker.get_diagnostics()

    # Query MT5 connection status
    active_provider = worker.data_provider if (worker and worker.data_provider.is_connected()) else MT5MarketDataProvider()
    is_mt5_connected = active_provider.is_connected()
    account_info = active_provider.get_account_info() if is_mt5_connected else None

    return {
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "execution_mode": settings.EXECUTION_MODE.value,
        "demo_trading_gate_enabled": settings.DEMO_TRADING_ENABLED,
        "database_connected": True,
        "liveness": worker_diag["liveness"],
        "readiness": worker_diag["readiness"],
        "service_ready": worker_diag.get("service_ready", False),
        "service_ready_reason": worker_diag.get("service_ready_reason", ""),
        "market_data_ready": worker_diag.get("market_data_ready", False),
        "market_data_reason": worker_diag.get("market_data_reason", ""),
        "trading_ready": worker_diag.get("trading_ready", False),
        "trading_readiness_reason": worker_diag.get("trading_readiness_reason", ""),
        "worker": worker_diag,
        "emergency_stop": {
            "is_active": is_stop_active,
            "reason": stop_reason,
            "activated_at": stop_at.isoformat() if stop_at else None,
        },
        "mt5": {
            "terminal_path": settings.MT5_PATH,
            "is_connected": is_mt5_connected,
            "account_verified_demo": account_info.is_demo if account_info else False,
            "account_trade_mode": account_info.trade_mode if account_info else None,
            "server": account_info.server if account_info else "N/A",
            "currency": account_info.currency if account_info else "N/A",
            "balance": account_info.balance if account_info else 0.0,
            "equity": account_info.equity if account_info else 0.0,
            "login_masked": account_info.login_masked if account_info else "N/A",
        },
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/worker/status")
def get_worker_status(_token: str = Depends(require_api_token)) -> Dict[str, Any]:
    """Returns granular real-time telemetry from the background market worker (M15: gated)."""
    worker = get_market_worker()
    return worker.get_diagnostics()


@router.get("/worker/cycles")
def get_worker_cycles(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db), _token: str = Depends(require_api_token)) -> List[Dict[str, Any]]:
    """Returns recent persistent worker execution cycle records (M15: gated)."""
    # limit is bounded: SQLite treats LIMIT -1 as unlimited, so an unbounded
    # query param could dump the whole table (memory DoS / bulk exfiltration).
    records = db.query(WorkerCycleRecord).order_by(WorkerCycleRecord.id.desc()).limit(limit).all()
    return [{
        "id": r.id,
        "cycle_id": r.cycle_id,
        "symbol": r.symbol,
        "timeframe": r.timeframe,
        "bar_key": r.bar_key,
        "bar_time": r.bar_time.isoformat() if r.bar_time else None,
        "status_code": r.status_code,
        "lifecycle_phase": r.lifecycle_phase,
        "decision": r.decision,
        "signal_id": r.signal_id,
        "duration_ms": r.duration_ms,
        "details": r.details,
        "error_message": r.error_message,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    } for r in records]


@router.post("/emergency-stop/activate")
def activate_emergency_stop(req: EmergencyStopRequest, db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Triggers emergency stop circuit breaker immediately."""
    stop_mgr = EmergencyStopManager(db)
    stop_mgr.activate(reason=req.reason, actor=req.actor)
    return {"status": "ACTIVATED", "reason": req.reason}


@router.post("/emergency-stop/deactivate")
def deactivate_emergency_stop(req: EmergencyStopDeactivateRequest, db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Deactivates emergency stop circuit breaker."""
    stop_mgr = EmergencyStopManager(db)
    stop_mgr.deactivate(actor=req.actor, notes=req.notes)
    return {"status": "DEACTIVATED", "notes": req.notes}


@router.get("/audit-logs")
def get_audit_logs(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Returns recent audit logs (M15: security-relevant history — gated)."""
    logs = db.query(AuditLog).order_by(AuditLog.id.desc()).limit(limit).all()
    return [{
        "id": l.id,
        "event_type": l.event_type,
        "severity": l.severity,
        "actor": l.actor,
        "details": l.details,
        "created_at": l.created_at.isoformat(),
    } for l in logs]
