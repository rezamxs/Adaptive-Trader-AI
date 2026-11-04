"""Persistent Circuit Breaker (Emergency Stop) Manager."""
from datetime import datetime
from typing import Optional, Tuple
from sqlalchemy.orm import Session

from backend.app.core.logging import logger, log_security_event
from backend.app.models.emergency_stop import EmergencyStopState
from backend.app.models.audit_log import AuditLog
from backend.app.db.base import utc_now_naive


class EmergencyStopManager:
    """Manages system-wide Emergency Stop state across memory and database."""

    _in_memory_state: bool = False
    _last_reason: str = "System boot"

    def __init__(self, db: Session):
        self.db = db
        self._sync_with_db()

    def _sync_with_db(self) -> None:
        """Loads the latest persistent emergency stop state from database."""
        latest = self.db.query(EmergencyStopState).order_by(EmergencyStopState.id.desc()).first()
        if latest:
            self._in_memory_state = latest.is_active
            self._last_reason = latest.reason
        else:
            self._in_memory_state = False
            self._last_reason = "No previous state recorded"

    def is_active(self) -> bool:
        """Checks if emergency stop is currently active.
        Re-queries DB to ensure cross-process consistency.
        """
        self._sync_with_db()
        return self._in_memory_state

    def get_status(self) -> Tuple[bool, str, Optional[datetime]]:
        """Returns (is_active, reason, activated_at)."""
        latest = self.db.query(EmergencyStopState).order_by(EmergencyStopState.id.desc()).first()
        if latest:
            return latest.is_active, latest.reason, latest.activated_at
        return False, "Not active", None

    def activate(self, reason: str, actor: str = "USER") -> None:
        """Immediately activates emergency stop and blocks all order execution."""
        state = EmergencyStopState(
            is_active=True,
            activated_by=actor,
            reason=reason,
            activated_at=utc_now_naive(),
        )
        self.db.add(state)
        
        audit = AuditLog(
            event_type="EMERGENCY_STOP_ACTIVATED",
            severity="CRITICAL",
            actor=actor,
            details={"reason": reason}
        )
        self.db.add(audit)
        self.db.commit()

        self._in_memory_state = True
        self._last_reason = reason

        log_security_event("EMERGENCY_STOP_ACTIVATED", {"actor": actor, "reason": reason}, "CRITICAL")
        logger.critical(f"EMERGENCY STOP ACTIVATED by {actor}. Reason: {reason}")

    def deactivate(self, actor: str = "USER", notes: str = "Manual resumption") -> None:
        """Deactivates emergency stop only via explicit administrative command."""
        state = EmergencyStopState(
            is_active=False,
            activated_by=actor,
            reason=f"Deactivated: {notes}",
            activated_at=utc_now_naive(),
            deactivated_at=utc_now_naive(),
            deactivated_by=actor,
        )
        self.db.add(state)

        audit = AuditLog(
            event_type="EMERGENCY_STOP_DEACTIVATED",
            severity="WARNING",
            actor=actor,
            details={"notes": notes}
        )
        self.db.add(audit)
        self.db.commit()

        self._in_memory_state = False
        self._last_reason = f"Deactivated by {actor}: {notes}"

        log_security_event("EMERGENCY_STOP_DEACTIVATED", {"actor": actor, "notes": notes}, "WARNING")
        logger.warning(f"EMERGENCY STOP DEACTIVATED by {actor}. Notes: {notes}")
