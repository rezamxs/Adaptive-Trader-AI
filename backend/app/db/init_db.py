"""Database table creation and initial safety state seeding."""
from sqlalchemy.orm import Session
from backend.app.db.base import Base
from backend.app.db.session import engine, SessionLocal
from backend.app.models.emergency_stop import EmergencyStopState
from backend.app.models.audit_log import AuditLog
from backend.app.core.logging import logger


def init_db(db: Session = None) -> None:
    """Creates all database tables and verifies initial safety state."""
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)

    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    try:
        # Check if an emergency stop state exists; if not, initialize as NOT active
        last_state = db.query(EmergencyStopState).order_by(EmergencyStopState.id.desc()).first()
        if not last_state:
            initial_state = EmergencyStopState(
                is_active=False,
                activated_by="SYSTEM_BOOTSTRAP",
                reason="System initialized with safety circuit breaker inactive by default."
            )
            db.add(initial_state)
            
            audit = AuditLog(
                event_type="SYSTEM_BOOTSTRAP",
                severity="INFO",
                actor="SYSTEM",
                details={"status": "Database initialized, emergency stop ready."}
            )
            db.add(audit)
            db.commit()
            logger.info("Initialized default emergency stop state (Inactive).")
        else:
            logger.info(f"Loaded existing emergency stop state: Active={last_state.is_active}")
    finally:
        if close_session:
            db.close()


if __name__ == "__main__":
    init_db()
    print("Database tables initialized successfully.")
