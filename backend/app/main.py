"""FastAPI Main Application for Adaptive Trader AI."""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.db.init_db import init_db
from backend.app.api.v1.api_router import api_router
from backend.app.services.worker.market_worker import get_market_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Controlled application lifespan managing DB initialization and background worker."""
    # Startup phase
    init_db()
    # Owner decision 2026-11-01: retention windows default to 30 days per
    # table (a monthly summary is archived before pruning; per-table 0 still
    # disables). See docs/ROADMAP.md, owner-decisions table.
    try:
        from backend.app.db.session import SessionLocal
        from backend.app.services.worker.market_worker import prune_operational_records
        if (
            settings.WORKER_CYCLE_RETENTION_DAYS
            or settings.AUDIT_LOG_RETENTION_DAYS
            or settings.TRADE_SIGNAL_RETENTION_DAYS
            or settings.DEMO_ORDER_RETENTION_DAYS
        ):
            with SessionLocal() as db:
                pruned = prune_operational_records(
                    db,
                    worker_cycle_days=settings.WORKER_CYCLE_RETENTION_DAYS,
                    audit_log_days=settings.AUDIT_LOG_RETENTION_DAYS,
                    trade_signal_days=settings.TRADE_SIGNAL_RETENTION_DAYS,
                    demo_order_days=settings.DEMO_ORDER_RETENTION_DAYS,
                )
            logger.info("Retention sweep at startup: %s", pruned)
    except Exception as e:
        logger.warning(f"Retention sweep failed at startup: {e}")
    # Step 3.1 (owner request 2026-11-01): documents dropped into
    # data/document_inbox/ are ingested automatically at startup through the
    # same transactional pipeline as the upload endpoint. Parsing is heavy,
    # so the scan runs off the event loop (same rule as the M11 fix).
    try:
        import asyncio
        from pathlib import Path as _Path

        from backend.app.db.session import SessionLocal as _SessionLocal
        from backend.app.services.knowledge.inbox import process_inbox

        if _Path(settings.DOCUMENT_INBOX_DIR).exists():
            with _SessionLocal() as inbox_db:
                inbox_summary = await asyncio.to_thread(process_inbox, inbox_db)
            logger.info("Document inbox scan at startup: %s", inbox_summary)
    except Exception as e:
        logger.warning(f"Document inbox scan failed at startup: {e}")
    # Step 3.4: nightly scheduled candidate training (never promotes).
    training_scheduler = None
    try:
        from backend.app.db.session import SessionLocal as _TrainingSession
        from backend.app.services.ml.training_scheduler import TrainingScheduler

        training_scheduler = TrainingScheduler(db_factory=_TrainingSession)
        training_scheduler.start()
    except Exception as e:
        logger.warning(f"Training scheduler failed to start: {e}")
    worker = None
    if settings.WORKER_ENABLED:
        try:
            worker = get_market_worker()
            await worker.start()
            logger.info("MarketMonitoringWorker started during application lifespan startup.")
        except Exception as e:
            logger.error(f"Failed to start MarketMonitoringWorker during lifespan startup: {e}")

    yield

    # Shutdown phase
    if training_scheduler:
        try:
            training_scheduler.stop()
            logger.info("Training scheduler stopped during application lifespan shutdown.")
        except Exception as e:
            logger.warning(f"Error stopping training scheduler during shutdown: {e}")
    if worker and worker.is_running:
        try:
            await worker.stop()
            logger.info("MarketMonitoringWorker stopped during application lifespan shutdown.")
        except Exception as e:
            logger.warning(f"Error stopping MarketMonitoringWorker during lifespan shutdown: {e}")


app = FastAPI(
    title=settings.APP_NAME,
    description="Adaptive Algorithmic Trading AI with MT5 Integration, Background Market Worker, and Risk Safety Gates",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Minor 3, phase 2: script-src carries NO 'unsafe-inline' anymore. The
# dashboard JS lives in the external same-origin file /static/app.js and every
# inline onclick= handler was replaced with data-action delegation, so the
# dashboard AND the whole API run under a strict 'self'-only script policy.
# The ONLY exception is FastAPI's own Swagger UI (/docs, /redoc), whose
# bundled bootstrap is an inline script served by the framework itself —
# pinned by test_audit_fixes_corr1 as a single, documented exception.
_SWAGGER_INLINE_PATHS = ("/docs", "/redoc", "/openapi.json")


def _content_security_policy(path: str) -> str:
    if path in _SWAGGER_INLINE_PATHS:
        return (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com "
            "https://cdn.jsdelivr.net; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'"
        )
    return (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'"
    )


@app.middleware("http")
async def security_headers(request, call_next):
    """Baseline hardening headers on every response, with the CSP scoped to
    the path class (strict everywhere; the Swagger exception only on the
    Swagger paths)."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Content-Security-Policy",
        _content_security_policy(request.url.path),
    )
    return response

# Include API Router
app.include_router(api_router, prefix="/api/v1")

# Mount Static Frontend
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    def serve_dashboard():
        index_file = frontend_dir / "index.html"
        return FileResponse(index_file)


@app.get("/health", tags=["Health"])
def health_check():
    """Liveness and separated readiness check with worker status."""
    worker = get_market_worker()
    # One shared governance probe for both trading-readiness fields: a single
    # model-artifact load per health poll instead of two pickle deserializations.
    probe_ready, probe_reason = worker.trading_governance_probe()
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "mode": settings.EXECUTION_MODE.value,
        "liveness": worker.liveness_status,
        "readiness": worker.readiness_status,
        "service_ready": worker.service_ready,
        "service_ready_reason": worker.service_ready_reason,
        "market_data_ready": worker.market_data_ready,
        "market_data_reason": worker.market_data_reason,
        "trading_ready": probe_ready,
        "trading_readiness_reason": probe_reason,
        "worker_running": worker.is_running,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)
