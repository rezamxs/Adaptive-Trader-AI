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
        "trading_ready": worker.trading_ready,
        "trading_readiness_reason": worker.trading_readiness_reason,
        "worker_running": worker.is_running,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)
