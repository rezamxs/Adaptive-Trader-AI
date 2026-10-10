"""FastAPI Main Application for Adaptive Trader AI."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.db.init_db import init_db
from backend.app.api.v1.api_router import api_router

# Initialize Database Schema on start
init_db()

app = FastAPI(
    title=settings.APP_NAME,
    description="Adaptive Algorithmic Trading AI with MT5 Integration and Risk Safety Gates",
    version="1.0.0",
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
    """Simple liveness check."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "mode": settings.EXECUTION_MODE.value,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)
