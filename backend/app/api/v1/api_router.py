"""Consolidated API Router for v1 endpoints."""
from fastapi import APIRouter

from backend.app.api.v1.endpoints.system import router as system_router
from backend.app.api.v1.endpoints.documents import router as documents_router
from backend.app.api.v1.endpoints.rules import router as rules_router
from backend.app.api.v1.endpoints.market import router as market_router
from backend.app.api.v1.endpoints.ml import router as ml_router
from backend.app.api.v1.endpoints.backtest import router as backtest_router
from backend.app.api.v1.endpoints.signals import router as signals_router

api_router = APIRouter()

api_router.include_router(system_router, prefix="/system", tags=["System"])
api_router.include_router(documents_router, prefix="/documents", tags=["Documents"])
api_router.include_router(rules_router, prefix="/rules", tags=["Trading Rules"])
api_router.include_router(market_router, prefix="/market-data", tags=["Market Data"])
api_router.include_router(ml_router, prefix="/ml", tags=["Machine Learning"])
api_router.include_router(backtest_router, prefix="/backtest", tags=["Backtesting"])
api_router.include_router(signals_router, prefix="/signals", tags=["Signals & Orders"])
