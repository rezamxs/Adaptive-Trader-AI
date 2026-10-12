"""Machine Learning and Model Training Endpoints."""
from typing import Any, Dict, Optional
import math
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.services.market_data.mt5_provider import MT5MarketDataProvider
from backend.app.services.market_data.base import TimeFrame
from backend.app.services.dataset.pipeline import DatasetPipeline
from backend.app.services.ml.trainer import ModelTrainer
from backend.app.services.ml.registry import ModelRegistry
from backend.app.models.ml_model import MLModelRecord

router = APIRouter()


class TrainModelRequest(BaseModel):
    symbol: str = Field(default="EURUSD", min_length=1, max_length=32)
    timeframe: TimeFrame = TimeFrame.M15
    algorithm: str = "logistic_regression"  # logistic_regression, random_forest, baseline
    bars_count: int = Field(default=300, ge=100, le=100000)
    horizon_bars: int = Field(default=15, ge=1, le=1000)


class PromoteModelRequest(BaseModel):
    notes: Optional[str] = "Promoted by user"
    actor: str = "USER"


@router.get("/models")
def list_models(db: Session = Depends(get_db)):
    """Lists trained models and candidate artifacts."""
    registry = ModelRegistry(db)
    models = registry.list_models()
    return [{
        "model_id": m.model_id,
        "algorithm": m.algorithm,
        "status": m.status,
        "dataset_version": m.dataset_version_tag,
        "test_metrics": m.test_metrics,
        "trained_at": m.trained_at.isoformat(),
        "promotion_notes": m.promotion_notes,
    } for m in models]


@router.post("/train")
def train_model(req: TrainModelRequest, db: Session = Depends(get_db)):
    """Trains a new candidate model on historical data with strict temporal splitting."""
    provider = MT5MarketDataProvider()
    if not provider.connect():
        raise HTTPException(
            status_code=503,
            detail="MT5 is unavailable; model training with synthetic fallback data is disabled.",
        )

    try:
        df_raw = provider.get_historical_dataframe(req.symbol, req.timeframe, count=req.bars_count)
        symbol_info = provider.get_symbol_info(req.symbol)
    finally:
        provider.disconnect()

    if symbol_info is None:
        raise HTTPException(status_code=503, detail=f"MT5 did not return broker specifications for {req.symbol}.")
    if (
        not math.isfinite(float(symbol_info.point))
        or float(symbol_info.point) <= 0
        or int(symbol_info.spread) < 0
    ):
        raise HTTPException(status_code=503, detail=f"MT5 returned invalid point/spread data for {req.symbol}.")

    if len(df_raw) < 100:
        raise HTTPException(status_code=400, detail="Insufficient bars to build training dataset.")

    # Build dataset with purge/embargo
    pipeline = DatasetPipeline(db=db)
    try:
        ds_res = pipeline.build_dataset(
            df_raw,
            symbol=req.symbol,
            timeframe=req.timeframe.value,
            horizon_bars=req.horizon_bars,
            spread_points=max(0, int(symbol_info.spread)),
            point_value=float(symbol_info.point),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset construction failed: {str(e)}")

    # Train model
    trainer = ModelTrainer(db=db)
    try:
        train_res = trainer.train_candidate(
            dataset_result=ds_res,
            algorithm=req.algorithm
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

    return {
        "model_id": train_res["model_id"],
        "status": train_res["status"],
        "validation_metrics": train_res["validation_metrics"],
        "test_metrics": train_res["test_metrics"],
        "features_count": len(train_res["features_used"]),
    }


@router.get("/models/{model_id}/compare")
def compare_model(model_id: str, db: Session = Depends(get_db)):
    """Compares candidate model metrics against active model."""
    registry = ModelRegistry(db)
    try:
        return registry.compare_models(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/models/{model_id}/promote")
def promote_model(model_id: str, req: PromoteModelRequest, db: Session = Depends(get_db)):
    """Promotes candidate model to active status."""
    registry = ModelRegistry(db)
    try:
        promoted = registry.promote_candidate(model_id, actor=req.actor, notes=req.notes)
        return {"status": "SUCCESS", "model_id": promoted.model_id, "state": promoted.status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
