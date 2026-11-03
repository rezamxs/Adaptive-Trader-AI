"""Machine Learning and Model Training Endpoints."""
from typing import Literal, Optional
import math
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.core.auth import require_api_token
from backend.app.core.errors import ModelGovernanceError
from backend.app.core.logging import logger
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
    algorithm: Literal["logistic_regression", "random_forest", "baseline"] = "logistic_regression"
    bars_count: int = Field(default=300, ge=100, le=100000)
    horizon_bars: int = Field(default=15, ge=1, le=1000)


class PromoteModelRequest(BaseModel):
    notes: Optional[str] = "Promoted by user"
    actor: str = "USER"


@router.get("/models")
def list_models(db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Lists trained models and candidate artifacts (M15: model governance data — gated)."""
    registry = ModelRegistry(db)
    models = registry.list_models()
    return [{
        "model_id": m.model_id,
        "algorithm": m.algorithm,
        "status": m.status,
        "dataset_version": m.dataset_version_tag,
        "validation_metrics": m.validation_metrics,
        "test_metrics": m.test_metrics,
        "trained_at": m.trained_at.isoformat(),
        "promotion_notes": m.promotion_notes,
    } for m in models]


@router.post("/train")
def train_model(req: TrainModelRequest, db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Trains a new candidate model on historical data with strict temporal splitting."""
    from backend.app.services.ml.training import (
        TrainingDataError,
        TrainingInternalError,
        train_candidate_from_market,
    )

    try:
        train_res = train_candidate_from_market(
            db,
            symbol=req.symbol,
            timeframe=req.timeframe.value,
            bars_count=req.bars_count,
            horizon_bars=req.horizon_bars,
            algorithm=req.algorithm,
        )
    except TrainingDataError as e:
        # Environmental/input failures: 503 for MT5 problems, 400 for
        # insufficient bars (same contract as before the shared-helper refactor).
        detail = str(e)
        status = 400 if "Insufficient bars" in detail else 503
        raise HTTPException(status_code=status, detail=detail)
    except TrainingInternalError:
        raise HTTPException(
            status_code=500,
            detail="Model training failed due to an internal error.",
        )

    return {
        "model_id": train_res["model_id"],
        "status": train_res["status"],
        "validation_metrics": train_res["validation_metrics"],
        "test_metrics": train_res["test_metrics"],
        "features_count": len(train_res["features_used"]),
    }


@router.get("/models/{model_id}/compare")
def compare_model(model_id: str, db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Compares candidate model metrics against active model (M15: gated)."""
    registry = ModelRegistry(db)
    try:
        return registry.compare_models(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/models/{model_id}/promotion-check")
def promotion_check(model_id: str, db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Step 3.5: read-only evaluation of every promotion gate for a candidate.

    The system runs all checks and reports a per-gate summary (promotable
    true/false + the exact first failure); the human still performs the
    activation itself via POST /promote. Unknown model id is 404; a genuine
    internal bug is a sanitized 500."""
    registry = ModelRegistry(db)
    try:
        return registry.promotion_check(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception(f"Promotion pre-check failed internally for {model_id}")
        raise HTTPException(
            status_code=500,
            detail="Promotion pre-check failed due to an internal error.",
        )


@router.post("/models/{model_id}/promote")
def promote_model(model_id: str, req: PromoteModelRequest, db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Promotes candidate model to active status.

    M22: the failure classes must stay distinguishable — an unknown model id
    is 404, a governance/provenance gate rejection is 409 (conflict with
    current server state), and an unexpected bug is a sanitized 500 (never a
    blanket 400, and never raw internals in the detail)."""
    registry = ModelRegistry(db)
    try:
        promoted = registry.promote_candidate(model_id, actor=req.actor, notes=req.notes)
        return {"status": "SUCCESS", "model_id": promoted.model_id, "state": promoted.status}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelGovernanceError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"Model promotion failed with an internal error for {model_id}")
        raise HTTPException(
            status_code=500,
            detail="Model promotion failed due to an internal error.",
        )
