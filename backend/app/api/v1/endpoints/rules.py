"""Trading Rules and Human Approval Endpoints."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.core.auth import require_api_token
from backend.app.core.logging import log_security_event
from backend.app.models.rule import TradingRule
from backend.app.schemas.rule_schema import (
    BatchReviewRequest,
    RuleDirectionUpdate,
    TradingRuleResponse,
)
from backend.app.services.knowledge.manager import KnowledgeManager

router = APIRouter()


class RejectRuleRequest(BaseModel):
    # Buglist item 19: the reason lands in an unbounded DB Text column —
    # the API schema is the effective length gate.
    reason: str = Field(min_length=1, max_length=2000)
    actor: str = Field(default="USER", max_length=100)


class ApproveRuleRequest(BaseModel):
    actor: str = "USER"


@router.get("/", response_model=List[TradingRuleResponse])
def list_rules(
    approval_status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    _token: str = Depends(require_api_token),
):
    """Lists trading rules, optionally filtered by status ('pending', 'approved', 'rejected'). M15: strategy definitions are gated."""
    km = KnowledgeManager(db)
    return km.get_rules(approval_status=approval_status, limit=limit)


@router.post("/{rule_id}/approve", response_model=TradingRuleResponse)
def approve_rule(rule_id: str, req: ApproveRuleRequest, db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Human approval of a candidate trading rule. Authorizes it for strategy use."""
    km = KnowledgeManager(db)
    try:
        return km.approve_rule(rule_id=rule_id, actor=req.actor)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{rule_id}/reject", response_model=TradingRuleResponse)
def reject_rule(rule_id: str, req: RejectRuleRequest, db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Human rejection of a trading rule with an explicit rationale."""
    km = KnowledgeManager(db)
    try:
        return km.reject_rule(rule_id=rule_id, reason=req.reason, actor=req.actor)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{rule_id}/direction", response_model=TradingRuleResponse)
def set_rule_direction(rule_id: str, req: RuleDirectionUpdate, db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """M4: assigns (or clears) the explicit direction of a rule.

    This is the ONLY path that decides a rule's direction — the generator and
    conflict detection never infer it from the rule name. A null direction
    leaves the rule fail-closed (it triggers but casts no direction vote).
    """
    rule = db.query(TradingRule).filter(TradingRule.rule_id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail=f"Rule with ID {rule_id} not found.")
    rule.direction = req.direction.value if req.direction is not None else None
    db.commit()
    db.refresh(rule)
    log_security_event(
        "RULE_DIRECTION_SET",
        {"rule_id": rule_id, "direction": rule.direction, "notes": req.notes, "actor": req.actor},
        "INFO",
    )
    return rule


@router.post("/batch-review")
def batch_review_rules(req: BatchReviewRequest, db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Step 3.3: human BATCH review — the system proposes, the human disposes.

    Each item independently approves (optionally assigning the executable
    direction: 'BUY'/'SELL' explicitly, 'SUGGESTED' to adopt the extractor's
    suggestion, or null for a directionless fail-closed rule) or rejects with
    a reason. One bad item never aborts the batch; every item returns an
    explicit result.
    """
    km = KnowledgeManager(db)
    results = km.batch_review_rules(
        [item.model_dump() for item in req.items], actor="USER"
    )
    return {"results": results}


@router.get("/conflicts")
def check_rule_conflicts(db: Session = Depends(get_db), _token: str = Depends(require_api_token)):
    """Checks for contradictory rules among approved strategy rules (M15: gated)."""
    km = KnowledgeManager(db)
    return km.check_rule_conflicts()
