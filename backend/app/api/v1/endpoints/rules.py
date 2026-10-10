"""Trading Rules and Human Approval Endpoints."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.schemas.rule_schema import TradingRuleResponse, ApprovalStatus
from backend.app.services.knowledge.manager import KnowledgeManager

router = APIRouter()


class RejectRuleRequest(BaseModel):
    reason: str
    actor: str = "USER"


class ApproveRuleRequest(BaseModel):
    actor: str = "USER"


@router.get("/", response_model=List[TradingRuleResponse])
def list_rules(approval_status: Optional[str] = None, db: Session = Depends(get_db)):
    """Lists trading rules, optionally filtered by status ('pending', 'approved', 'rejected')."""
    km = KnowledgeManager(db)
    return km.get_rules(approval_status=approval_status)


@router.post("/{rule_id}/approve", response_model=TradingRuleResponse)
def approve_rule(rule_id: str, req: ApproveRuleRequest, db: Session = Depends(get_db)):
    """Human approval of a candidate trading rule. Authorizes it for strategy use."""
    km = KnowledgeManager(db)
    try:
        return km.approve_rule(rule_id=rule_id, actor=req.actor)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{rule_id}/reject", response_model=TradingRuleResponse)
def reject_rule(rule_id: str, req: RejectRuleRequest, db: Session = Depends(get_db)):
    """Human rejection of a trading rule with an explicit rationale."""
    km = KnowledgeManager(db)
    try:
        return km.reject_rule(rule_id=rule_id, reason=req.reason, actor=req.actor)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/conflicts")
def check_rule_conflicts(db: Session = Depends(get_db)):
    """Checks for contradictory rules among approved strategy rules."""
    km = KnowledgeManager(db)
    return km.check_rule_conflicts()
