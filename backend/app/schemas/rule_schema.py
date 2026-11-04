"""Pydantic schemas for Trading Rules and Human Approval."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class RuleCategory(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"
    FILTER = "filter"
    RISK = "risk"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class RuleDirection(str, Enum):
    """M4: explicit direction. The rule name is never used to infer it."""

    BUY = "BUY"
    SELL = "SELL"


class TradingRuleBase(BaseModel):
    name: str
    category: RuleCategory
    # M4: explicit direction; NULL (undecided) fails closed at execution.
    direction: Optional[RuleDirection] = None
    # Step 3.3: extractor SUGGESTION from the document text — never
    # executable on its own; the human batch review decides the real
    # `direction`. Kept separate from `direction` so a suggestion can never
    # vote (the generator reads `direction` only).
    suggested_direction: Optional[RuleDirection] = None
    description: str
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    symbol_scope: List[str] = Field(default_factory=list)
    timeframe_scope: List[str] = Field(default_factory=list)
    invalidation_conditions: List[str] = Field(default_factory=list)
    source_document: Optional[str] = None
    source_pages: List[int] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    ambiguities: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None


class TradingRuleCreate(TradingRuleBase):
    pass


class RuleDirectionUpdate(BaseModel):
    """M4: conscious, reviewable direction assignment (a Phase P human decision).

    Sending null clears the direction — the rule then never casts a direction
    vote (fail-closed)."""

    direction: Optional[RuleDirection] = None
    notes: Optional[str] = None
    actor: str = "USER"


class BatchReviewItem(BaseModel):
    """One human decision inside a batch rule review (step 3.3).

    action=approve: `direction` may be 'BUY'/'SELL' (explicit assignment),
    'SUGGESTED' (adopt the extractor's suggested_direction), or None (rule
    approved with no direction — it will never cast a direction vote).
    action=reject: requires `reason`."""

    rule_id: str
    action: str  # 'approve' | 'reject' (validated per item, not by the enum)
    direction: Optional[str] = None  # 'BUY' | 'SELL' | 'SUGGESTED' | None
    reason: Optional[str] = None
    notes: Optional[str] = None


class BatchReviewRequest(BaseModel):
    items: List[BatchReviewItem] = Field(min_length=1, max_length=200)


class RuleApprovalUpdate(BaseModel):
    approval_status: ApprovalStatus
    rejection_reason: Optional[str] = None
    notes: Optional[str] = None


class TradingRuleResponse(TradingRuleBase):
    id: int
    rule_id: str
    approval_status: ApprovalStatus
    rejection_reason: Optional[str] = None
    version: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
