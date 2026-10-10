"""Pydantic schemas for Trading Rules and Human Approval."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RuleCategory(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"
    FILTER = "filter"
    RISK = "risk"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ConditionItem(BaseModel):
    indicator_or_field: str
    operator: str  # >, <, >=, <=, ==, CROSSES_ABOVE, CROSSES_BELOW
    value_or_threshold: Any
    description: Optional[str] = None


class TradingRuleBase(BaseModel):
    name: str
    category: RuleCategory
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

    class Config:
        from_attributes = True
