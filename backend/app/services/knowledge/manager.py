"""Knowledge and Rules Management Service with strict human approval workflows.
Ensures unapproved rules NEVER enter the active trading strategy.
"""
import uuid
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.logging import logger, log_security_event
from backend.app.models.document import Document
from backend.app.models.knowledge_chunk import KnowledgeChunk
from backend.app.models.rule import TradingRule
from backend.app.models.audit_log import AuditLog
from backend.app.schemas.rule_schema import ApprovalStatus, RuleApprovalUpdate, TradingRuleCreate
from backend.app.services.knowledge.parsers import DocumentParser
from backend.app.services.knowledge.extractor import RuleExtractor
from backend.app.services.knowledge.retriever import KnowledgeRetriever


class KnowledgeManager:
    """Orchestrates document ingestion, extraction, ambiguity tracking, and human review."""

    def __init__(self, db: Session):
        self.db = db
        self.retriever = KnowledgeRetriever(db)

    def ingest_document(
        self,
        file_name: str,
        file_bytes: bytes,
        source_description: Optional[str] = None
    ) -> Tuple[Document, List[TradingRule]]:
        """Ingests a file, saves to raw storage, parses content, and extracts pending rules."""
        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        file_size = len(file_bytes)

        # Validate file size (max 250MB to support large trading course books)
        if file_size > 250 * 1024 * 1024:
            raise ValueError("File exceeds maximum allowed size of 250MB.")

        # Keep uploads inside the raw-data directory; never trust path components from a filename.
        safe_file_name = Path(str(file_name).replace("\\", "/")).name
        ext = Path(safe_file_name).suffix.lower()
        storage_path = settings.DATA_RAW_DIR / f"{document_id}_{safe_file_name}"
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(storage_path, "wb") as f:
            f.write(file_bytes)

        # Parse file
        try:
            full_text, pages_count, page_chunks = DocumentParser.parse_file(safe_file_name, file_bytes)
            status = "completed"
            error_msg = None
        except Exception as e:
            logger.error(f"Failed to parse document {file_name}: {e}")
            full_text = ""
            pages_count = 0
            page_chunks = []
            status = "failed"
            error_msg = str(e)

        # Create Document record
        doc_record = Document(
            document_id=document_id,
            file_name=safe_file_name,
            file_type=ext.lstrip("."),
            file_size_bytes=file_size,
            source_description=source_description or "Uploaded knowledge document",
            processing_status=status,
            pages_count=pages_count,
            raw_text=full_text,
            error_message=error_msg,
        )
        self.db.add(doc_record)
        self.db.commit()
        self.db.refresh(doc_record)

        # Persist searchable evidence chunks before extracting rules. Retrieval reads
        # KnowledgeChunk, so saving only Document.raw_text would make uploads invisible.
        created_chunks: List[KnowledgeChunk] = []
        created_rules: List[TradingRule] = []
        if status == "completed" and page_chunks:
            for index, part in enumerate(page_chunks, start=1):
                if not isinstance(part, dict):
                    continue
                chunk_text = str(part.get("text", "")).strip()
                if not chunk_text:
                    continue
                sheet_name = part.get("sheet")
                page_value = part.get("page", index)
                try:
                    page_number = int(page_value) if page_value is not None else None
                except (TypeError, ValueError):
                    page_number = None
                if sheet_name:
                    location_type = "sheet"
                    location_reference = f"Sheet: {str(sheet_name)[:100]}"
                else:
                    location_type = "page"
                    location_reference = f"Page {page_number or index}"
                first_line = next((line.strip() for line in chunk_text.splitlines() if line.strip()), "")
                title = first_line[:255] or safe_file_name
                chunk = KnowledgeChunk(
                    chunk_id=f"chunk_{uuid.uuid4().hex[:16]}",
                    document_id=document_id,
                    source_file=safe_file_name,
                    location_type=location_type,
                    location_reference=location_reference,
                    page_number=page_number,
                    title=title,
                    content=chunk_text,
                    key_concepts=[],
                    parameters={},
                    exceptions_and_limits=[],
                )
                self.db.add(chunk)
                created_chunks.append(chunk)
            self.db.commit()
            self.retriever.refresh()

            rule_creates = RuleExtractor.extract_from_pages(document_id, page_chunks)
            for rc in rule_creates:
                rule_id = f"rule_{uuid.uuid4().hex[:8]}"
                rule_record = TradingRule(
                    rule_id=rule_id,
                    name=rc.name,
                    category=rc.category.value,
                    description=rc.description,
                    conditions=rc.conditions,
                    parameters=rc.parameters,
                    symbol_scope=rc.symbol_scope,
                    timeframe_scope=rc.timeframe_scope,
                    invalidation_conditions=rc.invalidation_conditions,
                    source_document=rc.source_document,
                    source_pages=rc.source_pages,
                    assumptions=rc.assumptions,
                    ambiguities=rc.ambiguities,
                    confidence=rc.confidence,
                    approval_status=ApprovalStatus.PENDING.value,
                    version=1,
                )
                self.db.add(rule_record)
                created_rules.append(rule_record)

            self.db.commit()

        # Audit log
        audit = AuditLog(
            event_type="DOCUMENT_INGESTION",
            severity="INFO",
            actor="USER",
            details={
                "document_id": document_id,
                "file_name": safe_file_name,
                "rules_extracted": len(created_rules),
            }
        )
        self.db.add(audit)
        self.db.commit()

        return doc_record, created_rules

    def get_rules(self, approval_status: Optional[str] = None) -> List[TradingRule]:
        """Retrieves rules, optionally filtered by approval status."""
        query = self.db.query(TradingRule)
        if approval_status:
            query = query.filter(TradingRule.approval_status == approval_status)
        return query.order_by(TradingRule.id.desc()).all()

    def approve_rule(self, rule_id: str, actor: str = "USER") -> TradingRule:
        """Explicitly approves a rule for inclusion in active strategy logic."""
        rule = self.db.query(TradingRule).filter(TradingRule.rule_id == rule_id).first()
        if not rule:
            raise ValueError(f"Rule with ID {rule_id} not found.")

        rule.approval_status = ApprovalStatus.APPROVED.value
        rule.rejection_reason = None
        self.db.commit()
        self.db.refresh(rule)

        log_security_event("RULE_APPROVED", {"rule_id": rule_id, "name": rule.name, "actor": actor}, "INFO")
        return rule

    def reject_rule(self, rule_id: str, reason: str, actor: str = "USER") -> TradingRule:
        """Rejects a candidate rule with an explicit reason."""
        rule = self.db.query(TradingRule).filter(TradingRule.rule_id == rule_id).first()
        if not rule:
            raise ValueError(f"Rule with ID {rule_id} not found.")

        rule.approval_status = ApprovalStatus.REJECTED.value
        rule.rejection_reason = reason
        self.db.commit()
        self.db.refresh(rule)

        log_security_event("RULE_REJECTED", {"rule_id": rule_id, "reason": reason, "actor": actor}, "INFO")
        return rule

    def get_active_strategy_rules(self) -> List[TradingRule]:
        """CRITICAL SAFETY INVARIANT: Only returns rules with approval_status == 'approved'.
        Pending or rejected rules can NEVER be returned by this method.
        """
        return self.db.query(TradingRule).filter(
            TradingRule.approval_status == ApprovalStatus.APPROVED.value
        ).all()

    def check_rule_conflicts(self) -> List[Dict[str, Any]]:
        """Identifies conflicting rules (e.g. conflicting directions or contradictory thresholds)."""
        rules = self.get_active_strategy_rules()
        conflicts = []

        for i, r1 in enumerate(rules):
            for r2 in rules[i + 1:]:
                # Check overlapping symbol and timeframe scopes
                common_symbols = set(r1.symbol_scope).intersection(set(r2.symbol_scope))
                common_tf = set(r1.timeframe_scope).intersection(set(r2.timeframe_scope))
                
                if common_symbols and common_tf:
                    # Detect opposite directions in entry rules
                    is_buy_1 = "buy" in r1.name.lower() or "خرید" in r1.name
                    is_sell_1 = "sell" in r1.name.lower() or "فروش" in r1.name
                    is_buy_2 = "buy" in r2.name.lower() or "خرید" in r2.name
                    is_sell_2 = "sell" in r2.name.lower() or "فروش" in r2.name

                    if (is_buy_1 and is_sell_2) or (is_sell_1 and is_buy_2):
                        conflicts.append({
                            "rule_1_id": r1.rule_id,
                            "rule_1_name": r1.name,
                            "rule_2_id": r2.rule_id,
                            "rule_2_name": r2.name,
                            "symbols": list(common_symbols),
                            "conflict_type": "CONTRADICTORY_DIRECTION",
                            "message": "Both BUY and SELL rules active concurrently on same scope without priority resolution."
                        })

        return conflicts

    def retrieve_knowledge(
        self,
        query: str,
        top_k: int = 5,
        source_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Searches knowledge chunks and returns structured, source-cited results."""
        return self.retriever.search(query=query, top_k=top_k, source_file=source_file)

    def query_knowledge_grounded(
        self,
        query: str,
        top_k: int = 5,
        source_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Queries knowledge base with strict grounding and zero-hallucination verification."""
        return self.retriever.query_with_grounding(query=query, top_k=top_k, source_file=source_file)

