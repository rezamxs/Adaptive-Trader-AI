"""Knowledge and Rules Management Service with strict human approval workflows.
Ensures unapproved rules NEVER enter the active trading strategy.
"""
import uuid
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

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
    """Orchestrates document ingestion, extraction, ambiguity tracking, and human review.

    Document processing state machine (processing_status):

        (new/recovery) ──► processing ──► completed
                              │               ▲
                              │               └── retry: incomplete docs are
                              │                   detected by hash, partial
                              │                   chunks/rules deleted, and
                              │                   the whole parse+persist
                              │                   re-run exactly once more
                              ▼
                           failed ───────────────┘

        - `completed` is committed ONLY after the chunk batch and the rule
          batch have both been committed (never before; failure between the
          two keeps the document recoverable, not completed-without-chunks).
        - `failed` is committed after a rollback, so no partial chunk/rule
          batch survives inside a committed 'failed' row; chunk batches
          committed before a later rule-stage failure are removed by the
          recovery path on the next ingestion of the same content hash.
        - There is no transition out of `completed` except recovery for the
          legacy corrupted case (completed with zero chunks and no
          NO_EXTRACTABLE_CONTENT marker).
    """

    _CHUNK = 1024 * 1024

    def __init__(self, db: Session):
        self.db = db
        self.retriever = KnowledgeRetriever(db)

    @classmethod
    def _hash_file_chunked(cls, path: Path) -> str:
        """SHA-256 over the file read in 1MB chunks; never holds the whole file."""
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                block = f.read(cls._CHUNK)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    def ingest_document(
        self,
        file_name: str,
        file_bytes: bytes,
        source_description: Optional[str] = None
    ) -> Tuple[Document, List[TradingRule]]:
        """Ingests file bytes (kept for API compatibility; the upload endpoint
        uses ingest_document_path so raw files never become one bytes object)."""
        return self._ingest(file_name, source_description, file_bytes=file_bytes)

    def ingest_document_path(
        self,
        file_name: str,
        file_path,
        source_description: Optional[str] = None
    ) -> Tuple[Document, List[TradingRule]]:
        """Ingests a file directly from disk: chunk-wise hashing, file-to-file
        raw storage, and path-based parsing. The raw file is never materialized
        as one bytes object."""
        return self._ingest(file_name, source_description, source_path=Path(file_path))

    def _ingest(
        self,
        file_name: str,
        source_description: Optional[str],
        *,
        file_bytes: Optional[bytes] = None,
        source_path: Optional[Path] = None,
    ) -> Tuple[Document, List[TradingRule]]:
        """Shared ingestion body; exactly one of file_bytes/source_path is set."""
        # (name, category) of human-approved rules preserved during recovery —
        # empty for fresh documents; populated only on the recovery path.
        approved_rule_keys: set = set()
        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        if source_path is not None:
            file_size = source_path.stat().st_size
        else:
            file_size = len(file_bytes)

        # Validate file size (max 250MB to support large trading course books)
        if file_size > 250 * 1024 * 1024:
            raise ValueError("File exceeds maximum allowed size of 250MB.")

        # Keep uploads inside the raw-data directory; never trust path components from a filename.
        safe_file_name = Path(str(file_name).replace("\\", "/")).name
        ext = Path(safe_file_name).suffix.lower()

        # Compute cryptographic content hash (chunk-wise for path sources)
        if source_path is not None:
            content_sha256 = self._hash_file_chunked(source_path)
        else:
            content_sha256 = hashlib.sha256(file_bytes).hexdigest()

        # Idempotency check: query existing document record by cryptographic content hash.
        # Deterministic winner: lowest id (oldest row) when duplicate hash rows
        # exist (e.g. a failed + a completed twin from a pre-fix race) — an
        # unordered .first() made recovery nondeterministic.
        existing_doc = self.db.query(Document).filter(
            Document.content_sha256 == content_sha256,
        ).order_by(Document.id).first()

        # Provenance-verified fallback for legacy records where content_sha256 is null:
        # Crucial: Never assume equality based solely on filename and file size.
        # Equality must be cryptographically proven against stored file bytes or exact raw_text.
        # If equality cannot be proven, the incoming content MUST NOT be merged or attributed to legacy.
        if not existing_doc:
            legacy_candidates = self.db.query(Document).filter(
                Document.file_name == safe_file_name,
                Document.file_size_bytes == file_size,
                Document.content_sha256.is_(None),
                Document.processing_status == "completed"
            ).all()

            for legacy_doc in legacy_candidates:
                is_identical = False
                legacy_file_path = settings.DATA_RAW_DIR / f"{legacy_doc.document_id}_{safe_file_name}"

                # 1. Cryptographically verify exact raw bytes on disk if available
                if legacy_file_path.is_file():
                    try:
                        if self._hash_file_chunked(legacy_file_path) == content_sha256:
                            is_identical = True
                    except Exception as e:
                        logger.warning(f"Could not verify legacy file on disk: {e}")

                # 2. Cryptographically verify against raw_text if text-based and exact match
                if not is_identical and legacy_doc.raw_text:
                    try:
                        raw_bytes = legacy_doc.raw_text.encode("utf-8")
                        if hashlib.sha256(raw_bytes).hexdigest() == content_sha256:
                            is_identical = True
                    except Exception:
                        pass

                if is_identical:
                    # Content equality strictly proven: safe to backfill hash.
                    # Backfill EVERY proven-identical legacy row (not just the
                    # first): a remaining NULL-hash twin would re-trigger this
                    # legacy scan on every future re-ingest of the same bytes.
                    legacy_doc.content_sha256 = content_sha256
                    self.db.commit()
                    if existing_doc is None:
                        existing_doc = legacy_doc

        if existing_doc:
            # Check whether existing document is genuinely complete (has persisted chunks or was empty)
            existing_chunks_count = self.db.query(KnowledgeChunk).filter(
                KnowledgeChunk.document_id == existing_doc.document_id
            ).count()
            is_complete = (
                existing_doc.processing_status == "completed"
                and (existing_chunks_count > 0 or existing_doc.pages_count == 0)
            )
            # Item 35: an empty/blank source is a TERMINAL completed state —
            # there is genuinely nothing to chunk, so recovery would re-parse
            # it forever. The explicit marker records WHY it has zero chunks.
            if (
                not is_complete
                and existing_doc.processing_status == "completed"
                and existing_chunks_count == 0
                and not (existing_doc.raw_text or "").strip()
            ):
                existing_doc.error_message = "NO_EXTRACTABLE_CONTENT: source yielded no usable text chunks."
                self.db.commit()
                logger.info(
                    f"Document {existing_doc.document_id} is a blank/empty source "
                    "(terminal completed state); returning it without recovery."
                )
                return existing_doc, []
            if is_complete:
                logger.info(f"Document with hash {content_sha256} already ingested. Returning existing record.")
                existing_rules = self.db.query(TradingRule).filter(TradingRule.source_document == existing_doc.document_id).all()
                return existing_doc, existing_rules

            logger.warning(
                f"Document {existing_doc.document_id} with hash {content_sha256} found in incomplete state "
                f"(status={existing_doc.processing_status}, chunks={existing_chunks_count}). Initiating recovery."
            )
            # Reusing existing_doc to prevent duplicate Document records
            document_id = existing_doc.document_id
            doc_record = existing_doc
            # HUMAN-APPROVED rules are governance state, not processing
            # artifacts: recovery must preserve them with their identity
            # (rule_id + approval_status). Deleting everything would silently
            # de-approve rules the user already approved if a later boundary
            # failed after their approval. Only PENDING/REJECTED rules are
            # rebuilt; fresh extraction below skips exact duplicates of the
            # preserved approved rules.
            approved_rules = self.db.query(TradingRule).filter(
                TradingRule.source_document == document_id,
                TradingRule.approval_status == ApprovalStatus.APPROVED.value,
            ).all()
            approved_rule_keys = {(r.name, r.category) for r in approved_rules}
            # Clean up partial chunks/rules before re-populating. The matched
            # ORM objects are expunged FIRST: after a mid-ingest rollback,
            # Query.delete() removes the DB rows but does NOT evict the
            # identity-map entries, and the re-created rows reuse the freed
            # autoincrement PKs — SQLAlchemy then warns "Identity map already
            # had an identity" and can silently bind the stale object.
            stale_chunks = self.db.query(KnowledgeChunk).filter(
                KnowledgeChunk.document_id == document_id).all()
            stale_rules = self.db.query(TradingRule).filter(
                TradingRule.source_document == document_id,
                TradingRule.approval_status != ApprovalStatus.APPROVED.value,
            ).all()
            for stale_obj in (*stale_chunks, *stale_rules):
                self.db.expunge(stale_obj)
            self.db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == document_id).delete()
            self.db.query(TradingRule).filter(
                TradingRule.source_document == document_id,
                TradingRule.approval_status != ApprovalStatus.APPROVED.value,
            ).delete(synchronize_session=False)
            self.db.commit()
        else:
            document_id = f"doc_{uuid.uuid4().hex[:12]}"
            doc_record = None

        # Defense-in-depth for path construction: document_id is
        # server-generated ("doc_" + hex) and safe_file_name is a bare
        # basename (see sanitization above). The centralised helper validates
        # both segments, refuses separators/traversal, and containment-checks
        # the resolved path BEFORE any byte is written.
        from backend.app.core.security import build_contained_path

        storage_path = build_contained_path(
            settings.DATA_RAW_DIR, document_id, safe_file_name)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path is not None:
            shutil.copyfile(source_path, storage_path)
        else:
            storage_path.write_bytes(file_bytes)

        # Parse file (path-based when the source is a file, so raw bytes are
        # never held as one bytes object for the binary formats)
        try:
            if source_path is not None:
                full_text, pages_count, page_chunks = DocumentParser.parse_file_path(safe_file_name, source_path)
            else:
                full_text, pages_count, page_chunks = DocumentParser.parse_file(safe_file_name, file_bytes)
            status = "processing"
            error_msg = None
        except Exception as e:
            logger.error(f"Failed to parse document {file_name}: {e}")
            full_text = ""
            pages_count = 0
            page_chunks = []
            status = "failed"
            error_msg = str(e)

        if doc_record is None:
            # Create new Document record in processing status
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
                content_sha256=content_sha256,
            )
            self.db.add(doc_record)
            try:
                self.db.commit()
                self.db.refresh(doc_record)
            except IntegrityError:
                self.db.rollback()
                logger.info(f"Concurrent ingestion collision detected for hash {content_sha256}. Returning existing document.")
                concurrent_doc = self.db.query(Document).filter(
                    Document.content_sha256 == content_sha256,
                    Document.processing_status == "completed"
                ).first()
                if concurrent_doc:
                    concurrent_rules = self.db.query(TradingRule).filter(TradingRule.source_document == concurrent_doc.document_id).all()
                    return concurrent_doc, concurrent_rules
                raise
        else:
            # Update existing record for recovery
            doc_record.file_name = safe_file_name
            doc_record.file_type = ext.lstrip(".")
            doc_record.file_size_bytes = file_size
            if source_description:
                doc_record.source_description = source_description
            doc_record.processing_status = status
            doc_record.pages_count = pages_count
            doc_record.raw_text = full_text
            doc_record.error_message = error_msg
            doc_record.content_sha256 = content_sha256
            self.db.commit()
            self.db.refresh(doc_record)

        # Persist searchable evidence chunks before extracting rules. Retrieval reads
        # KnowledgeChunk, so saving only Document.raw_text would make uploads invisible.
        created_chunks: List[KnowledgeChunk] = []
        created_rules: List[TradingRule] = []

        if status == "processing":
            try:
                if page_chunks:
                    total_chunks = len(page_chunks)
                    for index, part in enumerate(page_chunks, start=1):
                        # Step 3.2: heavy documents report progress instead of
                        # appearing to hang during a long page-by-page ingest.
                        if index == 1 or index % 25 == 0 or index == total_chunks:
                            logger.info(
                                "Ingestion progress for %s: chunk %d/%d",
                                safe_file_name, index, total_chunks,
                            )
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

                        loc_type = part.get("location_type")
                        loc_ref = part.get("location_reference")
                        if loc_type and loc_ref:
                            location_type = loc_type
                            location_reference = loc_ref
                        elif sheet_name:
                            location_type = "sheet"
                            location_reference = f"Sheet: {str(sheet_name)[:100]}"
                        else:
                            location_type = "page"
                            location_reference = f"Page {page_number or index}"

                        first_line = next((line.strip() for line in chunk_text.splitlines() if line.strip()), "")
                        title = part.get("title") or (first_line[:255] if first_line else safe_file_name)
                        chunk_params = dict(part.get("parameters") or {})
                        if "page_type" in part and "page_type" not in chunk_params:
                            chunk_params["page_type"] = part["page_type"]
                        if "is_semantic_content" in part and "is_semantic_content" not in chunk_params:
                            chunk_params["is_semantic_content"] = part["is_semantic_content"]
                        if "image_count" in part and "image_count" not in chunk_params:
                            chunk_params["image_count"] = part["image_count"]
                        if "image_dimensions" in part and "image_dimensions" not in chunk_params:
                            chunk_params["image_dimensions"] = part["image_dimensions"]

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
                            parameters=chunk_params,
                            exceptions_and_limits=[],
                        )
                        self.db.add(chunk)
                        created_chunks.append(chunk)

                    self.db.commit()
                    self.retriever.refresh()

                    rule_creates = RuleExtractor.extract_from_pages(document_id, page_chunks)
                    for rc in rule_creates:
                        # Recovery only: a fresh PENDING clone of a rule the
                        # user already APPROVED (same name+category on the
                        # same document) must not be re-created — the approved
                        # original was preserved above with its identity.
                        if (rc.name, rc.category) in approved_rule_keys:
                            continue
                        rule_id = f"rule_{uuid.uuid4().hex[:8]}"
                        rule_record = TradingRule(
                            rule_id=rule_id,
                            name=rc.name,
                            category=rc.category.value,
                            suggested_direction=rc.suggested_direction,
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

                # Transition status to completed only after chunks and rules are committed
                doc_record.processing_status = "completed"
                if created_chunks:
                    doc_record.error_message = None
                else:
                    # Explicit terminal state for sources that genuinely yield
                    # no extractable text: without this marker the doc would be
                    # treated as "incomplete" forever and re-parsed on every
                    # re-ingest (item 35).
                    doc_record.error_message = "NO_EXTRACTABLE_CONTENT: source yielded no usable text chunks."
                self.db.commit()
                self.db.refresh(doc_record)
            except Exception as e:
                self.db.rollback()
                logger.error(f"Failed during chunk or rule persistence for document {document_id}: {e}")
                doc_record.processing_status = "failed"
                doc_record.error_message = f"Chunk/rule persistence failed: {str(e)}"
                try:
                    self.db.commit()
                except Exception:
                    self.db.rollback()
                raise

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
                    # M4: opposite directions come from the explicit direction
                    # field only; the rule name is never inspected.
                    direction_1 = str(getattr(r1, "direction", "") or "").strip().upper()
                    direction_2 = str(getattr(r2, "direction", "") or "").strip().upper()

                    if {direction_1, direction_2} == {"BUY", "SELL"}:
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

    def batch_review_rules(self, items: List[Dict[str, Any]], actor: str = "USER") -> List[Dict[str, Any]]:
        """Step 3.3: human BATCH review of pending rules — the system proposes
        (suggested_direction), the human disposes.

        Per item: action='approve' sets approval_status=approved; `direction`
        may be 'BUY'/'SELL' (explicit human assignment), 'SUGGESTED' (adopt
        suggested_direction — error if the rule has none), or None/absent
        (rule approved with NO direction: it can trigger but never votes,
        fail-closed). action='reject' requires `reason`. One bad item never
        aborts the batch — every item gets an explicit result.
        """
        results: List[Dict[str, Any]] = []
        for item in items:
            rule_id = str(item.get("rule_id") or "")
            action = str(item.get("action") or "").strip().lower()
            outcome: Dict[str, Any] = {"rule_id": rule_id, "ok": False}
            try:
                if action not in {"approve", "reject"}:
                    raise ValueError("action must be 'approve' or 'reject'")
                rule = self.db.query(TradingRule).filter(TradingRule.rule_id == rule_id).first()
                if not rule:
                    raise ValueError(f"Rule with ID {rule_id} not found.")

                if action == "approve":
                    direction_input = item.get("direction")
                    if direction_input is None:
                        rule.direction = None
                    else:
                        direction_input = str(direction_input).strip().upper()
                        if direction_input == "SUGGESTED":
                            suggested = str(rule.suggested_direction or "").strip().upper()
                            if suggested not in {"BUY", "SELL"}:
                                raise ValueError(
                                    f"Rule {rule_id} has no usable suggested_direction to adopt."
                                )
                            rule.direction = suggested
                        elif direction_input in {"BUY", "SELL"}:
                            rule.direction = direction_input
                        else:
                            raise ValueError(
                                "direction must be 'BUY', 'SELL', 'SUGGESTED', or null"
                            )
                    rule.approval_status = ApprovalStatus.APPROVED.value
                    rule.rejection_reason = None
                    outcome["direction"] = rule.direction
                    outcome["approved"] = True
                else:  # reject
                    reason = str(item.get("reason") or "").strip()
                    if not reason:
                        raise ValueError("reject requires a reason")
                    rule.approval_status = ApprovalStatus.REJECTED.value
                    rule.rejection_reason = reason
                    outcome["rejected"] = True

                self.db.commit()
                self.db.refresh(rule)
                outcome["ok"] = True
                outcome["approval_status"] = rule.approval_status
            except Exception as e:
                self.db.rollback()
                outcome["error"] = str(e)
            results.append(outcome)

        approved_n = sum(1 for r in results if r.get("approved"))
        rejected_n = sum(1 for r in results if r.get("rejected"))
        failed_n = sum(1 for r in results if not r.get("ok"))
        log_security_event(
            "RULE_BATCH_REVIEW",
            {
                "actor": actor,
                "items": len(items),
                "approved": approved_n,
                "rejected": rejected_n,
                "failed": failed_n,
            },
            "INFO",
        )
        return results

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

