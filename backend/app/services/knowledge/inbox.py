"""Step 3.1: automatic ingestion of documents dropped into a watched inbox folder.

The owner drops PDF/DOCX/XLSX/ZIP/TXT/CSV/MD files into
`data/document_inbox/`; at application startup the folder is scanned and every
supported file is ingested through the same transactional KnowledgeManager
pipeline the upload endpoint uses (idempotent via content_sha256 — re-scanning
an already-ingested file returns the SAME document_id and never duplicates
chunks or rules). On success the file moves to `<inbox>/processed/`; on
failure to `<inbox>/failed/`. Nothing is ever silently skipped: unsupported
extensions are counted and reported, and every failure is logged with its
cause.
"""
import shutil
from pathlib import Path
from typing import Dict, Optional, Set

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.services.knowledge.manager import KnowledgeManager

SUPPORTED_INBOX_EXTENSIONS: Set[str] = {".pdf", ".docx", ".zip", ".xlsx", ".txt", ".csv", ".md"}

# Mirrors the upload endpoint's cap: an inbox file above this size is reported
# as skipped (never silently dropped) and left in place for the owner.
_MAX_INBOX_FILE_BYTES = 250 * 1024 * 1024


def process_inbox(db, inbox_dir: Optional[Path] = None) -> Dict:
    """Scans the inbox folder once, ingesting every supported file.

    Returns a summary dict (and logs the per-file progress) — a single bad
    file never aborts the scan; a file whose ingestion fails moves to failed/.
    """
    folder = Path(inbox_dir or settings.DOCUMENT_INBOX_DIR)
    summary: Dict = {
        "inbox": str(folder),
        "found": 0,
        "ingested": 0,
        "failed": 0,
        "skipped_unsupported": 0,
        "skipped_too_large": 0,
        "documents": [],
    }
    if not folder.exists():
        summary["note"] = "inbox folder does not exist (nothing to ingest)"
        return summary

    for entry in sorted(folder.iterdir()):
        if not entry.is_file():
            continue
        summary["found"] += 1
        suffix = entry.suffix.lower()
        if suffix not in SUPPORTED_INBOX_EXTENSIONS:
            summary["skipped_unsupported"] += 1
            logger.warning("Inbox skip (unsupported extension %r): %s", suffix, entry.name)
            continue
        if entry.stat().st_size > _MAX_INBOX_FILE_BYTES:
            summary["skipped_too_large"] += 1
            logger.warning(
                "Inbox skip (file exceeds %d-byte cap): %s", _MAX_INBOX_FILE_BYTES, entry.name)
            continue

        result = _ingest_one(db, entry)
        if result["ok"]:
            summary["ingested"] += 1
        else:
            summary["failed"] += 1
        summary["documents"].append(result)

    if summary["found"]:
        logger.info(
            "Inbox scan: %d file(s), %d ingested, %d failed, %d unsupported, %d too large",
            summary["found"], summary["ingested"], summary["failed"],
            summary["skipped_unsupported"], summary["skipped_too_large"],
        )
    return summary


def _ingest_one(db, entry: Path) -> Dict:
    """Ingests one inbox file and moves it to processed/ or failed/.

    The move happens only after the transactional ingest returns; a crash
    before the move leaves the file in place, and the next scan retries it
    (idempotent via content_sha256).
    """
    from backend.app.services.knowledge.manager import KnowledgeManager as _KM

    result: Dict = {"file": entry.name, "ok": False}
    try:
        doc, rules = _KM(db).ingest_document_path(
            entry.name, entry, source_description="document inbox"
        )
        result.update({
            "ok": True,
            "document_id": doc.document_id,
            "processing_status": doc.processing_status,
            "rules_extracted": len(rules),
            "page_type_summary": get_page_type_summary(db, doc.document_id),
        })
        _move(entry, entry.parent / "processed" / entry.name)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        logger.error("Inbox ingestion FAILED for %s: %s", entry.name, e)
        try:
            _move(entry, entry.parent / "failed" / entry.name)
        except Exception as move_error:
            result["move_error"] = str(move_error)
    logger.info(
        "Inbox file %s -> %s (rules: %s)",
        entry.name, "ingested" if result.get("ok") else "FAILED", result.get("rules_extracted"),
    )
    return result


def _move(entry: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        stem, suffix = entry.stem, entry.suffix
        target = target.with_name(f"{stem}_{uuid4hex()}{suffix}")
    shutil.move(str(entry), str(target))


def uuid4hex() -> str:
    import uuid

    return uuid.uuid4().hex[:6]


def get_page_type_summary(db, document_id: str) -> Dict[str, list]:
    """Aggregates each persisted chunk's page_type (image-only, vector-only,
    blank, text) into a per-document report: {page_type: [page numbers]}.

    Step 3.2: image-only (scanned) pages are REPORTED separately — they never
    silently disappear and never produce a rule; the extractor skips chunks
    without text by construction.
    """
    from backend.app.models.knowledge_chunk import KnowledgeChunk

    chunks = (
        db.query(KnowledgeChunk.page_number, KnowledgeChunk.parameters)
        .filter(KnowledgeChunk.document_id == document_id)
        .all()
    )
    summary: Dict[str, list] = {}
    for page_number, parameters in chunks:
        page_type = "TEXT_PAGE"
        if isinstance(parameters, dict):
            page_type = str(parameters.get("page_type") or "TEXT_PAGE")
        summary.setdefault(page_type, []).append(page_number)
    for key in summary:
        summary[key] = sorted(set(summary[key]))
    return summary
