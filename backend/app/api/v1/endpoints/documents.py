"""Documents and Knowledge Ingestion Endpoints."""
import asyncio
import os
import tempfile
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, HTTPException
from sqlalchemy.orm import Session, defer

from backend.app.core.logging import logger
from backend.app.db.session import get_db
from backend.app.core.auth import require_api_token
from backend.app.models.document import Document
from backend.app.schemas.document_schema import DocumentListResponse, DocumentResponse
from backend.app.services.knowledge.manager import KnowledgeManager

router = APIRouter()

_MAX_UPLOAD_BYTES = 250 * 1024 * 1024


@router.get("/", response_model=List[DocumentListResponse])
def list_documents(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    _token: str = Depends(require_api_token),
):
    """Lists uploaded trading knowledge documents (M15: token-gated).

    Buglist item 20: the listing deliberately does NOT carry raw_text —
    up to 500 documents' full source text per page inflated both the response
    body and the memory needed to build it. The column is deferred at the ORM
    level so it is never even fetched for this query."""
    return (
        db.query(Document)
        .options(defer(Document.raw_text))
        .order_by(Document.id.desc())
        .limit(limit)
        .all()
    )


async def _write_upload_to_temp(file: UploadFile) -> tuple[str, int]:
    """Streams an upload into a bounded temp file (M11, phase 2).

    Reads stay on the event loop (UploadFile.read is async) while every 1MB
    chunk write is offloaded to a worker thread — the disk write is blocking
    IO and must never run on the loop.
    """
    file_path = None
    size = 0
    try:
        with tempfile.NamedTemporaryFile(dir=tempfile.gettempdir(), suffix="_upload", delete=False) as tmp:
            file_path = tmp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > _MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File exceeds maximum allowed size of 250MB.")
                await asyncio.to_thread(tmp.write, chunk)
    except Exception:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except OSError:
                pass
        raise
    return file_path, size


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    # Buglist item 19: bound the free-text description (persisted as-is).
    description: Optional[str] = Form("Uploaded manual strategy", max_length=500),
    db: Session = Depends(get_db),
    _token: str = Depends(require_api_token),
):
    """Uploads a knowledge file (PDF/TXT/CSV/XLSX), extracts text, and parses rules.

    The upload is streamed into a bounded temp file and ingested straight from
    that path: hashing is chunk-wise, raw storage is a file-to-file copy, and
    the binary parsers open the file by path. The raw file is never held as one
    bytes object. (The extracted text itself is persisted as Document.raw_text,
    so full-text memory for text formats is by design.)
    """
    if file.size is not None and file.size > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds maximum allowed size of 250MB.")

    file_path, size = await _write_upload_to_temp(file)
    try:
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        km = KnowledgeManager(db)
        try:
            # M11: parsing (pymupdf/openpyxl), chunk-wise hashing, the raw
            # storage copy, and the retriever index refresh are pure CPU/IO —
            # run them in a worker thread so a large upload cannot stall the
            # event loop for every other request. The engine is configured
            # with check_same_thread=False and the session is touched only
            # inside this call.
            doc, rules = await asyncio.to_thread(
                km.ingest_document_path,
                file_name=file.filename,
                file_path=file_path,
                source_description=description,
            )
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            logger.exception("Document ingestion failed for %r", file.filename)
            raise HTTPException(status_code=500, detail="Failed to ingest document (internal error logged).")
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except OSError:
                pass

    return {
        "document_id": doc.document_id,
        "file_name": doc.file_name,
        "pages_count": doc.pages_count,
        "processing_status": doc.processing_status,
        "rules_extracted_count": len(rules),
    }
