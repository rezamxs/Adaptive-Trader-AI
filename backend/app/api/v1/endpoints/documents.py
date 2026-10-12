"""Documents and Knowledge Ingestion Endpoints."""
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.models.document import Document
from backend.app.schemas.document_schema import DocumentResponse
from backend.app.services.knowledge.manager import KnowledgeManager

router = APIRouter()


@router.get("/", response_model=List[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    """Lists uploaded trading knowledge documents."""
    return db.query(Document).order_by(Document.id.desc()).all()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    description: Optional[str] = Form("Uploaded manual strategy"),
    db: Session = Depends(get_db)
):
    """Uploads a knowledge file (PDF/TXT/CSV/XLSX), extracts text, and parses rules."""
    max_upload_bytes = 250 * 1024 * 1024
    file_bytes = await file.read(max_upload_bytes + 1)
    if len(file_bytes) > max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds maximum allowed size of 250MB.")
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    km = KnowledgeManager(db)
    try:
        doc, rules = km.ingest_document(
            file_name=file.filename,
            file_bytes=file_bytes,
            source_description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")

    return {
        "document_id": doc.document_id,
        "file_name": doc.file_name,
        "pages_count": doc.pages_count,
        "processing_status": doc.processing_status,
        "rules_extracted_count": len(rules),
    }
