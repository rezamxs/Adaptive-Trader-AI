"""Pydantic schemas for Document Management."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class DocumentBase(BaseModel):
    file_name: str
    file_type: str
    file_size_bytes: int
    source_description: Optional[str] = None


class DocumentCreate(DocumentBase):
    pass


class DocumentResponse(DocumentBase):
    document_id: str
    processing_status: str
    pages_count: int
    raw_text: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
