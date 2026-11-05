"""Pydantic schemas for Document Management."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


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

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(DocumentBase):
    """Buglist item 20: the LIST endpoint must not carry Document.raw_text —
    the full source text of up to 500 documents per page inflated the response
    body and the response-construction memory for a listing that only displays
    metadata. Single-document reads keep raw_text via DocumentResponse."""
    document_id: str
    processing_status: str
    pages_count: int
    error_message: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
