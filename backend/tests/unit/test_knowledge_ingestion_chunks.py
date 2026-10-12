"""Uploaded documents must create retrievable KnowledgeChunk records."""
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import backend.app.models  # noqa: F401 - register all ORM models
from backend.app.core.config import settings
from backend.app.db.base import Base
from backend.app.models.knowledge_chunk import KnowledgeChunk
from backend.app.services.knowledge.manager import KnowledgeManager


def test_ingestion_persists_searchable_chunks_and_sanitizes_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_RAW_DIR", tmp_path)
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            manager = KnowledgeManager(db)
            document, _ = manager.ingest_document(
                r"..\\outside\\lesson.txt",
                b"Slippage is the difference between expected order price and execution price.",
            )

            chunks = db.query(KnowledgeChunk).filter_by(document_id=document.document_id).all()
            assert document.file_name == "lesson.txt"
            assert len(chunks) == 1
            assert chunks[0].source_file == "lesson.txt"
            assert chunks[0].location_type == "page"
            assert "Slippage" in chunks[0].content
            assert len(manager.retriever._chunks_cache) == 1
            assert (tmp_path / f"{document.document_id}_lesson.txt").is_file()
    finally:
        engine.dispose()
