"""One-time (and re-runnable) builder for the Hybrid RAG vector index.

Downloads/caches the multilingual embedding model EXPLICITLY (owner action —
never a hidden side effect of ingestion), then indexes every knowledge chunk
in the configured DATABASE_URL into the local Qdrant store.

Usage (from the repo root):
    python scripts/rag_build_index.py            # build/update the index
    python scripts/rag_build_index.py --rebuild  # purge the store, re-index all

The database documents/chunks are never modified; the vector store is derived
data. BM25 keeps working with or without this index.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import sessionmaker  # noqa: E402

import backend.app.models  # noqa: F401,E402
from backend.app.core.config import settings  # noqa: E402
from backend.app.db.session import engine  # noqa: E402
from backend.app.models.knowledge_chunk import KnowledgeChunk  # noqa: E402
from backend.app.services.knowledge.semantic import SemanticRetriever  # noqa: E402


def main() -> int:
    rebuild = "--rebuild" in sys.argv
    session = sessionmaker(bind=engine)()
    sem = SemanticRetriever(session)
    if not sem.available:
        print(f"semantic retriever unavailable: {sem.unavailable_reason}")
        return 1
    chunks = session.query(KnowledgeChunk).all()
    if rebuild:
        stats = sem.rebuild_index()
        print(f"REBUILT: indexed {stats['indexed']} chunk(s)")
    else:
        stats = sem.index_chunks(chunks)
        print(f"UPDATED: indexed {stats['indexed']}, already present {stats['skipped']}")
    print(f"vectors in store: {sem.count()} | SQL chunks: {len(chunks)}")
    sem._client.close()
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
