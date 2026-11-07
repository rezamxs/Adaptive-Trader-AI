"""Semantic retrieval + hybrid fusion on top of the existing BM25 retriever.

Design constraints (mission 2026-11-07):
- Qdrant runs in LOCAL mode (embedded, persistent directory) — NO server.
- fastembed provides the fa/en multilingual embeddings
  (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, 384-dim).
- BM25 (KnowledgeRetriever) stays the primary/authoritative scorer; the
  semantic path is additive. If fastembed/qdrant or the model cache is
  unavailable, everything degrades gracefully to BM25-only with a warning —
  the knowledge pipeline never hard-fails on the semantic layer.
- No duplicate indexing: chunk_id is the deterministic Qdrant point id;
  already-indexed chunks are skipped via client.retrieve.
- rebuild_index() drops the collection and re-embeds EVERYTHING from SQL —
  original documents/chunks are never touched (vectors are derived data).
"""
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.core.config import settings

logger = logging.getLogger("adaptive_trader")

_COLLECTION = "knowledge_chunks"
_DIM = 384
_RRF_K = 60  # standard reciprocal-rank-fusion constant


def _model_cached() -> bool:
    """True when the multilingual model is already in the fastembed cache."""
    import tempfile

    cache = Path(tempfile.gettempdir()) / "fastembed_cache"
    return any(
        "paraphrase-multilingual" in d.name.lower()
        for d in cache.glob("models--*")
    ) if cache.exists() else False


def _chunk_point_id(chunk_id: str) -> int:
    """Deterministic positive int64 point id from the chunk_id string."""
    return int(uuid.uuid5(uuid.NAMESPACE_URL, f"chunk:{chunk_id}").int >> 64) & 0x7FFFFFFFFFFFFFFF


class SemanticRetriever:
    """fa/en semantic search over knowledge chunks with a local Qdrant store."""

    def __init__(self, db=None):
        self.db = db
        self.available = False
        self.unavailable_reason: Optional[str] = None
        self._model = None
        self._client = None
        try:
            from fastembed import TextEmbedding
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._TextEmbedding = TextEmbedding
            self._QdrantClient = QdrantClient
            self._Distance = Distance
            self._VectorParams = VectorParams
        except Exception as exc:  # pragma: no cover - depends on env
            self.unavailable_reason = f"import failed: {type(exc).__name__}: {exc}"
            return

        if not getattr(settings, "SEMANTIC_RETRIEVAL_ENABLED", True):
            self.unavailable_reason = "disabled: SEMANTIC_RETRIEVAL_ENABLED=False"
            return
        if not _model_cached():
            # Deliberately NOT downloading: a fresh clone/CI must never pull
            # ~700MB on an ingestion side-path. The owner builds the index
            # explicitly (scripts/rag_build_index.py), which caches the model.
            self.unavailable_reason = (
                "multilingual embedding model not cached; run scripts/rag_build_index.py "
                "once to download and build the index")
            return
        vector_dir = Path(getattr(settings, "VECTOR_STORE_DIR", Path("data/vector_store")))
        try:
            vector_dir.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(vector_dir))
            if not self._client.collection_exists(_COLLECTION):
                self._client.create_collection(
                    _COLLECTION,
                    vectors_config=self._VectorParams(size=_DIM, distance=self._Distance.COSINE),
                )
            self._model = TextEmbedding(getattr(settings, "EMBEDDING_MODEL", ""))
            self.available = True
        except Exception as exc:  # model download/network failure etc.
            self.unavailable_reason = f"init failed: {type(exc).__name__}: {exc}"
            self._client = None
            self._model = None

    # ------------------------------------------------------------- indexing
    def index_chunks(self, chunks: List[Any]) -> Dict[str, int]:
        """Indexes chunks (skipping already-indexed ones). Returns counters."""
        if not self.available:
            return {"indexed": 0, "skipped": 0}
        from qdrant_client.models import PointStruct

        fresh: List[Any] = []
        existing_ids = set()
        ids_to_check = [_chunk_point_id(self._attr(c, "chunk_id")) for c in chunks]
        try:
            found = self._client.retrieve(_COLLECTION, ids=ids_to_check, with_payload=False)
            existing_ids = {p.id for p in found}
        except Exception:
            existing_ids = set()
        for c in chunks:
            if _chunk_point_id(self._attr(c, "chunk_id")) not in existing_ids:
                fresh.append(c)

        indexed = 0
        BATCH = 32
        for i in range(0, len(fresh), BATCH):
            batch = fresh[i:i + BATCH]
            texts = [
                f"{self._attr(c, 'title') or ''}\n{self._attr(c, 'content') or ''}"[:2000] for c in batch
            ]
            vectors = list(self._model.embed(texts, batch_size=BATCH))
            points = [
                PointStruct(
                    id=_chunk_point_id(self._attr(c, "chunk_id")),
                    vector=[float(x) for x in vec],
                    payload={
                        "chunk_id": self._attr(c, "chunk_id"),
                        "document_id": self._attr(c, "document_id"),
                        "source_file": self._attr(c, "source_file"),
                        "location_type": self._attr(c, "location_type"),
                        "location_reference": self._attr(c, "location_reference"),
                        "page_number": self._attr(c, "page_number"),
                        "title": self._attr(c, "title") or "",
                    },
                )
                for c, vec in zip(batch, vectors)
            ]
            self._client.upsert(_COLLECTION, points=points)
            indexed += len(points)
        return {"indexed": indexed, "skipped": len(chunks) - indexed}

    @staticmethod
    def _attr(chunk: Any, name: str) -> Any:
        """Accepts ORM KnowledgeChunk objects AND plain dicts (the staged
        big-file pipeline builds dict chunks without a DB)."""
        if isinstance(chunk, dict):
            return chunk.get(name)
        return getattr(chunk, name, None)

    def rebuild_index(self) -> Dict[str, int]:
        """Drops the vector store and re-indexes EVERY chunk from SQL.
        Documents/chunks in SQL are untouched (vectors are derived data).

        qdrant-local quirk (verified 2026-11-07): delete_collection +
        create_collection on the SAME client resurrects the old points, so a
        true rebuild closes the client, purges the storage directory, and
        reopens — deterministic and local-only."""
        if not self.available or self.db is None:
            return {"indexed": 0}
        from backend.app.models.knowledge_chunk import KnowledgeChunk

        vector_dir = Path(getattr(settings, "VECTOR_STORE_DIR", Path("data/vector_store")))
        try:
            self._client.close()
        except Exception:
            pass
        import shutil as _shutil

        _shutil.rmtree(vector_dir, ignore_errors=True)
        vector_dir.mkdir(parents=True, exist_ok=True)
        self._client = self._QdrantClient(path=str(vector_dir))
        self._client.create_collection(
            _COLLECTION,
            vectors_config=self._VectorParams(size=_DIM, distance=self._Distance.COSINE),
        )
        chunks = self.db.query(KnowledgeChunk).all()
        return self.index_chunks(chunks)

    def count(self) -> int:
        if not self.available:
            return 0
        try:
            return self._client.count(_COLLECTION, exact=True).count
        except Exception:
            return 0

    # ------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 5, source_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Semantic search returning the SAME citation shape as the BM25 path."""
        if not self.available or not query.strip():
            return []
        vec = next(iter(self._model.embed([query[:1000]], batch_size=1)))
        flt = None
        if source_file:
            from qdrant_client.models import FieldCondition, Filter, MatchText

            flt = Filter(must=[FieldCondition(key="source_file", match=MatchText(text=source_file))])
        hits = self._client.query_points(
            _COLLECTION, query=[float(x) for x in vec], limit=top_k, query_filter=flt
        ).points
        return [
            {
                "chunk_id": h.payload.get("chunk_id"),
                "source_file": h.payload.get("source_file"),
                "location_reference": h.payload.get("location_reference"),
                "page_number": h.payload.get("page_number"),
                "title": h.payload.get("title"),
                "score": round(float(h.score), 4),
                "retriever": "semantic",
            }
            for h in hits
        ]


class HybridRetriever:
    """BM25 (authoritative lexical scorer) + semantic vectors, fused with RRF.

    Both legs return the BM25-shaped result dicts; fusion is rank-based
    (reciprocal rank fusion) so the two score scales never need calibrating.
    With the semantic leg unavailable this is a transparent BM25 pass-through.
    """

    def __init__(self, db, semantic: Optional[SemanticRetriever] = None):
        from backend.app.services.knowledge.retriever import KnowledgeRetriever

        self.bm25 = KnowledgeRetriever(db)
        self.semantic = semantic or SemanticRetriever(db)
        if not self.semantic.available:
            logger.warning(
                "Semantic retrieval unavailable (%s); hybrid falls back to BM25-only.",
                self.semantic.unavailable_reason,
            )

    def refresh(self) -> None:
        self.bm25.refresh()

    def search(self, query: str, top_k: int = 5, source_file: Optional[str] = None) -> Dict[str, Any]:
        """Returns {'results': [...], 'retrievers': [...], 'fusion': 'rrf'|'bm25_only'}."""
        bm25_hits = self.bm25.search(query=query, top_k=top_k * 2, source_file=source_file)
        used: List[str] = ["bm25"]
        if not self.semantic.available:
            return {"results": bm25_hits[:top_k], "retrievers": used, "fusion": "bm25_only"}

        sem_hits = self.semantic.search(query=query, top_k=top_k * 2, source_file=source_file)
        used.append("semantic")

        by_key: Dict[str, Dict[str, Any]] = {}
        rrf: Dict[str, float] = {}
        for hits in (bm25_hits, sem_hits):
            for rank, hit in enumerate(hits, start=1):
                key = hit.get("chunk_id") or f"{hit.get('source_file')}|{hit.get('location_reference')}"
                if key not in by_key:
                    merged = dict(hit)
                    body = None
                    if hit.get("retriever") == "semantic":
                        body = self._load_chunk(hit.get("chunk_id"))
                        if body is None:
                            continue
                        merged.update({
                            "content": body.content,
                            "parameters": body.parameters or {},
                            "exceptions_and_limits": body.exceptions_and_limits or [],
                            "is_semantic_content": True,
                        })
                    by_key[key] = merged
                rrf[key] = rrf.get(key, 0.0) + 1.0 / (_RRF_K + rank)

        fused = sorted(by_key.values(), key=lambda r: rrf.get(r.get("chunk_id") or "", 0.0), reverse=True)
        out = []
        for r in fused[:top_k]:
            r = dict(r)
            r["rrf_score"] = round(rrf.get(r.get("chunk_id") or "", 0.0), 6)
            r.setdefault("retriever", "bm25")
            out.append(r)
        return {"results": out, "retrievers": used, "fusion": "rrf"}

    def _load_chunk(self, chunk_id: Optional[str]):
        if not chunk_id or self.semantic.db is None:
            return None
        from backend.app.models.knowledge_chunk import KnowledgeChunk

        return (
            self.semantic.db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.chunk_id == chunk_id)
            .first()
        )
