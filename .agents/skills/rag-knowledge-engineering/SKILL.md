---
name: rag-knowledge-engineering
description: Hybrid BM25+semantic RAG for the Persian/English knowledge base — local Qdrant, fastembed multilingual embeddings, dedup-safe indexing, and honest NOT_FOUND behavior.
when_to_use: Building or extending the knowledge retrieval stack; adding documents; comparing BM25 vs semantic quality; rebuilding the vector index.
---

# RAG Knowledge Engineering (Adaptive-Trader-AI)

## Architecture (implemented 2026-11-07)
- BM25 (`KnowledgeRetriever`) stays authoritative; the semantic leg
  (`SemanticRetriever`, `backend/app/services/knowledge/semantic.py`) is
  additive and degrades gracefully (BM25-only) when unavailable.
- Embeddings: fastembed
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim) —
  fa/en cross-lingual verified: cos(fa-size, en-size)=0.814 vs
  cos(fa-size, unrelated)=0.261.
- Store: qdrant-client LOCAL mode at `VECTOR_STORE_DIR` (embedded, no server).
- Fusion: reciprocal rank fusion (k=60) — no score-scale calibration needed.
- Dedup: chunk_id → deterministic point id; re-indexing skips existing
  (real run: second pass = `{'indexed': 0, 'skipped': 102}`).
- Rebuild: close client → purge store dir → reopen → re-index from SQL
  (qdrant-local resurrects points on delete+recreate in the same session —
  verified; the directory purge is the reliable path). SQL documents/chunks
  are never touched by a rebuild.
- NOT_FOUND: unanswered queries return NOT_FOUND (grounded path verified) —
  never fabricate.

## Evidence (2026-11-07, real glossary docx, 102 chunks)
- BM25 failed BOTH cross-lingual queries; hybrid answered them:
  "leverage در فارکس یعنی چه" → BM25 NONE / HYBRID Row 10 (اهرم);
  "اسپرد چیست" → BM25 NONE / HYBRID Row 52.
- Grounded documented answer with citation: file + Row 10 + title.
- Regression tests: `backend/tests/unit/test_rag_semantic.py` (8) — gated on
  the local model cache so CI never downloads the model.

## Rules
- NOTHING in this stack auto-approves rules; classification gates what
  becomes a rule candidate at all (see the trading-safety skill).
