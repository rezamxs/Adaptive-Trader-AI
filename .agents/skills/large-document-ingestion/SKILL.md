---
name: large-document-ingestion
description: Staged, memory-bounded ingestion of very large educational PDFs with page-referenced chunks, batched embeddings, and RAM telemetry on Windows.
when_to_use: Ingesting PDFs of 50MB+; never load the whole file into RAM; embed and index in batches then free.
---

# Large Document Ingestion (Adaptive-Trader-AI)

## Iron rules
- Open the PDF BY PATH (`pymupdf.open(filename=...)`) — never `read_bytes()`.
- Process in page batches (8 pages per batch measured well) and DROP each
  batch's chunks after embedding+indexing.
- Every chunk keeps an explicit page reference (`location_reference`,
  `page_number`) so retrieval can cite the source page.
- Use a TEMP vector store for experiments; the owner indexes into the real
  store only on their own decision.

## RAM telemetry on Windows (no extra deps)
`ctypes` + `GetProcessMemoryInfo` (psapi) → `WorkingSetSize`; sample after
every batch and track the max.

## Evidence (2026-11-07, real file: 159.9MB price-action PDF, 89 pages)
- pages processed: 89/89, extraction errors: 0
- chunks indexed (each citing its page): 88
- wall time: 21.8s; RAM: base 776MB → max 883MB (delta ~107MB — bounded; the
  160MB file never entered RAM as one object)
- sample classification of the first 40 pages: 38 analytical_concept,
  1 ambiguous — a price-action textbook teaches CONCEPTS, not rules; the
  classifier kept it out of the rule registry.

## OCR note
Tesseract was NOT installed on this machine (verified: no binary on PATH, no
UB-Mannheim default dir). The target PDF is text-based (scan ratio 0.01) so
OCR was not needed for it. OCR support is INCOMPLETE until the owner installs
Tesseract (UB-Mannheim build) with `fas`+`eng` traineddata; the parser's
`OCR_ENABLED` flag is already wired.
