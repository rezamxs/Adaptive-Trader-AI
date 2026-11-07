---
name: database-recovery
description: Safe upgrade/backup/rollback/recovery procedure for the Adaptive-Trader-AI SQLite database, with rehearsal-first discipline and verified qdrant-vector recovery.
when_to_use: Before any alembic upgrade on a real database, after a crash, or when the vector store must be rebuilt without touching documents.
---

# Database Recovery (Adaptive-Trader-AI)

## Iron rules
- NEVER touch the production DB directly. Rehearse on a THREE-FILE copy first
  (`db`, `db-wal`, `db-shm` — a db-only copy can HIDE committed data inside the
  WAL: verified 2026-11-07 when a duplicate content_sha256 pair was visible
  only in the full copy).
- Read-only inspection uses `sqlite3.connect("file:...?mode=ro", uri=True)`.

## Procedure (each step executed for real on 2026-11-07; see Evidence)
1. Backup: stop the app, `PRAGMA wal_checkpoint(TRUNCATE)`, copy all three
   files, run `PRAGMA integrity_check` on the copy — must print `ok`.
2. Rehearse upgrade: `ALEMBIC_DATABASE_URL="sqlite:///./copy.db" alembic upgrade head`
   then verify `alembic current`, FK list (`PRAGMA foreign_key_list`), and that
   row counts match the pre-upgrade snapshot.
3. Rehearse rollback: `alembic downgrade <prod_version>` → integrity ok →
   re-upgrade (roundtrip). The dshq downgrade RESTORES NULLed duplicate hashes
   from the stored raw files — a real run restored 1 value.
4. Real migration only after steps 1-3 are green (owner authorization).
5. Vector store recovery: vectors are DERIVED data. Deleting the local store
   directory and re-indexing from SQL restores everything without touching
   documents (`SemanticRetriever.rebuild_index()` — real run: 16/16 re-indexed,
   SQL chunks intact).

## Evidence (2026-11-07)
- prod copy: 8 revisions upgraded to 20261106_bthr, counts preserved
  (6/318/47/78/168/609), FK probe rejects orphans, full rollback + roundtrip ok.
- qdrant-local quirk found and documented: `delete_collection` + recreate on
  the SAME client resurrects old points — the rebuild must close the client,
  purge the directory, and reopen.
