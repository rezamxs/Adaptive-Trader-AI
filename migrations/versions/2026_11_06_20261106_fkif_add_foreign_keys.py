"""Add database-level foreign keys (buglist item 13).

Until now the inter-record relations existed only as code conventions — the
database accepted any dangling reference. This revision adds FK constraints
for the four relations with a clean delete policy:

- knowledge_chunks.document_id    -> documents.document_id      ON DELETE CASCADE
  (chunks are owned by their document; recovery already deletes them together)
- trading_rules.source_document   -> documents.document_id      ON DELETE SET NULL
  (an approved rule stays alive even if its source document row vanishes)
- backtest_runs.model_id          -> ml_models.model_id         ON DELETE SET NULL
- ml_walk_forward_runs.model_id   -> ml_models.model_id         ON DELETE SET NULL
  (the M8 evidence binding; SET NULL when the model row is removed)

demo_orders.signal_id deliberately gets NO FK: the column is NOT NULL and the
30-day retention prunes trade_signals before demo_orders, so every available
policy is wrong — CASCADE would delete an un-pruned order (breaking buglist
item 2's protection), RESTRICT would fail the retention pass, and SET NULL
would violate the NOT NULL contract. The relation stays code-enforced there,
documented here.

SQLite cannot ADD CONSTRAINT, so each table is rebuilt via batch_alter_table.
Orphan rows are cleaned FIRST: with foreign_keys enforcement a batch recreate
fails on any dangling value, and the dangling values are unreachable garbage
anyway. The cleanup is bounded to exactly the four referencing columns.

Inspector-guarded like the sibling revisions: the FKs are skipped per table
when the constraint already exists (fresh chains built from Base.metadata
already carry them).

Revision ID: 20261106_fkif
Revises: 20261104_mlwfr
Create Date: 2026-11-06
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20261106_fkif"
down_revision: Union[str, None] = "20261104_mlwfr"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_FK_SPECS = (
    # (table, column, parent_table, parent_column, ondelete, fk_name)
    ("knowledge_chunks", "document_id", "documents", "document_id", "CASCADE",
     "fk_knowledge_chunks_document_id"),
    ("trading_rules", "source_document", "documents", "document_id", "SET NULL",
     "fk_trading_rules_source_document"),
    ("backtest_runs", "model_id", "ml_models", "model_id", "SET NULL",
     "fk_backtest_runs_model_id"),
    ("ml_walk_forward_runs", "model_id", "ml_models", "model_id", "SET NULL",
     "fk_ml_walk_forward_runs_model_id"),
)


def _fk_exists(inspector, table: str, parent_table: str, column: str,
               parent_column: str) -> bool:
    """True when an equivalent FK already exists on the table.

    SQLite reports unnamed FKs (name is None) for constraints coming from
    Base.metadata, so the check must compare the FK SHAPE — referred table +
    columns — not the constraint name (checking by name would double-create
    the constraint on fresh chains).
    """
    try:
        fks = inspector.get_foreign_keys(table)
    except Exception:
        return False
    for fk in fks:
        if (
            fk.get("referred_table") == parent_table
            and column in (fk.get("constrained_columns") or [])
            and parent_column in (fk.get("referred_columns") or [])
        ):
            return True
    return False


def _clean_orphans(conn, table: str, column: str, parent_table: str,
                   parent_column: str, set_null: bool) -> int:
    """Removes (or NULLs) rows whose reference no longer resolves."""
    orphan_ids = conn.execute(
        sa.text(
            f"SELECT {table}.id FROM {table} LEFT JOIN {parent_table} "
            f"ON {table}.{column} = {parent_table}.{parent_column} "
            f"WHERE {table}.{column} IS NOT NULL "
            f"AND {parent_table}.{parent_column} IS NULL"
        )
    ).fetchall()
    if not orphan_ids:
        return 0
    if set_null:
        conn.execute(sa.text(
            f"UPDATE {table} SET {column} = NULL WHERE id IN "
            f"({','.join(str(r[0]) for r in orphan_ids)})"
        ))
    else:
        conn.execute(sa.text(
            f"DELETE FROM {table} WHERE id IN "
            f"({','.join(str(r[0]) for r in orphan_ids)})"
        ))
    return len(orphan_ids)


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    conn = op.get_bind()

    # Orphan cleanup BEFORE the batch recreates (FK enforcement would fail the
    # table rebuild on any dangling value). knowledge_chunks rows are owned
    # data -> deleted; the three nullable references are NULLed instead.
    cleaned = {}
    for table, column, parent_table, parent_column, ondelete, _fk in _FK_SPECS:
        cleaned[f"{table}.{column}"] = _clean_orphans(
            conn, table, column, parent_table, parent_column,
            set_null=(ondelete == "SET NULL"),
        )
    if any(cleaned.values()):
        conn.commit()
        report = {k: v for k, v in cleaned.items() if v}
        print(f"FK migration orphan cleanup: {report}")

    for table, column, parent_table, parent_column, ondelete, fk_name in _FK_SPECS:
        if _fk_exists(inspector, table, parent_table, column, parent_column):
            continue  # fresh chain: Base.metadata already built the constraint
        with op.batch_alter_table(table) as batch_op:
            batch_op.create_foreign_key(
                fk_name, parent_table, [column], [parent_column],
                ondelete=ondelete,
            )


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    for table, column, parent_table, parent_column, _ondelete, fk_name in _FK_SPECS:
        if not _fk_exists(inspector, table, parent_table, column, parent_column):
            continue
        # SQLite cannot drop an unnamed (metadata-built) FK constraint by
        # name — only the constraint this revision itself created carries the
        # explicit name; unnamed ones are skipped with a note.
        try:
            with op.batch_alter_table(table) as batch_op:
                batch_op.drop_constraint(fk_name, type_="foreignkey")
        except Exception:
            print(
                f"downgrade: unnamed FK on {table}.{column} (from Base.metadata) "
                "cannot be dropped by name on SQLite; left in place."
            )
