"""initial_schema

Revision ID: 01f57c481926
Revises:
Create Date: 2026-10-10 17:34:37.255878

"""
import os
from typing import Sequence, Union

from alembic import op
from backend.app.db.base import Base
import backend.app.models  # noqa: F401 - register all ORM tables in metadata


# revision identifiers, used by Alembic.
revision: str = '01f57c481926'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create any missing application tables for a fresh or partially initialized DB.

    create_all is idempotent for existing tables. It does not alter columns in tables
    created by an older application version; those changes need dedicated migrations.
    """
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    """Drop every application table this revision's create_all can produce.

    The upgrade is Base.metadata.create_all, so the faithful inverse is
    Base.metadata.drop_all. Running it on a populated database is destructive
    by definition; that is inherent to a schema-creating baseline revision and
    is why real downgrades must only run on disposable/test databases.

    M21 guard: this downgrade refuses to run unless the operator explicitly
    opts in per invocation with ALEMBIC_ALLOW_DESTRUCTIVE_DOWNGRADE=yes. It
    still requires the destination to be pinned via ALEMBIC_DATABASE_URL.
    """
    if os.environ.get("ALEMBIC_ALLOW_DESTRUCTIVE_DOWNGRADE", "").strip().lower() != "yes":
        raise RuntimeError(
            "Refusing destructive downgrade to 'base': the baseline downgrade "
            "is Base.metadata.drop_all and drops every table with all data. "
            "To run it on a disposable database, re-run with "
            "ALEMBIC_ALLOW_DESTRUCTIVE_DOWNGRADE=yes set for this command only."
        )
    Base.metadata.drop_all(bind=op.get_bind())
