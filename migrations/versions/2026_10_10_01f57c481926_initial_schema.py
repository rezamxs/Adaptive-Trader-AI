"""initial_schema

Revision ID: 01f57c481926
Revises: 
Create Date: 2026-10-10 17:34:37.255878

"""
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
    """Keep downgrade non-destructive for databases that predate this migration.

    The upgrade uses create_all against existing databases. We cannot safely infer
    which tables were created by this revision, so dropping metadata here could erase
    user data.
    """
    pass
