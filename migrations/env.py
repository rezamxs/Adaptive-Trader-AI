import os
import sys
from logging.config import fileConfig

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

from backend.app.db.base import Base
import backend.app.models  # load all models

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def get_url():
    """Alembic destination safety: migrations run ONLY against an explicitly
    pinned ALEMBIC_DATABASE_URL. A silent fallback to the application
    settings URL is exactly how a stray `alembic upgrade head` in the wrong
    working directory could reach the production database (H4-incident
    class). Fail closed with a clear operator error instead; the settings
    URL is deliberately NOT echoed here so credentials never reach logs —
    the operator copies the intended value into the env var themselves."""
    url = os.environ.get("ALEMBIC_DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "ALEMBIC_DATABASE_URL is not set — refusing to migrate. Alembic "
            "never guesses its destination: export "
            "ALEMBIC_DATABASE_URL='<full database URL>' to pin the target "
            "database explicitly before running any migration. The "
            "application settings URL is intentionally not printed here to "
            "keep credentials out of logs; copy the intended value from your "
            "own environment only if you really mean to migrate that "
            "database."
        )
    return url

def run_migrations_offline() -> None:
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Buglist item 45: autogenerate previously compared nothing but
            # table/column existence — type changes, server-default changes
            # and new unique constraints were silently NOT generated, and
            # SQLite ALTERs failed without batch mode. Wire the standard
            # comparison trio + render_as_batch for the SQLite rebuild path.
            compare_type=True,
            compare_server_default=True,
            compare_user_options=True,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
