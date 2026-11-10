from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError


def make_engine(database_url: str) -> Engine:
    if not database_url.startswith("postgresql+psycopg://"):
        raise ValueError("CaseFlow requires a PostgreSQL URL using the psycopg driver.")
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=2,
        connect_args={"connect_timeout": 3},
    )


@contextmanager
def readonly_transaction(engine: Engine, statement_timeout_ms: int) -> Iterator:
    """Open a transaction that PostgreSQL itself marks read-only."""
    with engine.connect() as connection:
        with connection.begin():
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            connection.exec_driver_sql(
                f"SET LOCAL statement_timeout = '{int(statement_timeout_ms)}ms'"
            )
            yield connection


def check_database(engine: Engine) -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False
