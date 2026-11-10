from collections.abc import Iterable
from typing import Any

from sqlalchemy import Engine, func, inspect, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from caseflow.data.synthetic import (
    ATTEMPTS_PER_REGION_PER_DAY,
    DATASET_AS_OF,
    DATASET_START,
    DOCUMENTS,
    REGIONS,
    dataset_metadata,
    generate_attempts,
    generate_changes,
    generate_incidents,
)
from caseflow.models import Base, ChangeEvent, DatasetMetadata, Document, Incident, PaymentAttempt

BATCH_SIZE = 1000
CASEFLOW_TABLES = {
    "payment_attempts",
    "change_events",
    "incidents",
    "documents",
    "dataset_metadata",
}


def _insert_idempotently(engine: Engine, model: type, rows: Iterable[dict[str, Any]]) -> int:
    inserted_or_ignored = 0
    statement = pg_insert(model.__table__).on_conflict_do_nothing()
    batch: list[dict[str, Any]] = []
    with engine.begin() as connection:
        for row in rows:
            batch.append(row)
            if len(batch) == BATCH_SIZE:
                connection.execute(statement, batch)
                inserted_or_ignored += len(batch)
                batch.clear()
        if batch:
            connection.execute(statement, batch)
            inserted_or_ignored += len(batch)
    return inserted_or_ignored


def _assert_known_caseflow_schema(engine: Engine) -> None:
    existing = set(inspect(engine).get_table_names(schema="public"))
    unknown = existing - CASEFLOW_TABLES
    if unknown:
        names = ", ".join(sorted(unknown)[:10])
        raise ValueError("Refusing to seed a database with unrelated public tables: " + names)
    if existing and "dataset_metadata" not in existing:
        raise ValueError(
            "CaseFlow-named tables exist without its dataset marker; "
            "inspect the database before seeding."
        )
    if "dataset_metadata" in existing:
        with engine.connect() as connection:
            dataset_id = connection.scalar(
                select(DatasetMetadata.value).where(DatasetMetadata.key == "dataset_id")
            )
        expected = dataset_metadata()["dataset_id"]
        if dataset_id != expected:
            raise ValueError(
                "The existing dataset marker does not match this generator; refusing to seed."
            )


def seed_database(seed_engine: Engine) -> dict[str, int | str]:
    """Create missing schema objects and insert stable IDs; never deletes existing rows."""
    database_name = str(seed_engine.url.database or "")
    if database_name != "caseflow" and not database_name.endswith("_test"):
        raise ValueError(
            "Refusing to seed an unrecognized database name; use caseflow or "
            "an isolated *_test database."
        )
    _assert_known_caseflow_schema(seed_engine)
    Base.metadata.create_all(seed_engine)
    _insert_idempotently(seed_engine, PaymentAttempt, generate_attempts())
    _insert_idempotently(seed_engine, ChangeEvent, generate_changes())
    _insert_idempotently(seed_engine, Incident, generate_incidents())

    document_rows = [
        {
            **item,
            "updated_at": DATASET_AS_OF,
        }
        for item in DOCUMENTS
    ]
    _insert_idempotently(seed_engine, Document, document_rows)
    metadata_rows = [{"key": key, "value": value} for key, value in dataset_metadata().items()]
    _insert_idempotently(seed_engine, DatasetMetadata, metadata_rows)

    with seed_engine.begin() as connection:
        connection.exec_driver_sql(
            "GRANT SELECT ON TABLE payment_attempts, change_events, incidents, documents, "
            "dataset_metadata TO caseflow_app"
        )
        counts = {
            "payment_attempts": connection.scalar(select(func.count()).select_from(PaymentAttempt)),
            "change_events": connection.scalar(select(func.count()).select_from(ChangeEvent)),
            "incidents": connection.scalar(select(func.count()).select_from(Incident)),
            "documents": connection.scalar(select(func.count()).select_from(Document)),
            "dataset_metadata": connection.scalar(
                select(func.count()).select_from(DatasetMetadata)
            ),
        }
    return {
        **{name: int(value or 0) for name, value in counts.items()},
        "expected_payment_attempts": (
            (DATASET_AS_OF.date() - DATASET_START.date()).days
            * len(REGIONS)
            * ATTEMPTS_PER_REGION_PER_DAY
        ),
    }
