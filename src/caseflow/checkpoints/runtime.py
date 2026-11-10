from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from psycopg import OperationalError
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


def _driver_uri(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg://"):
        return "postgresql://" + database_url.removeprefix("postgresql+psycopg://")
    if database_url.startswith("postgresql://"):
        return database_url
    raise ValueError("CHECKPOINT_DATABASE_URL must use PostgreSQL with the psycopg driver.")


@contextmanager
def open_checkpointer(database_url: str) -> Iterator[object]:
    """Open a reconnecting pool for checkpoints; setup remains an explicit operation."""
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    from langgraph.checkpoint.postgres import PostgresSaver

    pool = ConnectionPool(
        conninfo=_driver_uri(database_url),
        min_size=1,
        max_size=4,
        timeout=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    try:
        pool.open(wait=True)
        yield PostgresSaver(pool)
    finally:
        pool.close()


class ThreadMetadata(TypedDict):
    created_at: str
    status: str


class ThreadRegistry:
    """Use a prefixed checkpoint thread ID to register user threads durably."""

    namespace = "caseflow-registry"

    def __init__(self, checkpointer: object):
        builder = StateGraph(ThreadMetadata)
        builder.add_node("register", lambda state: state)
        builder.add_edge(START, "register")
        builder.add_edge("register", END)
        self.graph = builder.compile(checkpointer=checkpointer)

    @staticmethod
    def _config(thread_id: str) -> dict[str, object]:
        return {
            "configurable": {"thread_id": f"{ThreadRegistry.namespace}:{thread_id}"}
        }

    def create(self) -> dict[str, str]:
        thread_id = str(uuid4())
        metadata = {
            "created_at": datetime.now(UTC).isoformat(),
            "status": "created",
        }
        self.graph.invoke(metadata, config=self._config(thread_id))
        return {"thread_id": thread_id, **metadata}

    def get(self, thread_id: str) -> dict[str, str] | None:
        config = self._config(thread_id)
        try:
            snapshot = self.graph.get_state(config)
        except OperationalError:
            # A database restart can invalidate an in-use pooled connection. The
            # pool discards it; retry once after it has replenished the connection.
            snapshot = self.graph.get_state(config)
        if not snapshot.values:
            return None
        return {
            "thread_id": thread_id,
            "created_at": str(snapshot.values["created_at"]),
            "status": str(snapshot.values.get("status", "created")),
        }
