from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph


def _driver_uri(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg://"):
        return "postgresql://" + database_url.removeprefix("postgresql+psycopg://")
    if database_url.startswith("postgresql://"):
        return database_url
    raise ValueError("CHECKPOINT_DATABASE_URL must use PostgreSQL with the psycopg driver.")


@contextmanager
def open_checkpointer(database_url: str) -> Iterator[object]:
    """Open a dedicated checkpointer connection; setup is an explicit deployment operation."""
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    from langgraph.checkpoint.postgres import PostgresSaver

    manager = PostgresSaver.from_conn_string(_driver_uri(database_url))
    checkpointer = manager.__enter__()
    try:
        yield checkpointer
    finally:
        manager.__exit__(None, None, None)


class ThreadMetadata(TypedDict):
    created_at: str
    status: str


class ThreadRegistry:
    """Use a separate checkpoint namespace to durably register threads before their first turn."""

    namespace = "caseflow_threads"

    def __init__(self, checkpointer: object):
        builder = StateGraph(ThreadMetadata)
        builder.add_node("register", lambda state: state)
        builder.add_edge(START, "register")
        builder.add_edge("register", END)
        self.graph = builder.compile(checkpointer=checkpointer)

    @staticmethod
    def _config(thread_id: str) -> dict[str, object]:
        return {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ThreadRegistry.namespace}
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
        snapshot = self.graph.get_state(self._config(thread_id))
        if not snapshot.values:
            return None
        return {
            "thread_id": thread_id,
            "created_at": str(snapshot.values["created_at"]),
            "status": str(snapshot.values.get("status", "created")),
        }
