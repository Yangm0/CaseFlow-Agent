from __future__ import annotations

import json
from typing import Any, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from sqlalchemy import text
from sqlalchemy.engine import Engine

from caseflow.db import make_engine
from caseflow.schemas import TicketDraft


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    draft: TicketDraft


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["approve", "reject"]


def _payload(draft: TicketDraft) -> dict[str, Any]:
    return draft.model_dump(mode="json")


def make_action_engine(database_url: str) -> Engine:
    """Create an engine for the function-only action_executor database role."""
    if database_url.startswith("postgresql://"):
        database_url = "postgresql+psycopg://" + database_url.removeprefix("postgresql://")
    return make_engine(database_url)


class ActionStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def create_pending(self, thread_id: str, draft: TicketDraft) -> dict[str, Any]:
        payload = _payload(draft)
        approval_id = str(uuid4())
        with self.engine.begin() as connection:
            result = connection.execute(
                text(
                    "SELECT caseflow_actions.create_pending_approval("
                    "CAST(:approval_id AS uuid), :thread_id, CAST(:payload AS jsonb))"
                ),
                {
                    "approval_id": approval_id,
                    "thread_id": thread_id,
                    "payload": json.dumps(payload, ensure_ascii=False),
                },
            ).scalar_one()
        return dict(result)

    def revise_pending(
        self,
        approval_id: str,
        expected_payload_hash: str,
        draft: TicketDraft,
    ) -> dict[str, Any]:
        """Atomically supersede an edited pending draft and bind a new interrupt payload."""
        with self.engine.begin() as connection:
            result = connection.execute(
                text(
                    "SELECT caseflow_actions.revise_pending_approval("
                    "CAST(:approval_id AS uuid), :expected_payload_hash, "
                    "CAST(:new_approval_id AS uuid), CAST(:payload AS jsonb))"
                ),
                {
                    "approval_id": approval_id,
                    "expected_payload_hash": expected_payload_hash,
                    "new_approval_id": str(uuid4()),
                    "payload": json.dumps(_payload(draft), ensure_ascii=False),
                },
            ).scalar_one()
        return dict(result)

    def get(self, approval_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            result = connection.execute(
                text("SELECT caseflow_actions.get_approval(CAST(:approval_id AS uuid))"),
                {"approval_id": approval_id},
            ).scalar_one()
        return dict(result) if result else None

    def list_for_thread(self, thread_id: str) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            result = connection.execute(
                text("SELECT caseflow_actions.list_thread_approvals(:thread_id)"),
                {"thread_id": thread_id},
            ).scalar_one()
        return [dict(item) for item in (result or [])]

    def approve_and_execute(self, approval_id: str, payload_hash: str) -> dict[str, Any]:
        with self.engine.begin() as connection:
            result = connection.execute(
                text(
                    "SELECT caseflow_actions.approve_and_create_ticket("
                    "CAST(:approval_id AS uuid), :payload_hash)"
                ),
                {"approval_id": approval_id, "payload_hash": payload_hash},
            ).scalar_one()
        return dict(result)

    def reject(self, approval_id: str, payload_hash: str) -> dict[str, Any]:
        with self.engine.begin() as connection:
            result = connection.execute(
                text(
                    "SELECT caseflow_actions.reject_approval("
                    "CAST(:approval_id AS uuid), :payload_hash)"
                ),
                {"approval_id": approval_id, "payload_hash": payload_hash},
            ).scalar_one()
        return dict(result)


class ApprovalState(TypedDict, total=False):
    approval_id: str
    thread_id: str
    payload_hash: str
    draft: dict[str, Any]
    decision: dict[str, Any]
    result: dict[str, Any]


class ApprovalWorkflow:
    def __init__(self, checkpointer: Any, store: ActionStore):
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import interrupt

        self.store = store

        def request_decision(state: ApprovalState) -> dict[str, Any]:
            decision = interrupt(
                {
                    "approval_id": state["approval_id"],
                    "thread_id": state["thread_id"],
                    "draft": state["draft"],
                    "payload_hash": state["payload_hash"],
                }
            )
            return {"decision": decision}

        def execute_decision(state: ApprovalState) -> dict[str, Any]:
            if state.get("decision", {}).get("decision") == "approve":
                result = self.store.approve_and_execute(
                    state["approval_id"], state["payload_hash"]
                )
            else:
                result = self.store.reject(state["approval_id"], state["payload_hash"])
            return {"result": result}

        builder = StateGraph(ApprovalState)
        builder.add_node("await_human", request_decision)
        builder.add_node("execute_approved_action", execute_decision)
        builder.add_edge(START, "await_human")
        builder.add_edge("await_human", "execute_approved_action")
        builder.add_edge("execute_approved_action", END)
        self.graph = builder.compile(checkpointer=checkpointer)

    @staticmethod
    def config(approval_id: str) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": f"caseflow-approval:{approval_id}",
            }
        }

    def request(self, approval: dict[str, Any]) -> dict[str, Any]:
        result = self.graph.invoke(
            {
                "approval_id": str(approval["approval_id"]),
                "thread_id": str(approval["thread_id"]),
                "draft": approval["action_payload"],
                "payload_hash": str(approval["payload_hash"]),
            },
            config=self.config(str(approval["approval_id"])),
        )
        return dict(result)

    def ensure_pending_interrupt(self, approval: dict[str, Any]) -> dict[str, Any]:
        """Recreate an interrupted checkpoint if DB creation committed before checkpointing."""
        if approval.get("status") != "pending":
            return approval
        snapshot = self.graph.get_state(self.config(str(approval["approval_id"])))
        if snapshot and snapshot.next:
            return dict(snapshot.values)
        return self.request(approval)

    def resume(self, approval_id: str, decision: ApprovalDecision) -> dict[str, Any]:
        from langgraph.types import Command

        result = self.graph.invoke(
            Command(resume={"decision": decision.decision}),
            config=self.config(approval_id),
        )
        return dict(result)
