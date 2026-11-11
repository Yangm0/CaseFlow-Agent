from __future__ import annotations

import asyncio
import os
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.engine import Engine

from caseflow.agent import InvestigationAgent, build_agent
from caseflow.approvals.service import (
    ActionStore,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalWorkflow,
    make_action_engine,
)
from caseflow.checkpoints.runtime import ThreadRegistry, open_checkpointer
from caseflow.config import Settings, get_settings
from caseflow.db import check_database, make_engine
from caseflow.events import ThreadEventHub, format_sse
from caseflow.mcp_client import build_mcp_tool_registry
from caseflow.schemas import (
    InvestigationReport,
    InvestigationRequest,
    TicketDraft,
    canonical_region,
)
from caseflow.tools.registry import build_tool_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = Path(os.environ.get("CASEFLOW_WEB_ROOT", PROJECT_ROOT / "web"))


class ThreadLocks:
    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}

    def for_thread(self, thread_id: str) -> asyncio.Lock:
        return self._locks.setdefault(thread_id, asyncio.Lock())


def _message_to_dict(message: Any) -> dict[str, str]:
    role = "assistant" if getattr(message, "type", "") in ("ai", "assistant") else "user"
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        content = str(content)
    return {"role": role, "content": content}


def _report_evidence_ids(report: dict[str, Any]) -> tuple[set[str], set[str]]:
    metric_ids = {
        str(item.get("evidence_id"))
        for item in report.get("statistical_facts", [])
        if item.get("evidence_id")
    }
    other_ids = {
        str(item.get("evidence_id"))
        for item in [*report.get("document_evidence", []), *report.get("system_events", [])]
        if item.get("evidence_id")
    }
    return metric_ids, other_ids


def _validate_ticket_draft(draft: TicketDraft, report: dict[str, Any]) -> None:
    if not report.get("investigation_id") or report.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="A completed evidence-backed investigation is required before approval.",
        )
    if draft.source_investigation_id != report["investigation_id"]:
        raise HTTPException(status_code=422, detail="Draft must reference this investigation.")
    metric_ids, other_ids = _report_evidence_ids(report)
    supplied = set(draft.supporting_evidence_ids)
    if not supplied or not supplied.intersection(metric_ids) or not supplied.issubset(
        metric_ids | other_ids
    ):
        raise HTTPException(
            status_code=422,
            detail="Draft evidence IDs must include a metric and belong to this report.",
        )
    regions = {
        canonical_region(str(item.get("query_parameters", {}).get("region", "")))
        for item in report.get("statistical_facts", [])
    }
    if regions and canonical_region(draft.region) not in regions:
        raise HTTPException(
            status_code=422,
            detail="Draft region must match the verified metric evidence.",
        )


def _update_thread_report(
    agent: InvestigationAgent,
    thread_id: str,
    updates: dict[str, Any],
) -> dict[str, Any] | None:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = agent.graph.get_state(config)
    if not snapshot or not snapshot.values or not snapshot.values.get("report"):
        return None
    report = dict(snapshot.values["report"])
    report.update(updates)
    update: dict[str, Any] = {"report": report}
    history = list(snapshot.values.get("history", []))
    assistant_index = next(
        (
            index
            for index in range(len(history) - 1, -1, -1)
            if getattr(history[index], "type", "") in {"ai", "assistant"}
        ),
        None,
    )
    if assistant_index is not None:
        assistant_message = history[assistant_index]
        content = str(report.get("summary") or getattr(assistant_message, "content", ""))
        status = report.get("action_status")
        if status == "awaiting_approval":
            suffix = " 已生成一份待人工审批的工单草稿；尚未执行任何业务写入。"
        elif status == "executed":
            result = report.get("action_result") or {}
            ticket_id = result.get("ticket_id") if isinstance(result, dict) else None
            suffix = " 工单已批准并创建。"
            if ticket_id:
                suffix = f" 工单已批准并创建，工单 ID：{ticket_id}。"
        elif status == "rejected":
            suffix = " 工单草稿已拒绝，未创建工单。"
        elif status == "action_failed":
            suffix = " 工单执行失败；未确认业务写入。"
        else:
            suffix = ""
        if suffix:
            history[assistant_index] = assistant_message.model_copy(
                update={"content": content + suffix}
            )
            update["history"] = [history[assistant_index]]
    agent.graph.update_state(config, update)
    return report


def create_app(
    settings: Settings | None = None,
    engine: Engine | None = None,
    agent: InvestigationAgent | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()
    app_engine = engine or make_engine(app_settings.database_url)
    tools = build_tool_registry(
        app_engine,
        app_settings.sql_statement_timeout_ms,
        app_settings,
    )
    events = ThreadEventHub()
    app_agent = agent if agent is not None else build_agent(app_settings, tools, event_sink=events)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.checkpointer = None
        application.state.thread_registry = None
        application.state.action_store = None
        try:
            with ExitStack() as stack:
                if app_settings.checkpoint_database_url:
                    checkpointer = stack.enter_context(
                        open_checkpointer(app_settings.checkpoint_database_url)
                    )
                    application.state.checkpointer = checkpointer
                    application.state.thread_registry = ThreadRegistry(checkpointer)
                    if agent is None:
                        application.state.agent = build_agent(
                            app_settings,
                            tools,
                            checkpointer=checkpointer,
                            event_sink=events,
                        )
                if app_settings.action_executor_database_url:
                    action_engine = make_action_engine(app_settings.action_executor_database_url)
                    stack.callback(action_engine.dispose)
                    application.state.action_store = ActionStore(action_engine)
                yield
        finally:
            application.state.checkpointer = None
            application.state.thread_registry = None
            application.state.action_store = None
            if engine is None:
                app_engine.dispose()

    application = FastAPI(
        title="CaseFlow Payment Investigation Agent",
        version="0.3.0",
        description=(
            "PostgreSQL-backed investigation threads with MCP tools and human-approved actions."
        ),
        lifespan=lifespan,
    )
    application.state.settings = app_settings
    application.state.engine = app_engine
    application.state.agent = app_agent
    application.state.events = events
    application.state.thread_locks = ThreadLocks()

    def require_thread_services() -> tuple[Any, ThreadRegistry, InvestigationAgent]:
        checkpointer = application.state.checkpointer
        registry = application.state.thread_registry
        current_agent = application.state.agent
        if not checkpointer or not registry or not current_agent:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Persistent threads are not configured. "
                    "Set up CHECKPOINT_DATABASE_URL first."
                ),
            )
        return checkpointer, registry, current_agent

    def require_action_store(checkpointer: Any) -> ActionStore:
        action_store = application.state.action_store
        if not action_store:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Approved ticket actions are not configured. "
                    "Set ACTION_EXECUTOR_DATABASE_URL."
                ),
            )
        return action_store

    @application.get("/health")
    def health():
        database_ok = check_database(app_engine)
        live_blocked = app_settings.mode == "live" and application.state.agent is None
        response = {
            "status": "ok" if database_ok and not live_blocked else "degraded",
            "database": "available" if database_ok else "unavailable",
            "mode": app_settings.mode,
            "model": (
                "offline_test_stub"
                if app_settings.mode == "offline_test"
                else "configured"
                if application.state.agent is not None
                else "credentials_missing"
            ),
            "persistent_threads": bool(application.state.checkpointer),
            "approved_actions": bool(application.state.action_store),
        }
        return JSONResponse(
            status_code=200 if database_ok and not live_blocked else 503, content=response
        )

    @application.post("/api/v1/investigations", response_model=InvestigationReport)
    def investigate(request: InvestigationRequest):
        current_agent = application.state.agent
        if current_agent is None:
            return InvestigationReport(
                investigation_id="unconfigured",
                trace_id="unconfigured",
                mode=app_settings.mode,
                status="blocked",
                summary="live 模式尚未配置模型凭据，本次调查未启动。",
                statistical_facts=[],
                document_evidence=[],
                system_events=[],
                hypotheses=[],
                uncertainties=[
                    "请通过 OPENAI_API_KEY、OPENAI_MODEL 和可选 OPENAI_BASE_URL 配置 live 模式。"
                ],
                recommendations=[],
                tool_calls=[],
                operations_executed=["未执行数据库或业务操作。"],
                error={
                    "code": "LLM_CREDENTIALS_MISSING",
                    "message": "Live model credentials are not configured.",
                },
            )
        return current_agent.investigate(request.question)

    @application.post("/api/v1/threads", status_code=201)
    async def create_thread():
        _, registry, _ = require_thread_services()
        return await asyncio.to_thread(registry.create)

    @application.get("/api/v1/threads/{thread_id}")
    async def get_thread(thread_id: str):
        _, registry, current_agent = require_thread_services()
        try:
            UUID(thread_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail="Thread not found.") from error
        metadata = await asyncio.to_thread(registry.get, thread_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        try:
            snapshot = await asyncio.to_thread(
                current_agent.graph.get_state,
                {"configurable": {"thread_id": thread_id}},
            )
        except Exception:
            snapshot = None
        state = snapshot.values if snapshot and snapshot.values else {}
        history = [_message_to_dict(item) for item in state.get("history", [])]
        pending_approvals = []
        latest_approval = None
        if application.state.action_store:
            all_approvals = await asyncio.to_thread(
                application.state.action_store.list_for_thread, thread_id
            )
            pending_approvals = [item for item in all_approvals if item.get("status") == "pending"]
            workflow = ApprovalWorkflow(
                application.state.checkpointer, application.state.action_store
            )
            for approval in pending_approvals:
                await asyncio.to_thread(workflow.ensure_pending_interrupt, approval)
            latest = all_approvals[0] if all_approvals else None
            latest_approval = latest
            if pending_approvals and state.get("report"):
                latest_pending = pending_approvals[0]
                await asyncio.to_thread(
                    _update_thread_report,
                    current_agent,
                    thread_id,
                    {
                        "action_status": "awaiting_approval",
                        "approval_id": str(latest_pending["approval_id"]),
                        "action_proposal": latest_pending.get("action_payload"),
                        "action_result": None,
                    },
                )
                snapshot = await asyncio.to_thread(
                    current_agent.graph.get_state,
                    {"configurable": {"thread_id": thread_id}},
                )
                state = snapshot.values if snapshot and snapshot.values else {}
            if (
                not pending_approvals
                and latest
                and latest.get("status") in {"executed", "rejected"}
                and state.get("report")
            ):
                status = str(latest["status"])
                await asyncio.to_thread(
                    _update_thread_report,
                    current_agent,
                    thread_id,
                    {
                        "action_status": status,
                        "approval_id": str(latest["approval_id"]),
                        "action_result": latest.get("execution_result") or {"status": status},
                    },
                )
                snapshot = await asyncio.to_thread(
                    current_agent.graph.get_state,
                    {"configurable": {"thread_id": thread_id}},
                )
                state = snapshot.values if snapshot and snapshot.values else {}
        history = [_message_to_dict(item) for item in state.get("history", [])]
        thread_status = (
            "awaiting_approval"
            if pending_approvals
            else "running"
            if application.state.thread_locks.for_thread(thread_id).locked()
            else "interrupted"
            if snapshot and snapshot.next
            else state.get("status", metadata["status"])
        )
        return {
            **metadata,
            "messages": history,
            "report": state.get("report"),
            "status": thread_status,
            "pending_approvals": pending_approvals,
            "latest_approval": latest_approval,
            "event_cursor": events.latest_event_id(thread_id),
            "next": list(snapshot.next) if snapshot else [],
        }

    @application.post("/api/v1/threads/{thread_id}/messages", response_model=InvestigationReport)
    async def send_message(thread_id: str, request: InvestigationRequest):
        checkpointer, registry, current_agent = require_thread_services()
        if await asyncio.to_thread(registry.get, thread_id) is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        lock = application.state.thread_locks.for_thread(thread_id)
        async with lock:
            action_store = application.state.action_store
            if action_store:
                approvals = await asyncio.to_thread(action_store.list_for_thread, thread_id)
                if any(item.get("status") == "pending" for item in approvals):
                    raise HTTPException(
                        status_code=409,
                        detail="This thread has a pending action; decide or revise it first.",
                    )
            snapshot = await asyncio.to_thread(
                current_agent.graph.get_state,
                {"configurable": {"thread_id": thread_id}},
            )
            if snapshot and snapshot.next:
                raise HTTPException(
                    status_code=409,
                    detail="This thread has an interrupted investigation; resume it first.",
                )

            def run_investigation() -> InvestigationReport:
                try:
                    remote_tools = build_mcp_tool_registry(app_settings)
                    thread_agent = build_agent(
                        app_settings,
                        remote_tools,
                        checkpointer=checkpointer,
                        event_sink=events,
                    )
                    if thread_agent is None:
                        raise RuntimeError("Live model credentials are not configured.")
                    report = thread_agent.investigate(request.question, thread_id=thread_id)
                    if report.action_proposal:
                        if not application.state.action_store:
                            report = report.model_copy(
                                update={"action_status": "action_failed", "approval_id": None}
                            )
                        else:
                            draft = report.action_proposal
                            approval = action_store.create_pending(thread_id, draft)
                            if approval.get("status") == "pending":
                                workflow = ApprovalWorkflow(
                                    checkpointer, application.state.action_store
                                )
                                workflow.ensure_pending_interrupt(approval)
                                report = report.model_copy(
                                    update={
                                        "action_status": "awaiting_approval",
                                        "approval_id": str(approval["approval_id"]),
                                        "action_result": None,
                                    }
                                )
                                events.publish(
                                    thread_id,
                                    "approval_required",
                                    {
                                        "approval_id": approval["approval_id"],
                                        "status": "pending",
                                        "draft": approval["action_payload"],
                                        "evidence_ids": draft.supporting_evidence_ids,
                                    },
                                )
                            else:
                                report = report.model_copy(
                                    update={
                                        "action_status": str(approval.get("status")),
                                        "approval_id": str(approval["approval_id"]),
                                        "action_result": approval.get("execution_result"),
                                    }
                                )
                    if report.action_proposal:
                        thread_agent.graph.update_state(
                            {"configurable": {"thread_id": thread_id}},
                            {"report": report.model_dump(mode="json")},
                        )
                    return report
                except Exception as error:
                    events.publish(thread_id, "error", {"code": "THREAD_INVESTIGATION_FAILED"})
                    raise HTTPException(
                        status_code=503,
                        detail="The MCP investigation could not be completed.",
                    ) from error

            return await asyncio.to_thread(run_investigation)

    @application.post(
        "/api/v1/threads/{thread_id}/resume", response_model=InvestigationReport
    )
    async def resume_thread(thread_id: str):
        checkpointer, registry, _ = require_thread_services()
        if await asyncio.to_thread(registry.get, thread_id) is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        lock = application.state.thread_locks.for_thread(thread_id)
        async with lock:
            def resume_investigation() -> InvestigationReport:
                try:
                    remote_tools = build_mcp_tool_registry(app_settings)
                    thread_agent = build_agent(
                        app_settings,
                        remote_tools,
                        checkpointer=checkpointer,
                        event_sink=events,
                    )
                    if thread_agent is None:
                        raise RuntimeError("Live model credentials are not configured.")
                except HTTPException:
                    raise
                except Exception as error:
                    events.publish(thread_id, "error", {"code": "THREAD_RESUME_FAILED"})
                    raise HTTPException(
                        status_code=503,
                        detail="The saved MCP investigation could not be resumed.",
                    ) from error

                try:
                    return thread_agent.resume(thread_id)
                except ValueError as error:
                    raise HTTPException(
                        status_code=409,
                        detail="The thread has no resumable saved investigation.",
                    ) from error
                except HTTPException:
                    raise
                except Exception as error:
                    events.publish(thread_id, "error", {"code": "THREAD_RESUME_FAILED"})
                    raise HTTPException(
                        status_code=503,
                        detail="The saved MCP investigation could not be resumed.",
                    ) from error

            return await asyncio.to_thread(resume_investigation)

    @application.get("/api/v1/threads/{thread_id}/events")
    async def thread_events(
        thread_id: str,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        after: str | None = None,
    ):
        _, registry, _ = require_thread_services()
        if await asyncio.to_thread(registry.get, thread_id) is None:
            raise HTTPException(status_code=404, detail="Thread not found.")

        async def generate():
            async for item in events.stream(thread_id, last_event_id or after):
                yield format_sse(item)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.post("/api/v1/threads/{thread_id}/approvals", status_code=202)
    async def create_approval(thread_id: str, request: ApprovalRequest):
        checkpointer, registry, current_agent = require_thread_services()
        action_store = require_action_store(checkpointer)
        if await asyncio.to_thread(registry.get, thread_id) is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        lock = application.state.thread_locks.for_thread(thread_id)
        async with lock:
            snapshot = await asyncio.to_thread(
                current_agent.graph.get_state,
                {"configurable": {"thread_id": thread_id}},
            )
            state = snapshot.values if snapshot and snapshot.values else {}
            report = state.get("report") or {}
            _validate_ticket_draft(request.draft, report)

            def create_and_pause() -> dict[str, Any]:
                existing = [
                    item
                    for item in action_store.list_for_thread(thread_id)
                    if item.get("status") == "pending"
                ]
                if len(existing) > 1:
                    raise HTTPException(
                        status_code=409,
                        detail="Multiple pending actions exist; resolve the thread before editing.",
                    )
                if existing:
                    current = existing[0]
                    if current.get("action_payload") == request.draft.model_dump(mode="json"):
                        approval = current
                    else:
                        approval = action_store.revise_pending(
                            str(current["approval_id"]),
                            str(current["payload_hash"]),
                            request.draft,
                        )
                else:
                    approval = action_store.create_pending(thread_id, request.draft)
                if approval.get("status") == "pending":
                    workflow = ApprovalWorkflow(checkpointer, action_store)
                    workflow.ensure_pending_interrupt(approval)
                return approval

            approval = await asyncio.to_thread(create_and_pause)
            if approval.get("status") == "pending":
                await asyncio.to_thread(
                    _update_thread_report,
                    current_agent,
                    thread_id,
                    {
                        "action_proposal": request.draft.model_dump(mode="json"),
                        "action_status": "awaiting_approval",
                        "approval_id": str(approval["approval_id"]),
                        "action_result": None,
                    },
                )
            event_name = (
                "approval_required" if approval.get("status") == "pending" else "status"
            )
            events.publish(
                thread_id,
                event_name,
                {
                    "approval_id": approval["approval_id"],
                    "status": approval["status"],
                    "draft": approval["action_payload"],
                    "evidence_ids": request.draft.supporting_evidence_ids,
                },
            )
            return approval

    @application.get("/api/v1/threads/{thread_id}/approvals")
    async def list_approvals(thread_id: str):
        _, registry, _ = require_thread_services()
        action_store = require_action_store(application.state.checkpointer)
        if await asyncio.to_thread(registry.get, thread_id) is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        approvals = await asyncio.to_thread(action_store.list_for_thread, thread_id)
        workflow = ApprovalWorkflow(application.state.checkpointer, action_store)
        for approval in approvals:
            if approval.get("status") == "pending":
                await asyncio.to_thread(workflow.ensure_pending_interrupt, approval)
        return approvals

    @application.get("/api/v1/threads/{thread_id}/approvals/{approval_id}")
    async def get_approval(thread_id: str, approval_id: str):
        _, registry, _ = require_thread_services()
        action_store = require_action_store(application.state.checkpointer)
        if await asyncio.to_thread(registry.get, thread_id) is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        approval = await asyncio.to_thread(action_store.get, approval_id)
        if approval is None or approval.get("thread_id") != thread_id:
            raise HTTPException(status_code=404, detail="Approval not found.")
        if approval.get("status") == "pending":
            workflow = ApprovalWorkflow(application.state.checkpointer, action_store)
            await asyncio.to_thread(workflow.ensure_pending_interrupt, approval)
        return approval

    @application.post("/api/v1/threads/{thread_id}/approvals/{approval_id}/decision")
    async def decide_approval(thread_id: str, approval_id: str, decision: ApprovalDecision):
        checkpointer, registry, current_agent = require_thread_services()
        action_store = require_action_store(checkpointer)
        if await asyncio.to_thread(registry.get, thread_id) is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        lock = application.state.thread_locks.for_thread(thread_id)
        async with lock:
            approval = await asyncio.to_thread(action_store.get, approval_id)
            if approval is None or approval.get("thread_id") != thread_id:
                raise HTTPException(status_code=404, detail="Approval not found.")
            expected_status = "executed" if decision.decision == "approve" else "rejected"
            if approval.get("status") == expected_status:
                return approval
            if approval.get("status") != "pending":
                raise HTTPException(status_code=409, detail="Approval is no longer pending.")

            def resume_workflow() -> dict[str, Any]:
                workflow = ApprovalWorkflow(checkpointer, action_store)
                workflow.ensure_pending_interrupt(approval)
                return workflow.resume(approval_id, decision)

            try:
                result = await asyncio.to_thread(resume_workflow)
            except Exception as error:
                raise HTTPException(
                    status_code=409,
                    detail="Approval could not be resumed; its stored payload was not changed.",
                ) from error
            final = result.get("result", {})
            final_status = str(final.get("status", "action_failed"))
            if final_status not in {"executed", "rejected"}:
                final_status = "action_failed"
            await asyncio.to_thread(
                _update_thread_report,
                current_agent,
                thread_id,
                {
                    "action_status": final_status,
                    "approval_id": approval_id,
                    "action_result": final,
                },
            )
            events.publish(
                thread_id,
                "status",
                {"status": final_status, "approval_id": approval_id},
            )
            return final

    application.mount(
        "/web",
        StaticFiles(directory=str(WEB_ROOT), html=True),
        name="web",
    )

    @application.get("/")
    def web_ui():
        from fastapi.responses import FileResponse

        return FileResponse(WEB_ROOT / "index.html")

    return application


app = create_app()
