from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url

from caseflow.api import create_app
from caseflow.approvals.service import ActionStore, ApprovalWorkflow, make_action_engine
from caseflow.config import get_settings
from caseflow.db import make_engine

ROOT = Path(__file__).resolve().parents[1]


def _docker_executable() -> str:
    from shutil import which

    return which("docker.exe") or str(
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs"
        / "DockerDesktop"
        / "resources"
        / "bin"
        / "docker.exe"
    )


def _docker_psql(sql: str) -> str:
    encoded = base64.b64encode(sql.encode("utf-8")).decode("ascii")
    container_command = (
        f'printf %s {encoded} | base64 -d | '
        'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d caseflow_test -At'
    )
    result = subprocess.run(
        [_docker_executable(), "compose", "exec", "-T", "postgres", "sh", "-lc", container_command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        raise RuntimeError("The isolated test database admin query failed.")
    return result.stdout.strip()


def _cleanup(thread_ids: list[str]) -> None:
    if not thread_ids:
        return
    values = ", ".join(f"('{thread_id}')" for thread_id in thread_ids)
    sql = f"""
BEGIN;
CREATE TEMP TABLE v1d_smoke_threads(thread_id text PRIMARY KEY) ON COMMIT DROP;
INSERT INTO v1d_smoke_threads VALUES {values};
CREATE TEMP TABLE v1d_smoke_approvals(approval_id uuid PRIMARY KEY) ON COMMIT DROP;
INSERT INTO v1d_smoke_approvals
SELECT approval_id FROM caseflow_actions.action_approvals
WHERE thread_id IN (SELECT thread_id FROM v1d_smoke_threads);
INSERT INTO v1d_smoke_threads
SELECT 'caseflow-approval:' || approval_id::text FROM v1d_smoke_approvals
ON CONFLICT DO NOTHING;
DELETE FROM caseflow_checkpoint.checkpoint_blobs
WHERE thread_id IN (SELECT thread_id FROM v1d_smoke_threads);
DELETE FROM caseflow_checkpoint.checkpoint_writes
WHERE thread_id IN (SELECT thread_id FROM v1d_smoke_threads);
DELETE FROM caseflow_checkpoint.checkpoints
WHERE thread_id IN (SELECT thread_id FROM v1d_smoke_threads);
DELETE FROM caseflow_actions.action_audit_log
WHERE approval_id IN (SELECT approval_id FROM v1d_smoke_approvals);
DELETE FROM caseflow_actions.support_tickets
WHERE approval_id IN (SELECT approval_id FROM v1d_smoke_approvals);
DELETE FROM caseflow_actions.action_approvals
WHERE approval_id IN (SELECT approval_id FROM v1d_smoke_approvals);
COMMIT;
SELECT count(*) FROM caseflow_actions.action_approvals
WHERE thread_id IN (SELECT thread_id FROM (VALUES {values}) AS smoke_threads(thread_id));
"""
    output = _docker_psql(sql)
    if output.splitlines()[-1:] != ["0"]:
        raise RuntimeError("V1-D smoke cleanup did not remove only its tagged artifacts.")


def _wait_for_mcp(process: subprocess.Popen, url: str) -> None:
    deadline = time.monotonic() + 20
    with httpx.Client(timeout=1) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("The isolated MCP process exited before becoming healthy.")
            try:
                response = client.get(url)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.25)
    raise RuntimeError("The isolated MCP process did not become healthy in time.")


def _assert_report(report: dict[str, object]) -> list[str]:
    metrics = report.get("statistical_facts") or []
    target = next(
        (
            item
            for item in metrics
            if item.get("metrics", {}).get("valid_attempts") == 50
            and item.get("metrics", {}).get("failed_attempts") == 22
        ),
        None,
    )
    if not target:
        raise AssertionError("The Agent report did not include the PostgreSQL 22/50 metric.")
    if target.get("baseline", {}).get("valid_attempts") != 330:
        raise AssertionError("The Agent report did not include the 10/330 baseline.")
    calls = report.get("tool_calls") or []
    names = [str(item.get("tool_name")) for item in calls]
    if names != ["query_payment_metrics", "query_change_events", "search_knowledge"]:
        raise AssertionError(f"Unexpected real MCP call order: {names}")
    for item in calls:
        if item.get("status") != "succeeded":
            raise AssertionError("An MCP call did not complete successfully.")
    proposal = report.get("action_proposal") or {}
    if proposal.get("priority") != "high" or proposal.get("region") != "北京":
        raise AssertionError("The Agent did not propose a high priority Beijing ticket.")
    if proposal.get("source_investigation_id") != report.get("investigation_id"):
        raise AssertionError("The server did not bind the proposal to the investigation.")
    metric_ids = {item.get("evidence_id") for item in metrics}
    cited_ids = set(proposal.get("supporting_evidence_ids") or [])
    if not cited_ids or not cited_ids.intersection(metric_ids):
        raise AssertionError("The proposal evidence does not include the database metric.")
    return names


def main() -> int:
    settings = get_settings()
    if settings.mode != "offline_test":
        raise SystemExit("V1-D deterministic smoke requires CASEFLOW_MODE=offline_test.")
    if not settings.test_database_url or not settings.test_checkpoint_database_url:
        raise SystemExit("V1-D smoke requires isolated test and checkpoint database URLs.")
    test_url = make_url(settings.test_database_url)
    checkpoint_url = make_url(settings.test_checkpoint_database_url)
    if test_url.database != "caseflow_test" or checkpoint_url.database != "caseflow_test":
        raise SystemExit("V1-D smoke refuses to run outside caseflow_test.")
    if not settings.action_executor_database_url:
        raise SystemExit("ACTION_EXECUTOR_DATABASE_URL is required for V1-D smoke.")
    action_url = make_url(settings.action_executor_database_url).set(database="caseflow_test")
    action_url_string = action_url.render_as_string(hide_password=False)
    port = 8762
    endpoint = f"http://127.0.0.1:{port}/mcp"
    environment = os.environ.copy()
    environment["DATABASE_URL"] = settings.test_database_url
    environment["CASEFLOW_MCP_HOST"] = "127.0.0.1"
    environment["CASEFLOW_MCP_PORT"] = str(port)
    server = subprocess.Popen(
        [sys.executable, "-m", "caseflow_mcp.server"],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    thread_ids: list[str] = []
    action_engine = make_action_engine(action_url_string)
    store = ActionStore(action_engine)
    report_data: dict[str, object] = {"database": "caseflow_test", "mcp_endpoint": endpoint}
    query = (
        "请调查北京地区 2025-01-21 09:00 到 12:00 支付失败率是否异常，"
        "查询相关变更和知识文档。如果需要处理，请帮我创建一个高优先级工单。"
    )
    smoke_settings = settings.model_copy(
        update={
            "database_url": settings.test_database_url,
            "checkpoint_database_url": settings.test_checkpoint_database_url,
            "action_executor_database_url": action_url_string,
            "mcp_server_url": endpoint,
            "max_model_calls": 8,
        }
    )
    read_engine = make_engine(settings.test_database_url)
    try:
        _wait_for_mcp(server, f"http://127.0.0.1:{port}/health")
        approval_id = ""
        approval_hash = ""
        thread_id = ""
        rejected_thread_id = ""
        with TestClient(create_app(smoke_settings, engine=read_engine)) as client:
            created = client.post("/api/v1/threads")
            created.raise_for_status()
            thread_id = created.json()["thread_id"]
            thread_ids.append(thread_id)
            response = client.post(
                f"/api/v1/threads/{thread_id}/messages", json={"question": query}
            )
            response.raise_for_status()
            report = response.json()
            call_order = _assert_report(report)
            restored_state = client.get(f"/api/v1/threads/{thread_id}")
            restored_state.raise_for_status()
            if not restored_state.json().get("event_cursor"):
                raise AssertionError("The thread did not expose its latest SSE replay cursor.")
            if report.get("action_status") != "awaiting_approval":
                raise AssertionError("The Agent proposal did not create a pending approval.")
            pending = client.get(f"/api/v1/threads/{thread_id}/approvals")
            pending.raise_for_status()
            approval = pending.json()[0]
            approval_id = approval["approval_id"]
            approval_hash = approval["payload_hash"]
            if approval.get("status") != "pending":
                raise AssertionError("The generated ticket was not pending approval.")
            if client.post(
                f"/api/v1/threads/{thread_id}/messages", json={"question": "再查一次"}
            ).status_code != 409:
                raise AssertionError("A pending action did not block a new thread message.")

            edited_draft = dict(approval["action_payload"])
            edited_draft["title"] += "（值班复核）"
            revision = client.post(
                f"/api/v1/threads/{thread_id}/approvals",
                json={"draft": edited_draft},
            )
            revision.raise_for_status()
            revised = revision.json()
            if revised["approval_id"] == approval_id or revised["payload_hash"] == approval_hash:
                raise AssertionError("Editing the draft did not bind a new approval hash.")
            if client.get(
                f"/api/v1/threads/{thread_id}/approvals/{approval_id}"
            ).json().get("status") != "superseded":
                raise AssertionError("The previous approval was not superseded.")
            try:
                store.approve_and_execute(approval_id, approval_hash)
            except Exception:
                report_data["superseded_approval_rejected"] = True
            else:
                raise AssertionError("A stale payload hash could execute after draft revision.")

            current_approval_id = revised["approval_id"]
            workflow = ApprovalWorkflow(client.app.state.checkpointer, store)
            if not workflow.graph.get_state(
                workflow.config(current_approval_id)
            ).next:
                raise AssertionError("LangGraph did not interrupt on the revised approval.")
            report_data.update(
                {
                    "thread_id": thread_id,
                    "investigation_id": report["investigation_id"],
                    "tool_order": call_order,
                    "metrics": report["statistical_facts"],
                    "evidence_ids": [
                        item.get("evidence_id")
                        for item in [
                            *report.get("statistical_facts", []),
                            *report.get("document_evidence", []),
                            *report.get("system_events", []),
                        ]
                    ],
                    "proposal": revised["action_payload"],
                    "approval_id": current_approval_id,
                    "edited_payload_hash_changed": True,
                    "action_state": "awaiting_approval",
                }
            )

            rejected_created = client.post("/api/v1/threads")
            rejected_created.raise_for_status()
            rejected_thread_id = rejected_created.json()["thread_id"]
            thread_ids.append(rejected_thread_id)
            reject_response = client.post(
                f"/api/v1/threads/{rejected_thread_id}/messages", json={"question": query}
            )
            reject_response.raise_for_status()
            reject_report = reject_response.json()
            _assert_report(reject_report)
            reject_pending = client.get(
                f"/api/v1/threads/{rejected_thread_id}/approvals"
            )
            reject_pending.raise_for_status()
            rejected_approval = reject_pending.json()[0]
            rejected_result = client.post(
                f"/api/v1/threads/{rejected_thread_id}/approvals/"
                f"{rejected_approval['approval_id']}/decision",
                json={"decision": "reject"},
            )
            rejected_result.raise_for_status()
            repeated_reject = client.post(
                f"/api/v1/threads/{rejected_thread_id}/approvals/"
                f"{rejected_approval['approval_id']}/decision",
                json={"decision": "reject"},
            )
            repeated_reject.raise_for_status()
            report_data["rejection_idempotent"] = True

        # A new application instance proves both the action interrupt and report survived restart.
        with TestClient(create_app(smoke_settings, engine=read_engine)) as client:
            restored = client.get(f"/api/v1/threads/{thread_id}")
            restored.raise_for_status()
            restored_data = restored.json()
            if restored_data["status"] != "awaiting_approval":
                raise AssertionError("Pending approval did not recover after API restart.")
            if not restored_data.get("messages") or not restored_data.get("report"):
                raise AssertionError("Thread messages and report were not restored.")
            approval_id = report_data["approval_id"]
            workflow = ApprovalWorkflow(client.app.state.checkpointer, store)
            if not workflow.graph.get_state(workflow.config(approval_id)).next:
                raise AssertionError("Approval interrupt did not survive API restart.")
            result = client.post(
                f"/api/v1/threads/{thread_id}/approvals/{approval_id}/decision",
                json={"decision": "approve"},
            )
            result.raise_for_status()
            execution = result.json()
            if execution.get("status") != "executed" or not execution.get("ticket_id"):
                raise AssertionError("Approved action did not create a support ticket.")
            duplicate = client.post(
                f"/api/v1/threads/{thread_id}/approvals/{approval_id}/decision",
                json={"decision": "approve"},
            )
            duplicate.raise_for_status()
            duplicate_data = duplicate.json()
            duplicate_ticket = duplicate_data.get("ticket_id") or (
                duplicate_data.get("execution_result") or {}
            ).get("ticket_id")
            if duplicate_ticket != execution["ticket_id"]:
                raise AssertionError("Repeated approval did not return the same ticket.")
            report_data["execution"] = execution
            report_data["repeated_approval_same_ticket"] = True

        ticket_count = _docker_psql(
            "SELECT count(*) FROM caseflow_actions.support_tickets t "
            "JOIN caseflow_actions.action_approvals a USING (approval_id) "
            f"WHERE a.thread_id IN ('{thread_id}', '{rejected_thread_id}') "
            "AND a.status = 'executed';"
        )
        if ticket_count != "1":
            raise AssertionError(
                f"Expected exactly one executed test ticket, found {ticket_count}."
            )
        report_data["test_ticket_count"] = int(ticket_count)
        destination = ROOT / ".tmp" / "v1d_smoke_last.json"
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report_data, ensure_ascii=False, indent=2))
        print("Artifacts will be removed from caseflow_test after the smoke.")
        return 0
    finally:
        read_engine.dispose()
        action_engine.dispose()
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        _cleanup(thread_ids)


if __name__ == "__main__":
    raise SystemExit(main())
