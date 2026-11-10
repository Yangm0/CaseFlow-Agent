from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url

from caseflow.api import create_app
from caseflow.config import get_settings
from caseflow.db import make_engine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTION = (
    "北京 2025-01-21 09:00 到 12:00 支付失败率是否异常？请查询相关变更和知识文档，分析可能原因。"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one live API investigation against synthetic PostgreSQL test data."
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--confirm-zero-cost-free-endpoint",
        action="store_true",
        help="Required guard: confirm the selected OpenRouter model uses a verified :free route.",
    )
    parser.add_argument(
        "--mcp-url",
        default="http://127.0.0.1:8101/mcp",
        help="Loopback MCP endpoint; Live smoke thread requests must use the MCP client.",
    )
    parser.add_argument(
        "--max-model-calls",
        type=int,
        default=5,
        choices=range(1, 6),
        help="Hard per-investigation cap for this zero-cost account's available free budget.",
    )
    args = parser.parse_args()
    settings = get_settings()
    if not args.confirm_zero_cost_free_endpoint:
        raise SystemExit("Pass --confirm-zero-cost-free-endpoint after verifying provider pricing.")
    if not settings.openai_api_key or not settings.openai_api_key.get_secret_value().strip():
        raise SystemExit("OPENAI_API_KEY is missing; no model request was made.")
    if not settings.openai_model.endswith(":free"):
        raise SystemExit("Refusing to run: OPENAI_MODEL must use a :free model variant.")
    if (
        not settings.openai_base_url
        or urlparse(settings.openai_base_url).hostname != "openrouter.ai"
    ):
        raise SystemExit(
            "Refusing to run: this smoke test is limited to the verified OpenRouter route."
        )
    if not settings.test_database_url:
        raise SystemExit("TEST_DATABASE_URL is required; no API request was made.")
    if (
        not settings.test_checkpoint_database_url
        or make_url(settings.test_checkpoint_database_url).database != "caseflow_test"
    ):
        raise SystemExit("TEST_CHECKPOINT_DATABASE_URL must target caseflow_test.")
    mcp_url = urlparse(args.mcp_url)
    if mcp_url.hostname not in {"127.0.0.1", "localhost", "::1"} or mcp_url.path != "/mcp":
        raise SystemExit("Refusing MCP smoke: --mcp-url must be a loopback /mcp endpoint.")

    action_url = None
    if settings.action_executor_database_url:
        action_url = make_url(settings.action_executor_database_url)
        if action_url.database not in {"caseflow", "caseflow_test"}:
            raise SystemExit("Action executor URL must resolve to a CaseFlow database.")
        action_url = action_url.set(database="caseflow_test")

    live_settings = settings.model_copy(
        update={
            "mode": "live",
            "database_url": settings.test_database_url,
            "seed_database_url": settings.test_seed_database_url,
            "checkpoint_database_url": settings.test_checkpoint_database_url,
            "action_executor_database_url": (
                action_url.render_as_string(hide_password=False) if action_url else None
            ),
            "mcp_server_url": args.mcp_url,
            "max_tool_calls": 3,
            "max_model_calls": args.max_model_calls,
            "max_agent_seconds": 60,
        }
    )
    engine = make_engine(live_settings.database_url)
    if engine.url.database != "caseflow_test":
        engine.dispose()
        raise SystemExit("Refusing smoke test outside the isolated caseflow_test database.")
    try:
        with TestClient(create_app(settings=live_settings, engine=engine)) as client:
            thread_response = client.post("/api/v1/threads")
            if thread_response.status_code != 201:
                raise SystemExit("Live smoke could not create an isolated test thread.")
            thread_id = thread_response.json()["thread_id"]
            response = client.post(
                f"/api/v1/threads/{thread_id}/messages",
                json={"question": args.question},
            )
        if response.status_code != 200:
            raise SystemExit(f"Live API returned HTTP {response.status_code}; no trace was saved.")
        report = response.json()
        model_calls = report.get("model_calls", [])
        if len(model_calls) > args.max_model_calls:
            raise SystemExit("Live model-call trace exceeded the configured request budget.")
        expected_tools = {"query_payment_metrics", "query_change_events", "search_knowledge"}
        observed_tools = {call.get("tool_name") for call in report.get("tool_calls", [])}
        if not observed_tools.issubset(expected_tools):
            raise SystemExit("Live smoke attempted a tool outside the MCP read-only allowlist.")
        report_persisted = False
        with TestClient(create_app(settings=live_settings, engine=engine)) as client:
            saved_thread = client.get(f"/api/v1/threads/{thread_id}")
            if saved_thread.status_code == 200:
                saved_report = saved_thread.json().get("report") or {}
                report_persisted = saved_report.get("investigation_id") == report.get(
                    "investigation_id"
                )
        serialized = json.dumps(report, ensure_ascii=False)
        secret = live_settings.openai_api_key.get_secret_value()
        if secret in serialized or live_settings.openai_base_url in serialized:
            raise SystemExit(
                "Sensitive model configuration appeared in the response; trace not saved."
            )
        destination = ROOT / ".tmp" / "v1c_live_smoke_trace.json"
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(serialized + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "mode": report["mode"],
                    "model": live_settings.openai_model,
                    "thread_id": thread_id,
                    "status": report["status"],
                    "report_persisted": report_persisted,
                    "trace_id": report["trace_id"],
                    "elapsed_ms": report["elapsed_ms"],
                    "model_calls": model_calls,
                    "tool_calls": report["tool_calls"],
                    "statistical_facts": report["statistical_facts"],
                    "citation_validation": report["citation_validation"],
                    "token_usage": report["token_usage"],
                    "error": report.get("error"),
                    "saved_trace": str(destination),
                    "mcp_url_host": mcp_url.hostname,
                    "mcp_tools_observed": sorted(observed_tools),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        engine.dispose()
    return 0 if report["status"] == "completed" and report_persisted else 1


if __name__ == "__main__":
    raise SystemExit(main())
