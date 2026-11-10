from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from fastapi.testclient import TestClient

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

    live_settings = settings.model_copy(
        update={
            "mode": "live",
            "database_url": settings.test_database_url,
            "seed_database_url": settings.test_seed_database_url,
            "max_tool_calls": 3,
            "max_model_calls": 4,
            "max_agent_seconds": 60,
        }
    )
    engine = make_engine(live_settings.database_url)
    if engine.url.database != "caseflow_test":
        engine.dispose()
        raise SystemExit("Refusing smoke test outside the isolated caseflow_test database.")
    try:
        with TestClient(create_app(settings=live_settings, engine=engine)) as client:
            response = client.post("/api/v1/investigations", json={"question": args.question})
        if response.status_code != 200:
            raise SystemExit(f"Live API returned HTTP {response.status_code}; no trace was saved.")
        report = response.json()
        serialized = json.dumps(report, ensure_ascii=False)
        secret = live_settings.openai_api_key.get_secret_value()
        if secret in serialized or live_settings.openai_base_url in serialized:
            raise SystemExit(
                "Sensitive model configuration appeared in the response; trace not saved."
            )
        destination = ROOT / ".tmp" / "v1b_live_smoke_trace.json"
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(serialized + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "mode": report["mode"],
                    "model": live_settings.openai_model,
                    "status": report["status"],
                    "trace_id": report["trace_id"],
                    "elapsed_ms": report["elapsed_ms"],
                    "model_calls": report["model_calls"],
                    "tool_calls": report["tool_calls"],
                    "statistical_facts": report["statistical_facts"],
                    "citation_validation": report["citation_validation"],
                    "token_usage": report["token_usage"],
                    "error": report.get("error"),
                    "saved_trace": str(destination),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
