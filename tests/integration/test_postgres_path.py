from datetime import datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError

from caseflow.agent import OfflineFakeModel, build_agent
from caseflow.api import create_app
from caseflow.config import Settings
from caseflow.data.seed import seed_database
from caseflow.models import Document, PaymentAttempt
from caseflow.schemas import ChangeEventsInput, KnowledgeSearchInput, PaymentMetricsInput
from caseflow.tools.changes import ChangeEventsService
from caseflow.tools.knowledge import KnowledgeService
from caseflow.tools.metrics import MetricsService
from caseflow.tools.registry import build_tool_registry

pytestmark = pytest.mark.integration


def test_postgres_seed_metrics_time_boundary_and_read_only_role(postgres_engines):
    app_engine, seed_engine = postgres_engines
    first = seed_database(seed_engine)
    second = seed_database(seed_engine)
    assert first["payment_attempts"] == second["payment_attempts"] >= 49_600
    assert first["documents"] == second["documents"] == 18
    with app_engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(Document)) == 18

    zone = ZoneInfo("Asia/Shanghai")
    start = datetime(2025, 1, 5, 0, 0, tzinfo=zone)
    end = datetime(2025, 1, 5, 1, 0, tzinfo=zone)
    region = f"边界-{uuid4().hex[:8]}"
    rows = [
        {"attempt_id": f"T-{uuid4().hex}", "region": region, "attempted_at": datetime(2025, 1, 4, 23, 59, 59, tzinfo=zone), "service_name": "payment-router", "status": "failed"},
        {"attempt_id": f"T-{uuid4().hex}", "region": region, "attempted_at": start, "service_name": "payment-router", "status": "failed"},
        {"attempt_id": f"T-{uuid4().hex}", "region": region, "attempted_at": end, "service_name": "payment-router", "status": "succeeded"},
    ]
    probe_ids = [row["attempt_id"] for row in rows]
    with seed_engine.begin() as connection:
        connection.execute(pg_insert(PaymentAttempt.__table__).on_conflict_do_nothing(), rows)
    try:
        result = MetricsService(app_engine).query(PaymentMetricsInput(region=region, start_at=start, end_at=end, group_by=["service_name"]))
        assert result["metrics"] == {"valid_attempts": 1, "failed_attempts": 1, "failure_rate": 1.0}
        assert result["groups"][0]["service_name"] == "payment-router"

        with app_engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.execute(text("INSERT INTO payment_attempts (attempt_id, region, attempted_at, service_name, status) VALUES ('read-only-probe', '北京', now(), 'payment-router', 'failed')"))
            except DBAPIError:
                transaction.rollback()
            else:
                transaction.rollback()
                pytest.fail("Application role unexpectedly has database write permission.")
    finally:
        with seed_engine.begin() as connection:
            connection.execute(delete(PaymentAttempt).where(PaymentAttempt.attempt_id.in_(probe_ids)))


def test_postgres_bm25_and_change_queries_return_real_source_rows(postgres_engines):
    app_engine, _ = postgres_engines
    search = KnowledgeService(app_engine).search(KnowledgeSearchInput(query="payment-router 路由优先级 上游超时", top_k=5))
    assert search["index_size_chunks"] >= 18 and search["results"]
    assert search["results"][0]["evidence_id"].startswith("KNOW-")
    assert search["results"][0]["text"]
    changes = ChangeEventsService(app_engine).query(
        ChangeEventsInput(region="北京", start_at="2025-01-21T08:00:00+08:00", end_at="2025-01-21T10:00:00+08:00")
    )
    assert {item["event_id"] for item in changes["events"]} >= {"CHG-20250121-BJ-ROUTE-01", "INC-20250121-BJ-01"}


@pytest.mark.e2e
def test_offline_fastapi_investigation_uses_postgres_and_bm25(postgres_engines):
    app_engine, _ = postgres_engines
    settings = Settings(mode="offline_test")
    tools = build_tool_registry(app_engine, statement_timeout_ms=3000)
    agent = build_agent(settings, tools, model_override=OfflineFakeModel())
    client = TestClient(create_app(settings=settings, engine=app_engine, agent=agent))
    response = client.post(
        "/api/v1/investigations",
        json={"question": "北京 2025-01-21 09:00 到 12:00 支付失败率是否异常？请查询相关变更和知识文档。"},
    )
    assert response.status_code == 200
    report = response.json()
    assert report["mode"] == "offline_test" and report["status"] == "completed"
    assert report["statistical_facts"][0]["metrics"] == {"valid_attempts": 50, "failed_attempts": 22, "failure_rate": 0.44}
    assert report["document_evidence"] and report["system_events"]
    assert [item["tool_name"] for item in report["tool_calls"]] == ["query_payment_metrics", "query_change_events", "search_knowledge"]
    assert all(item["evidence_ids"] for item in report["tool_calls"])
    assert report["hypotheses"]
    hypothesis = report["hypotheses"][0]
    assert "可能" in hypothesis["statement"] and "不足以确认因果关系" in hypothesis["statement"]
    source_text = {
        item["evidence_id"]: item.get("text", item.get("summary", ""))
        for item in [*report["document_evidence"], *report["system_events"]]
    }
    assert all(
        citation["evidence_id"] in source_text
        and citation["quote"] in source_text[citation["evidence_id"]]
        for citation in hypothesis["citations"]
    )
    assert report["trace_id"] and "因果" in report["summary"]
