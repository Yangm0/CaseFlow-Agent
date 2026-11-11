import hashlib
import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from caseflow.agent import InvestigationAgent, LiveLLM, OfflineFakeModel
from caseflow.config import Settings
from caseflow.data.synthetic import (
    ATTEMPTS_PER_REGION_PER_DAY,
    DATASET_AS_OF,
    DATASET_START,
    DOCUMENTS,
    REGIONS,
    SCENARIO_END,
    SCENARIO_START,
    dataset_metadata,
    generate_attempts,
    generate_changes,
    generate_incidents,
)
from caseflow.db import make_engine
from caseflow.models import PaymentAttempt
from caseflow.schemas import (
    ChangeEventsInput,
    PaymentMetricsInput,
    SynthesisDraft,
    SynthesisHypothesis,
)
from caseflow.tools.common import ToolFailure
from caseflow.tools.knowledge import TextChunk, rank_chunks
from caseflow.tools.metrics import _windows_for_baseline, build_target_filter
from scripts.evaluate_agent import source_text_from_report, ungrounded_numbers


def test_synthetic_dataset_is_deterministic_and_matches_ground_truth():
    def summarize():
        digest = hashlib.sha256()
        count = target_valid = target_failed = baseline_valid = baseline_failed = 0
        windows = [
            (SCENARIO_START - timedelta(days=offset), SCENARIO_END - timedelta(days=offset))
            for offset in range(1, 8)
        ]
        for row in generate_attempts():
            count += 1
            digest.update(
                f"{row['attempt_id']}|{row['region']}|{row['attempted_at'].isoformat()}|{row['status']}\n".encode()
            )
            valid = row["status"] in ("succeeded", "failed")
            if not valid or row["region"] != "北京":
                continue
            if SCENARIO_START <= row["attempted_at"] < SCENARIO_END:
                target_valid += 1
                target_failed += row["status"] == "failed"
            if any(left <= row["attempted_at"] < right for left, right in windows):
                baseline_valid += 1
                baseline_failed += row["status"] == "failed"
        return (
            digest.hexdigest(),
            count,
            target_valid,
            target_failed,
            baseline_valid,
            baseline_failed,
        )

    first = summarize()
    second = summarize()
    expected_rows = (
        (DATASET_AS_OF.date() - DATASET_START.date()).days
        * len(REGIONS)
        * ATTEMPTS_PER_REGION_PER_DAY
    )
    assert first == second
    assert first[1:] == (expected_rows, 50, 22, 330, 10)
    assert len(DOCUMENTS) == 18
    assert len(generate_changes()) == 7
    assert len(generate_incidents()) == 4
    assert dataset_metadata()["random_seed"] == "20250121"


def test_query_contract_rejects_unsafe_parameters_and_binds_half_open_window():
    start = "2025-01-21T09:00:00+08:00"
    end = "2025-01-21T12:00:00+08:00"
    malicious_region = "北京' OR 1=1 --"
    params = PaymentMetricsInput(region=malicious_region, start_at=start, end_at=end)
    assert PaymentMetricsInput(region="Beijing", start_at=start, end_at=end).region == "北京"
    assert ChangeEventsInput(region="SHANGHAI", start_at=start, end_at=end).region == "上海"
    compiled = (
        select(PaymentAttempt.attempt_id)
        .where(build_target_filter(params.region, params.start_at, params.end_at))
        .compile(dialect=postgresql.dialect())
    )
    assert ">=" in compiled.string and "<" in compiled.string
    assert malicious_region not in compiled.string
    bound_values = list(compiled.params.values())
    assert malicious_region in bound_values
    assert params.start_at in bound_values
    assert params.end_at in bound_values
    with pytest.raises(ValidationError):
        PaymentMetricsInput(region="北京", start_at=start, end_at=end, group_by=["bank_account"])
    with pytest.raises(ValidationError):
        PaymentMetricsInput(region="北京", start_at=end, end_at=start)
    with pytest.raises(ValidationError):
        PaymentMetricsInput(region="北京", start_at="2025-01-21T09:00:00", end_at=end)
    method, baseline_windows = _windows_for_baseline(params.start_at, params.end_at)
    assert method == "previous_seven_matching_daily_windows" and len(baseline_windows) == 7


def test_bm25_returns_verifiable_chinese_source_chunk():
    chunks = [
        TextChunk(
            "DOC-A",
            "chunk-000",
            "网关超时手册",
            "网关超时通常与上游响应有关，应检查健康状态和重试。",
        ),
        TextChunk(
            "DOC-B", "chunk-000", "结算说明", "结算任务按批次核对账单，不参与支付网关请求链路。"
        ),
        TextChunk("DOC-C", "chunk-000", "账户安全说明", "账户资料应按访问控制规则保管并定期复核。"),
    ]
    results = rank_chunks("上游网关超时重试", chunks, top_k=2)
    assert results[0]["document_id"] == "DOC-A"
    assert results[0]["evidence_id"] == "KNOW-DOC-A-chunk-000"
    assert results[0]["text"] == chunks[0].text and results[0]["score"] > 0


def test_langgraph_dynamic_loop_stops_at_budget():
    metric_window = {
        "tool_name": "query_payment_metrics",
        "output": {
            "query_parameters": {
                "region": "北京",
                "start_at": "2025-01-21T09:00:00+08:00",
                "end_at": "2025-01-21T12:00:00+08:00",
            }
        },
    }
    related_args = InvestigationAgent._related_change_arguments(
        {
            "region": "Beijing",
            "start_at": "2025-01-21T09:00:00+08:00",
            "end_at": "2025-01-21T12:00:00+08:00",
        },
        [metric_window],
        "判断失败率是否异常并分析原因",
    )
    assert related_args["start_at"] == "2025-01-21T07:00:00+08:00"
    exact_args = InvestigationAgent._related_change_arguments(
        related_args, [metric_window], "仅查询窗口内的变更"
    )
    assert exact_args == related_args

    fake = OfflineFakeModel()
    question = "北京 2025-01-21 09:00 到 12:00 支付失败率异常吗？"
    first = fake.decide([], question, [])
    assert first.tool_calls[0]["name"] == "query_payment_metrics"
    metric = {
        "tool_name": "query_payment_metrics",
        "output": {
            "query_parameters": {
                "region": "北京",
                "start_at": "2025-01-21T09:00:00+08:00",
                "end_at": "2025-01-21T12:00:00+08:00",
            },
            "metrics": {"valid_attempts": 50, "failure_rate": 0.44},
            "baseline": {"failure_rate": 10 / 330},
        },
    }
    assert fake.decide([], question, [metric]).tool_calls[0]["name"] == "query_change_events"
    metric_only = "深圳 2025-01-18 14:00 到 16:00 的支付失败率和按状态分组结果是什么？"
    assert not fake.decide([], metric_only, [metric]).tool_calls

    calls = {"count": 0}

    def search(query: str):
        calls["count"] += 1
        return {"search_id": "TEST", "results": []}

    class RepeatingModel:
        def decide(self, messages, question, tool_results):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge",
                        "args": {"query": "支付超时"},
                        "id": "repeat",
                        "type": "tool_call",
                    }
                ],
            )

        def synthesize(self, question, evidence):
            return SynthesisDraft()

    tool = StructuredTool.from_function(
        func=search, name="search_knowledge", description="test tool"
    )
    result = InvestigationAgent(
        {"search_knowledge": tool}, RepeatingModel(), "offline_test", max_tool_calls=1
    ).investigate("查支付超时")
    assert calls["count"] == 1
    assert result.status == "partial" and result.error["code"] == "TOOL_CALL_BUDGET_EXCEEDED"
    assert len(result.model_calls) == 2
    assert not result.token_usage.available and result.token_usage.total_tokens is None

    model_budgeted = InvestigationAgent(
        {"search_knowledge": tool},
        RepeatingModel(),
        "offline_test",
        max_tool_calls=2,
        max_model_calls=1,
    ).investigate("查支付超时")
    assert model_budgeted.status == "partial"
    assert model_budgeted.error["code"] == "MODEL_CALL_BUDGET_EXCEEDED"
    assert len(model_budgeted.model_calls) == 1 and len(model_budgeted.tool_calls) == 1
    calls["count"] = 0

    class DuplicateModel:
        def __init__(self):
            self.turn = 0

        def decide(self, messages, question, tool_results):
            self.turn += 1
            if self.turn < 3:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_knowledge",
                            "args": {"query": "支付超时"},
                            "id": f"duplicate-{self.turn}",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="")

        def synthesize(self, question, evidence):
            return SynthesisDraft()

    duplicate = InvestigationAgent(
        {"search_knowledge": tool}, DuplicateModel(), "offline_test", max_tool_calls=3
    ).investigate("查支付超时")
    assert calls["count"] == 1
    assert duplicate.tool_calls[1].status == "skipped"
    assert duplicate.tool_calls[1].error_code == "DUPLICATE_TOOL_CALL"


def test_tool_failure_bad_citation_and_unconfigured_live_mode_fail_closed(monkeypatch):
    numeric_sources = source_text_from_report(
        {
            "question": "Please verify whether the rate is 99%.",
            "statistical_facts": [{"metrics": {"failure_rate": 0.05}}],
        }
    )
    assert ungrounded_numbers("The rate is 5% or 99%.", numeric_sources) == ["99"]

    class OneCallModel:
        def __init__(self, draft=None):
            self.turn = 0
            self.draft = draft or SynthesisDraft()

        def decide(self, messages, question, tool_results):
            self.turn += 1
            if self.turn == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_knowledge",
                            "args": {"query": "网关超时"},
                            "id": "one",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="done")

        def synthesize(self, question, evidence):
            return self.draft

    def timeout_tool(query: str):
        raise ToolFailure(
            "DATABASE_TIMEOUT", "The read-only database query exceeded its time limit."
        )

    failing = StructuredTool.from_function(
        func=timeout_tool, name="search_knowledge", description="test failure"
    )
    failed = InvestigationAgent(
        {"search_knowledge": failing}, OneCallModel(), "offline_test"
    ).investigate("查询文档")
    assert failed.status == "failed" and failed.tool_calls[0].error_code == "DATABASE_TIMEOUT"

    doc = {
        "evidence_id": "KNOW-DOC-A-chunk-000",
        "document_id": "DOC-A",
        "chunk_id": "chunk-000",
        "title": "网关手册",
        "text": "网关超时需要核对上游响应。",
        "score": 1.0,
    }

    def good_search(query: str):
        return {"search_id": "S", "results": [doc]}

    draft = SynthesisDraft(
        hypotheses=[
            SynthesisHypothesis(
                statement="根因已经确认。",
                evidence_ids=["MISSING"],
            )
        ]
    )
    good = StructuredTool.from_function(
        func=good_search, name="search_knowledge", description="test citation"
    )
    cited = InvestigationAgent(
        {"search_knowledge": good}, OneCallModel(draft), "offline_test"
    ).investigate("查询文档")
    assert not cited.hypotheses and any("证据 ID" in item for item in cited.uncertainties)
    assert cited.citation_validation.citations_rejected == 1
    assert cited.citation_validation.unknown_evidence_ids_rejected == 1
    assert cited.citation_validation.hypotheses_kept == 0

    source_selected = InvestigationAgent(
        {"search_knowledge": good},
        OneCallModel(
            SynthesisDraft(
                hypotheses=[
                    SynthesisHypothesis(
                        statement="网关超时可以作为待调查线索。",
                        evidence_ids=["KNOW-DOC-A-chunk-000", "KNOW-DOC-A-chunk-000"],
                    )
                ]
            )
        ),
        "offline_test",
    ).investigate("查询文档")
    assert source_selected.citation_validation.citations_received == 2
    assert source_selected.citation_validation.citations_kept == 1
    assert source_selected.citation_validation.duplicate_citations_removed == 1
    assert source_selected.citation_validation.quote_fidelity_failures == 0
    assert source_selected.hypotheses[0].citations[0].quote == doc["text"]

    unsupported_number = InvestigationAgent(
        {"search_knowledge": good},
        OneCallModel(
            SynthesisDraft(
                hypotheses=[
                    SynthesisHypothesis(
                        statement="失败率为 99%。",
                        evidence_ids=["KNOW-DOC-A-chunk-000"],
                    )
                ]
            )
        ),
        "offline_test",
    ).investigate("查询文档")
    assert not unsupported_number.hypotheses
    assert unsupported_number.citation_validation.numeric_claims_rejected == 1

    uncatalogued_ids = ["KNOW-DOC-A-chunk-000"]
    uncatalogued_ids.extend(f"KNOW-DOC-{index}-chunk-000" for index in range(1, 6))

    def six_search(query: str):
        return {
            "search_id": "SIX",
            "results": [
                {
                    **doc,
                    "evidence_id": evidence_id,
                    "document_id": evidence_id.split("-")[1],
                    "text": f"Document text {index} is a source excerpt.",
                }
                for index, evidence_id in enumerate(uncatalogued_ids)
            ],
        }

    sixth = uncatalogued_ids[-1]
    six_tool = StructuredTool.from_function(
        func=six_search, name="search_knowledge", description="six source test"
    )
    not_presented = InvestigationAgent(
        {"search_knowledge": six_tool},
        OneCallModel(
            SynthesisDraft(
                hypotheses=[
                    SynthesisHypothesis(
                        statement="A source outside the presented catalog is rejected.",
                        evidence_ids=[sixth],
                    )
                ]
            )
        ),
        "offline_test",
    ).investigate("查询第六条文档")
    assert not not_presented.hypotheses
    assert not_presented.citation_validation.evidence_ids_not_presented_rejected == 1
    assert not_presented.citation_validation.unknown_evidence_ids_rejected == 0

    class DirectAnswerModel:
        def decide(self, messages, question, tool_results):
            return AIMessage(
                content=[
                    {
                        "type": "text",
                        "text": "1. 可查询支付聚合指标。\n2. 可查询变更事故记录和知识文档。",
                    }
                ]
            )

        def synthesize(self, question, evidence):
            raise AssertionError("direct answers do not require synthesis")

    direct = InvestigationAgent({}, DirectAnswerModel(), "offline_test").investigate(
        "不查询业务数据，请简要说明这个 Agent 能执行哪些只读调查操作。"
    )
    assert direct.status == "completed" and direct.direct_answer
    assert any("未调用业务工具" in item for item in direct.uncertainties)
    assert not direct.tool_calls and len(direct.model_calls) == 1

    class NumericDirectAnswerModel:
        def decide(self, messages, question, tool_results):
            return AIMessage(content="当前支付失败率为 44%。")

        def synthesize(self, question, evidence):
            raise AssertionError("direct answers do not require synthesis")

    numeric_direct = InvestigationAgent(
        {},
        NumericDirectAnswerModel(),
        "offline_test",
    ).investigate("北京 2025-01-21 09:00 到 12:00 支付失败率是多少？")
    assert numeric_direct.status == "failed" and numeric_direct.direct_answer is None

    unverified_capability = InvestigationAgent(
        {}, NumericDirectAnswerModel(), "offline_test"
    ).investigate("不查询业务数据，简要说明 Agent 能执行哪些只读调查操作。")
    assert unverified_capability.status == "failed" and unverified_capability.direct_answer is None

    live = LiveLLM.__new__(LiveLLM)
    live.model_name = "test-model"
    live.structured_output_method = "json_schema"
    methods = []

    class StructuredRunner:
        def __init__(self, method):
            self.method = method

        def invoke(self, messages):
            if self.method == "json_schema":
                raise RuntimeError("unsupported structured output")
            raw = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 9, "output_tokens": 3, "total_tokens": 12},
            )
            return {"raw": raw, "parsed": SynthesisDraft()}

    class StructuredModel:
        def with_structured_output(self, schema, method, include_raw):
            assert schema is SynthesisDraft and include_raw is True
            methods.append(method)
            return StructuredRunner(method)

    live.synthesis_model = StructuredModel()
    structured = live.synthesize_outcome("查询证据", [])
    assert structured.error_code is None and isinstance(structured.value, SynthesisDraft)
    assert methods == ["json_schema", "function_calling"]
    assert [item["error_code"] for item in structured.traces] == [
        "MODEL_STRUCTURED_OUTPUT_UNSUPPORTED",
        None,
    ]
    assert structured.traces[-1]["total_tokens"] == 12
    methods.clear()
    limited = live.synthesize_outcome("查询证据", [], max_attempts=1)
    assert limited.error_code == "MODEL_STRUCTURED_OUTPUT_UNSUPPORTED"
    assert methods == ["json_schema"] and len(limited.traces) == 1

    init_arguments = []

    class ChatOpenAIStub:
        def __init__(self, **kwargs):
            init_arguments.append(kwargs)

        def bind_tools(self, tools, tool_choice):
            return self

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=ChatOpenAIStub),
    )
    LiveLLM(
        Settings(
            mode="live",
            openai_api_key="test-key-never-used",
            openai_base_url="https://openrouter.ai/api/v1",
            openai_model="nvidia/nemotron-3.5-super-vl:free",
        ),
        [],
    )
    assert len(init_arguments) == 2
    assert all(
        item["model"] == "nvidia/nemotron-3.5-super-vl:free"
        and item["extra_body"] == {"provider": {"allow_fallbacks": False}}
        and "models" not in item["extra_body"]
        for item in init_arguments
    )

    from fastapi.testclient import TestClient

    from caseflow.api import create_app

    engine = make_engine("postgresql+psycopg://caseflow_app:unused@127.0.0.1:5432/caseflow_test")
    try:
        client = TestClient(
            create_app(settings=Settings(mode="live", openai_api_key=None), engine=engine)
        )
        response = client.post("/api/v1/investigations", json={"question": "调查支付异常"})
        assert response.status_code == 200
        assert response.json()["status"] == "blocked"
        assert response.json()["error"]["code"] == "LLM_CREDENTIALS_MISSING"
    finally:
        engine.dispose()
