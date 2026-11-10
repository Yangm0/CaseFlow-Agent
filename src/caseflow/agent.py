import json
import re
import time
from datetime import datetime, timedelta
from typing import Any, Literal, TypedDict
from uuid import uuid4
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from caseflow.config import Settings
from caseflow.schemas import (
    ChangeEvidence,
    DocumentEvidence,
    EvidenceQuote,
    Hypothesis,
    InvestigationReport,
    MetricEvidence,
    SynthesisDraft,
    ToolTrace,
)
from caseflow.tools.common import ToolFailure

LOCAL_TZ = ZoneInfo("Asia/Shanghai")
MAX_MODEL_CALLS = 10

AGENT_SYSTEM_PROMPT = """You are a payment-incident investigation agent. Use only the three provided read-only tools.
Never write SQL, request credentials, or claim that an operation was executed. Select tools based on the user's
question and the evidence already returned. Query metrics before making numerical claims. Use change records and
documents as evidence, but temporal overlap is not proof of causation. Stop when the evidence is sufficient or
the tool budget is exhausted. Do not invent numbers, document text, event IDs, or database facts. Synthetic
ground truth is not available to you. Final numeric facts are assembled from database tool output by the server.
"""

SYNTHESIS_SYSTEM_PROMPT = """Synthesize a concise Chinese investigation draft from the supplied evidence only.
Return hypotheses, uncertainties, and safe next-step recommendations. Hypotheses must be explicitly tentative,
must not assert causation as fact, and each must cite one or more exact quotations copied verbatim from the
provided evidence. Every citation quote must be a contiguous substring of the cited evidence content. Do not
repeat numeric measurements in prose; the server builds the statistical facts directly from query results.
Recommendations are proposals only; no action has been carried out. Do not infer a root cause from time overlap.
"""


class AgentState(TypedDict, total=False):
    question: str
    messages: list[BaseMessage]
    pending_tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    tool_traces: list[dict[str, Any]]
    tool_call_count: int
    model_call_count: int
    started_at: float
    status: Literal["completed", "partial", "failed", "blocked"]
    error: dict[str, str] | None
    report: dict[str, Any]


class OfflineFakeModel:
    """A deterministic test driver. It is not a natural-language model or a live capability claim."""

    regions = ("北京", "上海", "深圳", "杭州")

    def decide(
        self,
        messages: list[BaseMessage],
        question: str,
        tool_results: list[dict[str, Any]],
    ) -> AIMessage:
        used = [item["tool_name"] for item in tool_results]
        if any(item.get("error") for item in tool_results):
            return AIMessage(content="")

        metric_item = next(
            (item for item in reversed(tool_results) if item["tool_name"] == "query_payment_metrics"),
            None,
        )
        if metric_item is None and self._looks_like_metrics_question(question):
            region, start_at, end_at = self._parse_question(question)
            return self._call(
                "query_payment_metrics",
                {
                    "region": region,
                    "start_at": start_at.isoformat(),
                    "end_at": end_at.isoformat(),
                    "group_by": ["service_name", "status"],
                },
            )

        if metric_item is None and "search_knowledge" not in used:
            return self._call("search_knowledge", {"query": question, "top_k": 5})

        if metric_item is not None:
            metric_output = metric_item.get("output", {})
            current = metric_output.get("metrics", {}).get("failure_rate")
            baseline = metric_output.get("baseline", {}).get("failure_rate")
            valid = metric_output.get("metrics", {}).get("valid_attempts", 0)
            anomaly_signal = (
                isinstance(current, (int, float))
                and isinstance(baseline, (int, float))
                and valid >= 20
                and (current - baseline >= 0.05 or current >= max(0.05, baseline * 2.0))
            )
            if anomaly_signal and "query_change_events" not in used:
                params = metric_output.get("query_parameters", {})
                try:
                    change_start = datetime.fromisoformat(str(params.get("start_at"))) - timedelta(hours=2)
                except ValueError:
                    change_start = params.get("start_at")
                return self._call(
                    "query_change_events",
                    {
                        "region": params.get("region"),
                        "start_at": change_start.isoformat() if isinstance(change_start, datetime) else change_start,
                        "end_at": params.get("end_at"),
                    },
                )
            if "search_knowledge" not in used and (
                anomaly_signal or any(word in question for word in ("建议", "怎么", "原因", "排查", "处理"))
            ):
                context = " ".join(
                    str(event.get("summary", ""))
                    for event in self._events(tool_results)
                )
                return self._call(
                    "search_knowledge",
                    {"query": f"{question} {context}"[:500], "top_k": 5},
                )
        return AIMessage(content="")

    def synthesize(self, question: str, evidence: list[dict[str, Any]]) -> SynthesisDraft:
        changes = [
            item
            for source in evidence
            for item in source.get("events", [])
            if item.get("event_type") in ("change", "incident")
        ]
        documents = [item for source in evidence for item in source.get("documents", [])]
        hypotheses: list[Hypothesis] = []
        change = next((item for item in changes if item.get("event_type") == "change"), None)
        incident = next((item for item in changes if item.get("event_type") == "incident"), None)
        document = next(
            (
                item
                for item in documents
                if "路由" in str(item.get("text", ""))
                or "超时" in str(item.get("text", ""))
            ),
            None,
        )
        change_text = str(change.get("summary", "")) if change else ""
        document_text = str(document.get("text", "")) if document else ""
        if (
            change
            and document
            and ("路由" in change_text or "超时" in change_text)
            and ("路由" in document_text or "超时" in document_text)
        ):
            citations = [
                EvidenceQuote(evidence_id=change["evidence_id"], quote=change["summary"]),
                EvidenceQuote(evidence_id=document["evidence_id"], quote=document["text"][:160]),
            ]
            hypotheses.append(
                Hypothesis(
                    statement=(
                        "路由优先级或超时重试配置可能是本次波动的调查方向；目前只有时间线和操作手册线索，"
                        "不足以确认因果关系。"
                    ),
                    confidence="low",
                    citations=citations,
                )
            )
        uncertainties = [
            "当前合成数据未包含逐笔响应码、上游延迟分位数或完整发布前后日志。",
            "时间重叠与文档机制说明不能单独证明变更导致了指标变化。",
        ]
        if incident:
            uncertainties.append("事故记录仍处于调查状态，未给出已验证根因。")
        recommendations = [
            "由值班人员核对该窗口的上游超时、路由配置差异及受影响实例版本。",
            "对比相邻地区和变更前后同口径指标；满足既定回滚条件时由授权人员操作。",
        ]
        return SynthesisDraft(
            hypotheses=hypotheses,
            uncertainties=uncertainties,
            recommendations=recommendations,
        )

    @classmethod
    def _events(cls, tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for item in tool_results:
            if item["tool_name"] == "query_change_events":
                result.extend(item.get("output", {}).get("events", []))
        return result

    @staticmethod
    def _looks_like_metrics_question(question: str) -> bool:
        return any(word in question for word in ("支付", "失败率", "失败", "成功率", "异常", "指标"))

    @classmethod
    def _parse_question(cls, question: str) -> tuple[str, datetime, datetime]:
        region = next((item for item in cls.regions if item in question), "北京")
        date_match = re.search(
            r"(20\d{2})[-年](\d{1,2})[-月](\d{1,2})(?:日)?",
            question,
        )
        if date_match:
            year, month, day = (int(value) for value in date_match.groups())
        else:
            year, month, day = 2025, 1, 21
        tail = question[date_match.end() :] if date_match else question
        times = re.findall(r"(?<!\d)(\d{1,2})\s*[:：]\s*(\d{2})", tail)
        if len(times) >= 2:
            start_hour, start_minute = (int(value) for value in times[0])
            end_hour, end_minute = (int(value) for value in times[1])
        else:
            chinese_times = re.findall(r"(\d{1,2})\s*点(?:\s*(\d{1,2})\s*分)?", tail)
            if len(chinese_times) >= 2:
                start_hour = int(chinese_times[0][0])
                start_minute = int(chinese_times[0][1] or 0)
                end_hour = int(chinese_times[1][0])
                end_minute = int(chinese_times[1][1] or 0)
            else:
                start_hour, start_minute, end_hour, end_minute = 9, 0, 12, 0
        start_at = datetime(year, month, day, start_hour, start_minute, tzinfo=LOCAL_TZ)
        end_at = datetime(year, month, day, end_hour, end_minute, tzinfo=LOCAL_TZ)
        return region, start_at, end_at

    @staticmethod
    def _call(name: str, args: dict[str, Any]) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"offline-{uuid4().hex[:12]}", "type": "tool_call"}],
        )


class LiveLLM:
    def __init__(self, settings: Settings, tools: list[StructuredTool]):
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": settings.openai_model,
            "api_key": settings.openai_api_key.get_secret_value() if settings.openai_api_key else None,
            "temperature": 0,
            "timeout": settings.max_agent_seconds,
            "max_retries": 0,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        model = ChatOpenAI(**kwargs)
        self.decision_model = model.bind_tools(tools)
        self.synthesis_model = model.with_structured_output(SynthesisDraft)

    def decide(
        self,
        messages: list[BaseMessage],
        question: str,
        tool_results: list[dict[str, Any]],
    ) -> AIMessage:
        return self.decision_model.invoke(messages)

    def synthesize(self, question: str, evidence: list[dict[str, Any]]) -> SynthesisDraft:
        payload = json.dumps({"question": question, "evidence": evidence}, ensure_ascii=False, default=str)
        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=payload),
        ]
        result = self.synthesis_model.invoke(messages)
        if isinstance(result, SynthesisDraft):
            return result
        return SynthesisDraft.model_validate(result)


class InvestigationAgent:
    def __init__(
        self,
        tools: dict[str, StructuredTool],
        model: OfflineFakeModel | LiveLLM | Any,
        mode: Literal["offline_test", "live"],
        max_tool_calls: int = 8,
        max_agent_seconds: float = 25,
    ):
        self.tools = tools
        self.model = model
        self.mode = mode
        self.max_tool_calls = max_tool_calls
        self.max_agent_seconds = max_agent_seconds
        graph = StateGraph(AgentState)
        graph.add_node("llm_decision", self._llm_decision)
        graph.add_node("execute_tools", self._execute_tools)
        graph.add_node("finalize", self._finalize)
        graph.set_entry_point("llm_decision")
        graph.add_conditional_edges(
            "llm_decision",
            self._route_after_decision,
            {"execute_tools": "execute_tools", "finalize": "finalize"},
        )
        graph.add_edge("execute_tools", "llm_decision")
        graph.add_edge("finalize", END)
        self.graph = graph.compile()

    def investigate(self, question: str) -> InvestigationReport:
        investigation_id = str(uuid4())
        trace_id = str(uuid4())
        state: AgentState = {
            "question": question,
            "messages": [SystemMessage(content=AGENT_SYSTEM_PROMPT), HumanMessage(content=question)],
            "pending_tool_calls": [],
            "tool_results": [],
            "tool_traces": [],
            "tool_call_count": 0,
            "model_call_count": 0,
            "started_at": time.monotonic(),
            "status": "completed",
            "error": None,
        }
        final_state = self.graph.invoke(state)
        report_data = final_state.get("report")
        if not report_data:
            report_data = self._fallback_report(
                investigation_id,
                trace_id,
                question,
                final_state,
                {"code": "AGENT_NO_REPORT", "message": "The investigation ended without a report."},
            )
        report_data["investigation_id"] = investigation_id
        report_data["trace_id"] = trace_id
        report_data["mode"] = self.mode
        return InvestigationReport.model_validate(report_data)

    def _llm_decision(self, state: AgentState) -> dict[str, Any]:
        elapsed = time.monotonic() - state["started_at"]
        if elapsed >= self.max_agent_seconds:
            return self._budget_failure(state, "AGENT_TIME_BUDGET_EXCEEDED")
        if state.get("model_call_count", 0) >= MAX_MODEL_CALLS:
            return self._budget_failure(state, "MODEL_CALL_BUDGET_EXCEEDED")
        try:
            response = self.model.decide(
                state["messages"], state["question"], state.get("tool_results", [])
            )
        except Exception:
            return {
                "pending_tool_calls": [],
                "status": "partial" if state.get("tool_results") else "failed",
                "error": {"code": "MODEL_CALL_FAILED", "message": "The model decision call failed."},
                "model_call_count": state.get("model_call_count", 0) + 1,
            }
        tool_calls = list(getattr(response, "tool_calls", []) or [])
        if len(tool_calls) + state.get("tool_call_count", 0) > self.max_tool_calls:
            return {
                "messages": [*state["messages"], response],
                "pending_tool_calls": [],
                "status": "partial",
                "error": {"code": "TOOL_CALL_BUDGET_EXCEEDED", "message": "The investigation tool-call budget was reached."},
                "model_call_count": state.get("model_call_count", 0) + 1,
            }
        return {
            "messages": [*state["messages"], response],
            "pending_tool_calls": tool_calls,
            "model_call_count": state.get("model_call_count", 0) + 1,
        }

    def _execute_tools(self, state: AgentState) -> dict[str, Any]:
        tool_messages: list[ToolMessage] = []
        traces = list(state.get("tool_traces", []))
        results = list(state.get("tool_results", []))
        tool_call_count = state.get("tool_call_count", 0)
        status = state.get("status", "completed")
        error = state.get("error")
        for call in state.get("pending_tool_calls", []):
            if tool_call_count >= self.max_tool_calls:
                status = "partial"
                error = {"code": "TOOL_CALL_BUDGET_EXCEEDED", "message": "The investigation tool-call budget was reached."}
                break
            if time.monotonic() - state["started_at"] >= self.max_agent_seconds:
                status = "partial"
                error = {"code": "AGENT_TIME_BUDGET_EXCEEDED", "message": "The investigation time budget was reached."}
                break
            name = str(call.get("name", ""))
            call_id = str(call.get("id") or f"tool-{uuid4().hex[:12]}")
            raw_args = call.get("args", {})
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                if not isinstance(args, dict):
                    raise ValueError("tool arguments must be an object")
            except (TypeError, ValueError, json.JSONDecodeError):
                args = {}
                error_result = {"error": {"code": "INVALID_TOOL_ARGUMENTS", "message": "Tool arguments were not a valid object."}}
                traces.append(
                    ToolTrace(
                        tool_name=name or "unknown",
                        arguments={},
                        status="failed",
                        elapsed_ms=0,
                        error_code="INVALID_TOOL_ARGUMENTS",
                    ).model_dump()
                )
                results.append({"tool_name": name or "unknown", **error_result})
                tool_messages.append(ToolMessage(content=json.dumps(error_result), tool_call_id=call_id, name=name or "unknown"))
                tool_call_count += 1
                continue

            started = time.monotonic()
            try:
                if name not in self.tools:
                    raise ToolFailure("UNKNOWN_TOOL", "The requested tool is not available.")
                output = self.tools[name].invoke(args)
                tool_result = {"tool_name": name, "output": output}
                result_content = json.dumps(output, ensure_ascii=False, default=str)
                trace = ToolTrace(
                    tool_name=name,
                    arguments=args,
                    status="succeeded",
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    evidence_ids=self._output_evidence_ids(output),
                )
            except ToolFailure as failure:
                error_result = {"error": {"code": failure.code, "message": failure.public_message}}
                tool_result = {"tool_name": name, **error_result}
                result_content = json.dumps(error_result, ensure_ascii=False)
                trace = ToolTrace(
                    tool_name=name,
                    arguments=args,
                    status="failed",
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    error_code=failure.code,
                )
            except ValidationError:
                error_result = {"error": {"code": "INVALID_TOOL_ARGUMENTS", "message": "Tool arguments failed validation."}}
                tool_result = {"tool_name": name, **error_result}
                result_content = json.dumps(error_result, ensure_ascii=False)
                trace = ToolTrace(
                    tool_name=name,
                    arguments=args,
                    status="failed",
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    error_code="INVALID_TOOL_ARGUMENTS",
                )
            except Exception:
                error_result = {"error": {"code": "TOOL_EXECUTION_FAILED", "message": "The requested tool failed."}}
                tool_result = {"tool_name": name, **error_result}
                result_content = json.dumps(error_result, ensure_ascii=False)
                trace = ToolTrace(
                    tool_name=name,
                    arguments=args,
                    status="failed",
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    error_code="TOOL_EXECUTION_FAILED",
                )
            traces.append(trace.model_dump())
            results.append(tool_result)
            tool_messages.append(ToolMessage(content=result_content, tool_call_id=call_id, name=name or "unknown"))
            tool_call_count += 1

        if any(item.get("status") == "failed" for item in traces):
            status = "partial" if any(item.get("status") == "succeeded" for item in traces) else "failed"
            if error is None:
                last_error = next((item for item in reversed(traces) if item.get("status") == "failed"), {})
                error = {
                    "code": str(last_error.get("error_code") or "TOOL_EXECUTION_FAILED"),
                    "message": "One or more read-only tools failed.",
                }
        return {
            "messages": [*state["messages"], *tool_messages],
            "pending_tool_calls": [],
            "tool_results": results,
            "tool_traces": traces,
            "tool_call_count": tool_call_count,
            "status": status,
            "error": error,
        }

    def _finalize(self, state: AgentState) -> dict[str, Any]:
        evidence = self._synthesis_evidence(state.get("tool_results", []))
        try:
            draft = self.model.synthesize(state["question"], evidence)
            if not isinstance(draft, SynthesisDraft):
                draft = SynthesisDraft.model_validate(draft)
        except Exception:
            draft = SynthesisDraft(
                uncertainties=["结构化分析生成失败；保留工具返回的原始证据供复核。"],
                recommendations=["由值班人员根据引用的工具证据继续核对。"],
            )
            if state.get("status") == "completed":
                state_status: Literal["partial", "failed"] = "partial"
            else:
                state_status = state.get("status", "failed")
            state = {**state, "status": state_status}

        source_text = self._source_text_map(state.get("tool_results", []))
        valid_hypotheses: list[Hypothesis] = []
        invalid_hypothesis_count = 0
        for hypothesis in draft.hypotheses:
            if not hypothesis.citations:
                invalid_hypothesis_count += 1
                continue
            citation_ok = all(
                citation.evidence_id in source_text
                and citation.quote in source_text[citation.evidence_id]
                for citation in hypothesis.citations
            )
            claim_numbers = re.findall(r"(?<![\w])\d+(?:\.\d+)?%?", hypothesis.statement)
            cited_text = " ".join(citation.quote for citation in hypothesis.citations)
            citation_ok = citation_ok and all(number.rstrip("%") in cited_text for number in claim_numbers)
            if citation_ok:
                valid_hypotheses.append(hypothesis)
            else:
                invalid_hypothesis_count += 1
        uncertainties = list(draft.uncertainties)
        uncertainties = [item for item in uncertainties if not re.search(r"\d", item)]
        recommendations = [
            item for item in draft.recommendations if not re.search(r"\d", item)
        ]
        if invalid_hypothesis_count:
            uncertainties.append(
                "有假设因引用 ID 或逐字证据片段未通过校验而被移除。"
            )

        report = self._compile_report(state, valid_hypotheses, uncertainties, recommendations)
        return {"report": report}

    def _route_after_decision(self, state: AgentState) -> str:
        if state.get("pending_tool_calls") and state.get("status") not in ("failed", "blocked"):
            return "execute_tools"
        return "finalize"

    def _compile_report(
        self,
        state: AgentState,
        hypotheses: list[Hypothesis],
        uncertainties: list[str],
        recommendations: list[str],
    ) -> dict[str, Any]:
        metrics: list[MetricEvidence] = []
        documents: list[DocumentEvidence] = []
        events: list[ChangeEvidence] = []
        for item in state.get("tool_results", []):
            output = item.get("output", {})
            if item.get("tool_name") == "query_payment_metrics" and "evidence_id" in output:
                metrics.append(
                    MetricEvidence(
                        evidence_id=output["evidence_id"],
                        query_parameters=output["query_parameters"],
                        metrics=output["metrics"],
                        baseline=output["baseline"],
                        groups=output["groups"],
                        calculation=output["calculation"],
                    )
                )
            elif item.get("tool_name") == "search_knowledge":
                for result in output.get("results", []):
                    documents.append(DocumentEvidence.model_validate(result))
            elif item.get("tool_name") == "query_change_events":
                for result in output.get("events", []):
                    events.append(ChangeEvidence.model_validate(result))

        summary = self._summary(metrics, documents, events)
        error = state.get("error")
        status = state.get("status", "completed")
        successful_tools = any(
            item.get("status") == "succeeded" for item in state.get("tool_traces", [])
        )
        if not metrics and not documents and not events and error and not successful_tools:
            status = "failed"
        return {
            "investigation_id": "pending",
            "trace_id": "pending",
            "mode": self.mode,
            "status": status,
            "summary": summary,
            "statistical_facts": [item.model_dump() for item in metrics],
            "document_evidence": [item.model_dump() for item in documents],
            "system_events": [item.model_dump() for item in events],
            "hypotheses": [item.model_dump() for item in hypotheses],
            "uncertainties": uncertainties,
            "recommendations": recommendations,
            "tool_calls": [ToolTrace.model_validate(item).model_dump() for item in state.get("tool_traces", [])],
            "operations_executed": self._operations(state.get("tool_traces", [])),
            "error": error,
        }

    @staticmethod
    def _summary(
        metrics: list[MetricEvidence],
        documents: list[DocumentEvidence],
        events: list[ChangeEvidence],
    ) -> str:
        parts: list[str] = []
        for item in metrics:
            region = str(item.query_parameters.get("region", "未知地区"))
            start = str(item.query_parameters.get("start_at", ""))
            end = str(item.query_parameters.get("end_at", ""))
            valid = int(item.metrics.get("valid_attempts", 0))
            failed = int(item.metrics.get("failed_attempts", 0))
            rate = item.metrics.get("failure_rate")
            base_valid = int(item.baseline.get("valid_attempts", 0))
            base_rate = item.baseline.get("failure_rate")
            if isinstance(rate, (float, int)):
                detail = f"失败率 {float(rate):.2%}（{failed}/{valid} 个有效尝试）"
            else:
                detail = f"有效尝试数为 {valid}，无法计算失败率"
            if isinstance(base_rate, (float, int)) and isinstance(rate, (float, int)):
                delta = float(rate or 0) - float(base_rate)
                direction = "高于" if delta > 0 else "低于或等于"
                detail += f"；前期基线为 {float(base_rate):.2%}（{base_valid} 个有效尝试），当前{direction}基线"
            elif isinstance(base_rate, (float, int)):
                detail += f"；前期基线为 {float(base_rate):.2%}（{base_valid} 个有效尝试），当前窗口无有效样本"
            else:
                detail += "；匹配基线没有有效样本，无法比较"
            parts.append(f"{region} {start} 至 {end}：{detail}。")
        if events:
            parts.append(
                f"查询窗口内找到 {len(events)} 条变更或事故记录；时间关联仅作为排查线索，不能据此确认因果。"
            )
        if documents:
            parts.append(f"BM25 检索到 {len(documents)} 段可引用的业务文档内容。")
        if not parts:
            parts.append("本次没有取得可用于结论的业务证据。")
        return " ".join(parts)

    @staticmethod
    def _output_evidence_ids(output: Any) -> list[str]:
        ids: list[str] = []
        if isinstance(output, dict):
            direct = output.get("evidence_id")
            if direct:
                ids.append(str(direct))
            for key in ("results", "events"):
                for item in output.get(key, []) or []:
                    if isinstance(item, dict) and item.get("evidence_id"):
                        ids.append(str(item["evidence_id"]))
        return ids

    @staticmethod
    def _synthesis_evidence(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for item in tool_results:
            if item.get("error"):
                continue
            output = item.get("output", {})
            if item.get("tool_name") == "query_payment_metrics":
                evidence.append({"metrics": output})
            elif item.get("tool_name") == "search_knowledge":
                evidence.append({"documents": output.get("results", [])})
            elif item.get("tool_name") == "query_change_events":
                evidence.append({"events": output.get("events", [])})
        return evidence

    @staticmethod
    def _source_text_map(tool_results: list[dict[str, Any]]) -> dict[str, str]:
        source_map: dict[str, str] = {}
        for item in tool_results:
            if item.get("error"):
                continue
            output = item.get("output", {})
            if item.get("tool_name") == "search_knowledge":
                for result in output.get("results", []):
                    source_map[str(result["evidence_id"])] = str(result["text"])
            elif item.get("tool_name") == "query_change_events":
                for result in output.get("events", []):
                    source_map[str(result["evidence_id"])] = str(result["summary"])
            elif item.get("tool_name") == "query_payment_metrics":
                source_map[str(output["evidence_id"])] = json.dumps(
                    {
                        "query_parameters": output["query_parameters"],
                        "metrics": output["metrics"],
                        "baseline": output["baseline"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
        return source_map

    @staticmethod
    def _operations(traces: list[dict[str, Any]]) -> list[str]:
        successful = [item["tool_name"] for item in traces if item.get("status") == "succeeded"]
        operations = [f"只读查询工具已执行：{', '.join(dict.fromkeys(successful))}。"] if successful else []
        operations.append("未执行任何支付写入、配置变更、切流或回滚操作。")
        return operations

    def _budget_failure(self, state: AgentState, code: str) -> dict[str, Any]:
        return {
            "pending_tool_calls": [],
            "status": "partial" if state.get("tool_results") else "failed",
            "error": {"code": code, "message": "The investigation budget was reached."},
        }

    def _fallback_report(
        self,
        investigation_id: str,
        trace_id: str,
        question: str,
        state: AgentState,
        error: dict[str, str],
    ) -> dict[str, Any]:
        return {
            "investigation_id": investigation_id,
            "trace_id": trace_id,
            "mode": self.mode,
            "status": state.get("status", "failed"),
            "summary": "调查未能生成完整报告。",
            "statistical_facts": [],
            "document_evidence": [],
            "system_events": [],
            "hypotheses": [],
            "uncertainties": ["请检查工具、模型配置和运行预算。"],
            "recommendations": [],
            "tool_calls": state.get("tool_traces", []),
            "operations_executed": self._operations(state.get("tool_traces", [])),
            "error": error,
        }


def build_agent(
    settings: Settings,
    tools: dict[str, StructuredTool],
    model_override: Any | None = None,
) -> InvestigationAgent | None:
    if model_override is not None:
        model = model_override
    elif settings.mode == "offline_test":
        model = OfflineFakeModel()
    elif not settings.openai_api_key or not settings.openai_api_key.get_secret_value().strip():
        return None
    else:
        model = LiveLLM(settings, list(tools.values()))
    return InvestigationAgent(
        tools=tools,
        model=model,
        mode=settings.mode,
        max_tool_calls=settings.max_tool_calls,
        max_agent_seconds=settings.max_agent_seconds,
    )
