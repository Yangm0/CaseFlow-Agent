from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

RegionName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]
REGION_ALIASES = {
    "beijing": "北京",
    "shanghai": "上海",
    "shenzhen": "深圳",
    "hangzhou": "杭州",
}


def canonical_region(value: str) -> str:
    normalized = value.strip()
    return REGION_ALIASES.get(normalized.casefold(), normalized)


class GroupDimension(StrEnum):
    SERVICE_NAME = "service_name"
    STATUS = "status"
    HOUR_OF_DAY = "hour_of_day"


class TimeWindowInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_at: datetime
    end_at: datetime

    @field_validator("start_at", "end_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a UTC offset")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_window(self):
        if self.start_at >= self.end_at:
            raise ValueError("start_at must be earlier than end_at")
        if self.end_at - self.start_at > timedelta(days=31):
            raise ValueError("time range cannot exceed 31 days")
        return self


class PaymentMetricsInput(TimeWindowInput):
    region: RegionName
    group_by: list[GroupDimension] = Field(default_factory=list, max_length=2)

    @field_validator("region")
    @classmethod
    def normalize_region(cls, value: str) -> str:
        return canonical_region(value)

    @field_validator("group_by")
    @classmethod
    def unique_grouping(cls, values: list[GroupDimension]) -> list[GroupDimension]:
        if len(values) != len(set(values)):
            raise ValueError("group_by values must be unique")
        return values


class KnowledgeSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=500)]
    top_k: int = Field(default=5, ge=1, le=10)


class ChangeEventsInput(TimeWindowInput):
    region: RegionName | None = None

    @field_validator("region")
    @classmethod
    def normalize_region(cls, value: str | None) -> str | None:
        return canonical_region(value) if value else value


class InvestigationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=3, max_length=2000)
    ]


class EvidenceQuote(BaseModel):
    evidence_id: str
    quote: str = Field(min_length=4, max_length=600)


class Hypothesis(BaseModel):
    statement: str = Field(min_length=4, max_length=700)
    confidence: Literal["low", "medium", "high"] = "low"
    citations: list[EvidenceQuote] = Field(default_factory=list, max_length=5)


class SynthesisDraft(BaseModel):
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=5)
    uncertainties: list[str] = Field(default_factory=list, max_length=8)
    recommendations: list[str] = Field(default_factory=list, max_length=8)


class ToolTrace(BaseModel):
    tool_name: str
    arguments: dict[str, object]
    status: Literal["succeeded", "failed", "skipped"]
    elapsed_ms: int = Field(ge=0)
    evidence_ids: list[str] = Field(default_factory=list)
    error_code: str | None = None


class ModelCallTrace(BaseModel):
    stage: Literal["decision", "synthesis"]
    model: str
    method: str
    status: Literal["succeeded", "failed"]
    elapsed_ms: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    error_code: str | None = None
    error_type: str | None = None
    provider_status_code: int | None = Field(default=None, ge=100, le=599)


class TokenUsageSummary(BaseModel):
    available: bool = False
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    unavailable_calls: int = Field(default=0, ge=0)


class CitationValidationSummary(BaseModel):
    hypotheses_received: int = Field(default=0, ge=0)
    hypotheses_kept: int = Field(default=0, ge=0)
    citations_received: int = Field(default=0, ge=0)
    citations_kept: int = Field(default=0, ge=0)
    citations_rejected: int = Field(default=0, ge=0)
    duplicate_citations_removed: int = Field(default=0, ge=0)


class MetricEvidence(BaseModel):
    evidence_id: str
    query_parameters: dict[str, object]
    metrics: dict[str, object]
    baseline: dict[str, object]
    groups: list[dict[str, object]]
    calculation: str


class DocumentEvidence(BaseModel):
    evidence_id: str
    document_id: str
    chunk_id: str
    title: str
    text: str
    score: float


class ChangeEvidence(BaseModel):
    evidence_id: str
    event_type: Literal["change", "incident"]
    event_id: str
    occurred_at: str
    region: str | None
    service_name: str
    summary: str
    status: str | None = None


class InvestigationReport(BaseModel):
    investigation_id: str
    trace_id: str
    mode: Literal["offline_test", "live"]
    status: Literal["completed", "partial", "failed", "blocked"]
    summary: str
    direct_answer: str | None = None
    statistical_facts: list[MetricEvidence]
    document_evidence: list[DocumentEvidence]
    system_events: list[ChangeEvidence]
    hypotheses: list[Hypothesis]
    uncertainties: list[str]
    recommendations: list[str]
    tool_calls: list[ToolTrace]
    operations_executed: list[str]
    model_calls: list[ModelCallTrace] = Field(default_factory=list)
    token_usage: TokenUsageSummary = Field(default_factory=TokenUsageSummary)
    citation_validation: CitationValidationSummary = Field(
        default_factory=CitationValidationSummary
    )
    elapsed_ms: int = Field(default=0, ge=0)
    error: dict[str, str] | None = None
