import logging
from datetime import datetime

from langchain_core.tools import StructuredTool
from sqlalchemy.engine import Engine

from caseflow.config import Settings
from caseflow.retrieval.embeddings import build_embedding_provider
from caseflow.retrieval.vector import VectorSearchService
from caseflow.schemas import (
    ChangeEventsInput,
    GroupDimension,
    KnowledgeSearchInput,
    PaymentMetricsInput,
)
from caseflow.tools.changes import ChangeEventsService
from caseflow.tools.knowledge import KnowledgeService
from caseflow.tools.metrics import MetricsService

ALLOWED_TOOL_NAMES = frozenset({"query_payment_metrics", "search_knowledge", "query_change_events"})
logger = logging.getLogger(__name__)


class _UnavailableVectorSearch:
    def search(self, query, chunks, top_k):
        raise RuntimeError("Configured vector search is unavailable.")


def build_tool_registry(
    engine: Engine,
    statement_timeout_ms: int,
    settings: Settings | None = None,
) -> dict[str, StructuredTool]:
    metrics = MetricsService(engine, statement_timeout_ms)
    vector_search = None
    if settings:
        try:
            provider = build_embedding_provider(settings)
            if provider is not None:
                vector_search = VectorSearchService(engine, provider, statement_timeout_ms)
        except Exception as error:
            logger.warning(
                "Embedding provider unavailable (%s); BM25 fallback remains active.",
                type(error).__name__,
            )
            vector_search = _UnavailableVectorSearch()
    knowledge = KnowledgeService(engine, statement_timeout_ms, vector_search=vector_search)
    changes = ChangeEventsService(engine, statement_timeout_ms)

    def query_payment_metrics(
        region: str,
        start_at: datetime,
        end_at: datetime,
        group_by: list[GroupDimension] | None = None,
    ):
        params = PaymentMetricsInput(
            region=region,
            start_at=start_at,
            end_at=end_at,
            group_by=group_by or [],
        )
        return metrics.query(params)

    def search_knowledge(query: str, top_k: int = 5):
        return knowledge.search(KnowledgeSearchInput(query=query, top_k=top_k))

    def query_change_events(
        start_at: datetime,
        end_at: datetime,
        region: str | None = None,
    ):
        return changes.query(ChangeEventsInput(region=region, start_at=start_at, end_at=end_at))

    return {
        "query_payment_metrics": StructuredTool.from_function(
            func=query_payment_metrics,
            name="query_payment_metrics",
            description=(
                "Read payment-attempt counts, failures, and failure rate. Include seven matching "
                "prior-day windows (or an equal-duration preceding baseline) and optional "
                "allow-listed groupings from PostgreSQL. "
                "Pass timezone-aware timestamps. The interval is [start_at, end_at)."
            ),
            args_schema=PaymentMetricsInput,
        ),
        "search_knowledge": StructuredTool.from_function(
            func=search_knowledge,
            name="search_knowledge",
            description=(
                "Search generated payment manuals and runbooks with Chinese-tokenized BM25. "
                "Returns source text and stable document/chunk citations."
            ),
            args_schema=KnowledgeSearchInput,
        ),
        "query_change_events": StructuredTool.from_function(
            func=query_change_events,
            name="query_change_events",
            description=(
                "Read service changes and incident records in a timezone-aware "
                "[start_at, end_at) interval. In a multi-tool anomaly investigation, include "
                "the two hours before the metric window unless the user requests exact-window "
                "events only. "
                "A time overlap is a lead, not proof of causation."
            ),
            args_schema=ChangeEventsInput,
        ),
    }
