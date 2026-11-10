from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from caseflow.config import get_settings
from caseflow.db import make_engine
from caseflow.schemas import KnowledgeSearchInput
from caseflow.tools.knowledge import KnowledgeService

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    settings = get_settings()
    if not settings.test_database_url:
        raise SystemExit("TEST_DATABASE_URL is required; retrieval evaluation is test-db only.")
    engine = make_engine(settings.test_database_url)
    if engine.url.database != "caseflow_test":
        engine.dispose()
        raise SystemExit(
            "Refusing retrieval evaluation outside the isolated caseflow_test database."
        )

    cases = json.loads((ROOT / "evals" / "retrieval_relevance.json").read_text(encoding="utf-8"))
    service = KnowledgeService(engine, settings.sql_statement_timeout_ms)
    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            retrieved = service.search(KnowledgeSearchInput(query=case["query"], top_k=5))[
                "results"
            ]
            relevant = set(case["relevant_document_ids"])
            ranked_ids = [item["document_id"] for item in retrieved]
            first_rank = next(
                (rank for rank, doc_id in enumerate(ranked_ids, start=1) if doc_id in relevant),
                None,
            )
            found = relevant.intersection(ranked_ids)
            results.append(
                {
                    "query_id": case["query_id"],
                    "retrieved_document_ids": ranked_ids,
                    "relevant_document_ids": sorted(relevant),
                    "recall_at_5": len(found) / len(relevant) if relevant else None,
                    "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
                    "first_relevant_rank": first_rank,
                }
            )
    finally:
        engine.dispose()

    report = {
        "mode": "retrieval_only_postgresql_bm25",
        "dataset": "caseflow_test synthetic documents",
        "case_count": len(results),
        "mean_recall_at_5": mean(item["recall_at_5"] for item in results),
        "mean_reciprocal_rank": mean(item["reciprocal_rank"] for item in results),
        "cases": results,
    }
    destination = ROOT / ".tmp" / "v1b_retrieval_eval.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
