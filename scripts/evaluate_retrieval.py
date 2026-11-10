from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from caseflow.config import get_settings
from caseflow.db import make_engine
from caseflow.tools.knowledge import KnowledgeService

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure BM25 retrieval on labelled documents.")
    parser.add_argument("--split", choices=("development", "validation", "all"), default="all")
    args = parser.parse_args()
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
    if args.split != "all":
        cases = [case for case in cases if case["split"] == args.split]
    if not cases:
        engine.dispose()
        raise SystemExit(f"No retrieval queries found for split={args.split}.")
    service = KnowledgeService(engine, settings.sql_statement_timeout_ms)
    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            retriever = service._load_index()
            chunks = retriever.search(case["query"], top_k=len(retriever.chunks))
            ranked_document_ids = list(dict.fromkeys(item["document_id"] for item in chunks))
            retrieved_document_ids = ranked_document_ids[:5]
            relevant = set(case["relevant_document_ids"])
            first_rank = next(
                (
                    rank
                    for rank, doc_id in enumerate(retrieved_document_ids, start=1)
                    if doc_id in relevant
                ),
                None,
            )
            found = relevant.intersection(retrieved_document_ids)
            results.append(
                {
                    "query_id": case["query_id"],
                    "split": case["split"],
                    "retrieved_document_ids": retrieved_document_ids,
                    "relevant_document_ids": sorted(relevant),
                    "recall_at_5": len(found) / len(relevant) if relevant else None,
                    "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
                    "first_relevant_rank": first_rank,
                    "relevant_document_count": len(relevant),
                    "relevant_found_at_5": len(found),
                }
            )
    finally:
        engine.dispose()

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        recalls = [item["recall_at_5"] for item in items if item["recall_at_5"] is not None]
        reciprocal_ranks = [item["reciprocal_rank"] for item in items]
        return {
            "query_count": len(items),
            "recall_at_5_scored_queries": len(recalls),
            "mean_recall_at_5": mean(recalls) if recalls else None,
            "mrr_scored_queries": len(reciprocal_ranks),
            "mean_reciprocal_rank": mean(reciprocal_ranks) if reciprocal_ranks else None,
        }

    report = {
        "mode": "retrieval_only_postgresql_bm25",
        "dataset": "caseflow_test synthetic documents",
        "case_count": len(results),
        "recall_at_5_definition": (
            "unique relevant document IDs in top five unique document results / all labelled "
            "relevant document IDs; collapse all chunks by document ID before truncation"
        ),
        "mrr_definition": (
            "reciprocal rank of the first relevant unique document; zero if none is found"
        ),
        **summarize(results),
        "by_split": {
            split: summarize([item for item in results if item["split"] == split])
            for split in sorted({item["split"] for item in results})
        },
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
