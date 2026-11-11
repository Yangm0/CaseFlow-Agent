from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from caseflow.config import get_settings
from caseflow.db import make_engine
from caseflow.retrieval.embeddings import build_embedding_provider
from caseflow.retrieval.vector import VectorSearchService
from caseflow.tools.knowledge import KnowledgeService, _reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure labelled BM25/vector/hybrid retrieval.")
    parser.add_argument("--split", choices=("development", "validation", "all"), default="all")
    parser.add_argument("--retriever", choices=("bm25", "vector", "hybrid"), default="bm25")
    parser.add_argument("--k", type=int, choices=range(1, 11), default=5)
    parser.add_argument(
        "--confirm-embedding-api",
        action="store_true",
        help="Confirm that query text may be sent to the configured external embedding API.",
    )
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
    vector_search = None
    embedding_provider = None
    if args.retriever != "bm25":
        if settings.embedding_provider == "openai_compatible" and not args.confirm_embedding_api:
            engine.dispose()
            raise SystemExit(
                "Pass --confirm-embedding-api before sending queries to an external provider."
            )
        embedding_provider = build_embedding_provider(settings)
        if embedding_provider is None:
            engine.dispose()
            raise SystemExit("Configure an embedding provider, model, and dimensions first.")
        vector_search = VectorSearchService(
            engine,
            embedding_provider,
            settings.sql_statement_timeout_ms,
        )
    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            retriever = service._load_index()
            bm25 = retriever.search(case["query"], top_k=len(retriever.chunks))
            if args.retriever == "vector":
                ranked_chunks = vector_search.search(
                    case["query"], retriever.chunks, len(retriever.chunks)
                )
            elif args.retriever == "hybrid":
                vector = vector_search.search(
                    case["query"], retriever.chunks, len(retriever.chunks)
                )
                ranked_chunks = _reciprocal_rank_fusion(bm25, vector, len(retriever.chunks))
            else:
                ranked_chunks = bm25
            ranked_document_ids = list(dict.fromkeys(item["document_id"] for item in ranked_chunks))
            retrieved_document_ids = ranked_document_ids[: args.k]
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
            gains = [1.0 if doc_id in relevant else 0.0 for doc_id in retrieved_document_ids]
            dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
            ideal_count = min(len(relevant), args.k)
            ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
            results.append(
                {
                    "query_id": case["query_id"],
                    "split": case["split"],
                    "retrieved_document_ids": retrieved_document_ids,
                    "relevant_document_ids": sorted(relevant),
                    f"recall_at_{args.k}": len(found) / len(relevant) if relevant else None,
                    "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
                    f"ndcg_at_{args.k}": dcg / ideal_dcg if ideal_dcg else None,
                    "first_relevant_rank": first_rank,
                    "relevant_document_count": len(relevant),
                    f"relevant_found_at_{args.k}": len(found),
                }
            )
    finally:
        engine.dispose()

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        recall_key = f"recall_at_{args.k}"
        ndcg_key = f"ndcg_at_{args.k}"
        recalls = [item[recall_key] for item in items if item[recall_key] is not None]
        reciprocal_ranks = [item["reciprocal_rank"] for item in items]
        ndcgs = [item[ndcg_key] for item in items if item[ndcg_key] is not None]
        return {
            "query_count": len(items),
            f"recall_at_{args.k}_scored_queries": len(recalls),
            f"mean_recall_at_{args.k}": mean(recalls) if recalls else None,
            "mrr_scored_queries": len(reciprocal_ranks),
            "mean_reciprocal_rank": mean(reciprocal_ranks) if reciprocal_ranks else None,
            f"ndcg_at_{args.k}_scored_queries": len(ndcgs),
            f"mean_ndcg_at_{args.k}": mean(ndcgs) if ndcgs else None,
        }

    report = {
        "mode": f"retrieval_only_postgresql_{args.retriever}",
        "dataset": "caseflow_test synthetic documents",
        "embedding_model": embedding_provider.model if embedding_provider else None,
        "case_count": len(results),
        f"recall_at_{args.k}_definition": (
            f"unique relevant document IDs in top {args.k} unique document results / all labelled "
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
    destination = ROOT / ".tmp" / f"v1c_retrieval_{args.retriever}_eval.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
