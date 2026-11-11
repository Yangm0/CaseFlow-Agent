from __future__ import annotations

import argparse
import sys

from caseflow.config import get_settings
from caseflow.db import make_engine
from caseflow.retrieval.embeddings import OpenAIEmbeddingProvider
from caseflow.retrieval.vector import EmbeddingIndexer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index changed documents with the configured Embedding API."
    )
    parser.add_argument(
        "--confirm-embedding-api",
        action="store_true",
        help="Confirm that this command may send document text to the configured provider.",
    )
    args = parser.parse_args()
    if not args.confirm_embedding_api:
        print("Pass --confirm-embedding-api to send document text to the configured provider.")
        return 2
    settings = get_settings()
    if not (
        settings.embedding_api_key
        and settings.embedding_model
        and settings.embedding_dimensions
    ):
        print("EMBEDDING_API_KEY, EMBEDDING_MODEL, and EMBEDDING_DIMENSIONS are required.")
        return 2
    if not settings.embedding_index_database_url:
        print("EMBEDDING_INDEX_DATABASE_URL is required for the restricted vector-table writer.")
        return 2

    read_engine = make_engine(settings.database_url)
    write_engine = make_engine(settings.embedding_index_database_url)
    try:
        provider = OpenAIEmbeddingProvider(settings)
        result = EmbeddingIndexer(
            read_engine,
            write_engine,
            provider,
            settings.sql_statement_timeout_ms,
        ).index_documents()
    finally:
        read_engine.dispose()
        write_engine.dispose()
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
