from __future__ import annotations

import argparse
import sys

from sqlalchemy.engine import make_url

from caseflow.config import get_settings
from caseflow.db import make_engine
from caseflow.retrieval.embeddings import build_embedding_provider
from caseflow.retrieval.vector import EmbeddingIndexer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index changed documents with the configured Embedding API."
    )
    parser.add_argument(
        "--confirm-embedding-api",
        action="store_true",
        help="Confirm that this command may send document text to an external embedding API.",
    )
    parser.add_argument(
        "--database",
        choices=("test", "development"),
        default="test",
        help="Target database; the isolated test database is the default.",
    )
    args = parser.parse_args()
    settings = get_settings()
    if not settings.embedding_index_database_url:
        print("EMBEDDING_INDEX_DATABASE_URL is required for the restricted vector-table writer.")
        return 2
    if settings.embedding_provider == "openai_compatible" and not args.confirm_embedding_api:
        print("Pass --confirm-embedding-api before sending documents to an external provider.")
        return 2
    read_url = settings.test_database_url if args.database == "test" else settings.database_url
    if not read_url:
        print("The requested database URL is not configured.")
        return 2
    expected_database = "caseflow_test" if args.database == "test" else "caseflow"
    read_url = make_url(read_url)
    write_url = make_url(settings.embedding_index_database_url)
    if read_url.database != expected_database:
        print("Refusing to index a database outside the requested CaseFlow environment.")
        return 2
    write_url = write_url.set(database=expected_database)

    provider = build_embedding_provider(settings)
    if provider is None:
        print("Configure EMBEDDING_PROVIDER, EMBEDDING_MODEL, and EMBEDDING_DIMENSIONS first.")
        return 2
    read_engine = make_engine(read_url.render_as_string(hide_password=False))
    write_engine = make_engine(write_url.render_as_string(hide_password=False))
    try:
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
