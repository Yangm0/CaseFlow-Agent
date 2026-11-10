from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from caseflow.db import readonly_transaction
from caseflow.retrieval.embeddings import EmbeddingProvider, content_hash
from caseflow.tools.knowledge import TextChunk, chunk_text


class VectorSearchService:
    def __init__(
        self,
        engine: Engine,
        provider: EmbeddingProvider,
        statement_timeout_ms: int = 3000,
    ):
        self.engine = engine
        self.provider = provider
        self.statement_timeout_ms = statement_timeout_ms
        self._query_cache: dict[str, list[float]] = {}
        self._query_cache_lock = threading.Lock()

    def search(self, query: str, chunks: list[TextChunk], top_k: int) -> list[dict[str, Any]]:
        query_key = hashlib.sha256(
            f"{self.provider.model}\0{self.provider.dimensions}\0{query}".encode()
        ).hexdigest()
        with self._query_cache_lock:
            vector = self._query_cache.get(query_key)
        if vector is None:
            vector = self.provider.embed_query(query)
            with self._query_cache_lock:
                if len(self._query_cache) >= 256:
                    self._query_cache.pop(next(iter(self._query_cache)))
                self._query_cache[query_key] = vector
        if len(vector) != self.provider.dimensions:
            raise ValueError("Query embedding dimension does not match the configured model.")
        current_hashes = {
            (chunk.document_id, chunk.chunk_id): content_hash(chunk.title, chunk.text)
            for chunk in chunks
        }
        if not current_hashes:
            return []
        with readonly_transaction(self.engine, self.statement_timeout_ms) as connection:
            rows = connection.execute(
                text(
                    "WITH current_chunks AS (SELECT * FROM unnest("
                    "CAST(:document_ids AS text[]), CAST(:chunk_ids AS text[]), "
                    "CAST(:content_hashes AS text[])) AS c(document_id, chunk_id, content_hash)) "
                    "SELECT e.document_id, e.chunk_id, e.title, e.content, e.content_hash, "
                    "e.embedding <=> CAST(:embedding AS vector) AS distance "
                    "FROM caseflow_vectors.document_embeddings AS e "
                    "JOIN current_chunks AS c ON c.document_id = e.document_id "
                    "AND c.chunk_id = e.chunk_id AND c.content_hash = e.content_hash "
                    "WHERE e.embedding_model = :model AND e.dimensions = :dimensions "
                    "ORDER BY e.embedding <=> CAST(:embedding AS vector) LIMIT :top_k"
                ),
                {
                    "embedding": _vector_literal(vector),
                    "model": self.provider.model,
                    "dimensions": self.provider.dimensions,
                    "document_ids": [key[0] for key in current_hashes],
                    "chunk_ids": [key[1] for key in current_hashes],
                    "content_hashes": list(current_hashes.values()),
                    "top_k": top_k,
                },
            ).mappings().all()
        results: list[dict[str, Any]] = []
        for row in rows:
            key = (str(row["document_id"]), str(row["chunk_id"]))
            results.append(
                {
                    "evidence_id": f"KNOW-{key[0]}-{key[1]}",
                    "document_id": key[0],
                    "chunk_id": key[1],
                    "title": row["title"],
                    "text": row["content"],
                    "score": round(1 - float(row["distance"]), 6),
                    "retrieval_sources": ["pgvector"],
                }
            )
            if len(results) >= top_k:
                break
        return results


class EmbeddingIndexer:
    """Incrementally indexes changed document chunks using a separately privileged writer."""

    def __init__(
        self,
        read_engine: Engine,
        write_engine: Engine,
        provider: EmbeddingProvider,
        statement_timeout_ms: int = 3000,
    ):
        self.read_engine = read_engine
        self.write_engine = write_engine
        self.provider = provider
        self.statement_timeout_ms = statement_timeout_ms

    def index_documents(self, batch_size: int = 64) -> dict[str, int]:
        with readonly_transaction(self.read_engine, self.statement_timeout_ms) as connection:
            documents = connection.execute(
                text(
                    "SELECT document_id, title, content FROM documents ORDER BY document_id"
                )
            ).mappings().all()
        chunks = [
            chunk
            for row in documents
            for chunk in chunk_text(row["document_id"], row["title"], row["content"])
        ]
        with self.write_engine.connect() as connection:
            existing_rows = connection.execute(
                text(
                    "SELECT document_id, chunk_id, content_hash FROM "
                    "caseflow_vectors.document_embeddings WHERE embedding_model = :model"
                ),
                {"model": self.provider.model},
            ).mappings().all()
        existing = {
            (row["document_id"], row["chunk_id"]): row["content_hash"] for row in existing_rows
        }
        changed = [
            chunk
            for chunk in chunks
            if existing.get((chunk.document_id, chunk.chunk_id))
            != content_hash(chunk.title, chunk.text)
        ]
        indexed = 0
        for offset in range(0, len(changed), batch_size):
            batch = changed[offset : offset + batch_size]
            vectors = self.provider.embed_documents(
                [chunk.title + "\n" + chunk.text for chunk in batch]
            )
            if len(vectors) != len(batch) or any(
                len(vector) != self.provider.dimensions for vector in vectors
            ):
                raise ValueError("Embedding batch shape does not match the configured dimensions.")
            rows = [
                {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "content": chunk.text,
                    "content_hash": content_hash(chunk.title, chunk.text),
                    "model": self.provider.model,
                    "dimensions": self.provider.dimensions,
                    "embedding": _vector_literal(vector),
                }
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
            with self.write_engine.begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO caseflow_vectors.document_embeddings "
                        "(document_id, chunk_id, title, content, content_hash, embedding_model, "
                        "dimensions, embedding) VALUES "
                        "(:document_id, :chunk_id, :title, :content, :content_hash, :model, "
                        ":dimensions, CAST(:embedding AS vector)) "
                        "ON CONFLICT (document_id, chunk_id, embedding_model) DO UPDATE SET "
                        "title = excluded.title, content = excluded.content, "
                        "content_hash = excluded.content_hash, dimensions = excluded.dimensions, "
                        "embedding = excluded.embedding, updated_at = now()"
                    ),
                    rows,
                )
            indexed += len(batch)
        return {"documents": len(documents), "chunks": len(chunks), "indexed_chunks": indexed}


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(format(float(value), ".9g") for value in vector) + "]"
