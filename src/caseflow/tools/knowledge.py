import re
from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Any
from uuid import uuid4

import jieba
from rank_bm25 import BM25Okapi
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from caseflow.db import readonly_transaction
from caseflow.models import Document
from caseflow.schemas import KnowledgeSearchInput
from caseflow.tools.common import translate_database_error

TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")


@dataclass(frozen=True)
class TextChunk:
    document_id: str
    chunk_id: str
    title: str
    text: str


def tokenize(text: str) -> list[str]:
    segmented = jieba.lcut(text.lower(), HMM=True)
    result: list[str] = []
    for token in segmented:
        for match in TOKEN_RE.findall(token):
            if len(match.strip()) > 0:
                result.append(match.strip().lower())
    return result


def chunk_text(document_id: str, title: str, content: str, max_chars: int = 500) -> list[TextChunk]:
    if len(content) <= max_chars:
        return [TextChunk(document_id, "chunk-000", title, content)]
    chunks: list[TextChunk] = []
    step = max_chars - 60
    for index, start in enumerate(range(0, len(content), step)):
        chunk = content[start : start + max_chars]
        if chunk.strip():
            chunks.append(TextChunk(document_id, f"chunk-{index:03d}", title, chunk))
        if start + max_chars >= len(content):
            break
    return chunks


def rank_chunks(query: str, chunks: list[TextChunk], top_k: int) -> list[dict[str, Any]]:
    return BM25Retriever.build(chunks).search(query, top_k)


@dataclass
class BM25Retriever:
    chunks: list[TextChunk]
    index: BM25Okapi | None

    @classmethod
    def build(cls, chunks: list[TextChunk]) -> "BM25Retriever":
        tokenized_corpus = [tokenize(chunk.title + " " + chunk.text) for chunk in chunks]
        index = BM25Okapi(tokenized_corpus) if any(tokenized_corpus) else None
        return cls(chunks=chunks, index=index)

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks or self.index is None:
            return []
        scores = self.index.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        matches: list[dict[str, Any]] = []
        for chunk_index, raw_score in ranked:
            score = float(raw_score)
            if score <= 0:
                continue
            chunk = self.chunks[chunk_index]
            matches.append(
                {
                    "evidence_id": f"KNOW-{chunk.document_id}-{chunk.chunk_id}",
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "score": round(score, 6),
                }
            )
            if len(matches) >= top_k:
                break
        return matches


class KnowledgeService:
    def __init__(self, engine: Engine, statement_timeout_ms: int = 3000):
        self.engine = engine
        self.statement_timeout_ms = statement_timeout_ms
        self._cache_lock = RLock()
        self._cache_signature: tuple[int, datetime | None] | None = None
        self._retriever: BM25Retriever | None = None

    def refresh_index(self) -> int:
        with self._cache_lock:
            self._cache_signature = None
            self._retriever = None
            retriever = self._load_index()
            return len(retriever.chunks)

    def _load_index(self) -> BM25Retriever:
        with self._cache_lock:
            try:
                with readonly_transaction(self.engine, self.statement_timeout_ms) as connection:
                    count, updated_at = connection.execute(
                        select(func.count(Document.document_id), func.max(Document.updated_at))
                    ).one()
                    signature = (int(count), updated_at)
                    if signature == self._cache_signature and self._retriever is not None:
                        return self._retriever
                    rows = (
                        connection.execute(
                            select(
                                Document.document_id,
                                Document.title,
                                Document.content,
                            ).order_by(Document.document_id)
                        )
                        .mappings()
                        .all()
                    )
            except SQLAlchemyError as error:
                raise translate_database_error(error) from error

            chunks = [
                chunk
                for row in rows
                for chunk in chunk_text(row["document_id"], row["title"], row["content"])
            ]
            self._retriever = BM25Retriever.build(chunks)
            self._cache_signature = signature
            return self._retriever

    def search(self, params: KnowledgeSearchInput) -> dict[str, object]:
        retriever = self._load_index()
        results = retriever.search(params.query, params.top_k)
        return {
            "search_id": f"SRCH-{uuid4().hex[:16]}",
            "query": params.query,
            "retriever": "jieba-tokenized BM25Okapi",
            "index_size_chunks": len(retriever.chunks),
            "results": results,
        }
