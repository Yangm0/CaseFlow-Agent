import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import jieba
from rank_bm25 import BM25Okapi
from sqlalchemy import select
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
    query_tokens = tokenize(query)
    if not query_tokens or not chunks:
        return []
    tokenized_corpus = [tokenize(chunk.title + " " + chunk.text) for chunk in chunks]
    if not any(tokenized_corpus):
        return []
    index = BM25Okapi(tokenized_corpus)
    scores = index.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    matches: list[dict[str, Any]] = []
    for chunk_index, raw_score in ranked:
        score = float(raw_score)
        if score <= 0:
            continue
        chunk = chunks[chunk_index]
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

    def search(self, params: KnowledgeSearchInput) -> dict[str, object]:
        try:
            with readonly_transaction(self.engine, self.statement_timeout_ms) as connection:
                rows = connection.execute(
                    select(
                        Document.document_id,
                        Document.title,
                        Document.content,
                    ).order_by(Document.document_id)
                ).mappings().all()
        except SQLAlchemyError as error:
            raise translate_database_error(error) from error

        chunks = [
            chunk
            for row in rows
            for chunk in chunk_text(row["document_id"], row["title"], row["content"])
        ]
        results = rank_chunks(params.query, chunks, params.top_k)
        return {
            "search_id": f"SRCH-{uuid4().hex[:16]}",
            "query": params.query,
            "retriever": "jieba-tokenized BM25Okapi",
            "index_size_chunks": len(chunks),
            "results": results,
        }
