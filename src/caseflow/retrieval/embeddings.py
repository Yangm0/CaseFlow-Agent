from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from caseflow.config import Settings


class EmbeddingProvider(Protocol):
    model: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbeddingProvider:
    """OpenAI-compatible Embeddings API adapter; calls occur only when methods are invoked."""

    def __init__(self, settings: Settings):
        if not (
            settings.embedding_api_key
            and settings.embedding_api_key.get_secret_value().strip()
            and settings.embedding_model
            and settings.embedding_dimensions
        ):
            raise ValueError("Embedding API key, model, and dimensions are required.")
        from langchain_openai import OpenAIEmbeddings

        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self.client = OpenAIEmbeddings(
            model=self.model,
            dimensions=self.dimensions,
            api_key=settings.embedding_api_key.get_secret_value(),
            base_url=settings.embedding_base_url,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._validate(self.client.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        values = self._validate([self.client.embed_query(text)])
        return values[0]

    def _validate(self, vectors: Sequence[Sequence[float]]) -> list[list[float]]:
        result = [[float(value) for value in vector] for vector in vectors]
        if any(len(vector) != self.dimensions for vector in result):
            raise ValueError("Embedding response dimension does not match EMBEDDING_DIMENSIONS.")
        if any(
            not value == value or abs(value) == float("inf")
            for vector in result
            for value in vector
        ):
            raise ValueError("Embedding response contains a non-finite value.")
        return result


class FastEmbedEmbeddingProvider:
    """Small local CPU embedding adapter, loaded only when explicitly configured."""

    def __init__(self, settings: Settings):
        if not settings.embedding_model or not settings.embedding_dimensions:
            raise ValueError("EMBEDDING_MODEL and EMBEDDING_DIMENSIONS are required.")
        from fastembed import TextEmbedding

        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        cache_dir = Path(settings.embedding_cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = TextEmbedding(
            model_name=self.model,
            threads=2,
            cache_dir=str(cache_dir),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._validate(list(self.client.embed(texts)))

    def embed_query(self, text: str) -> list[float]:
        query = text
        if self.model.casefold() == "baai/bge-small-zh-v1.5":
            query = f"为这个句子生成表示以用于检索相关文章：{text}"
        return self._validate(list(self.client.query_embed(query)))[0]

    def _validate(self, vectors: Sequence[Sequence[float]]) -> list[list[float]]:
        result = [[float(value) for value in vector] for vector in vectors]
        if any(len(vector) != self.dimensions for vector in result):
            raise ValueError("Local embedding response dimension does not match configuration.")
        if any(
            not value == value or abs(value) == float("inf")
            for vector in result
            for value in vector
        ):
            raise ValueError("Local embedding response contains a non-finite value.")
        return result


def build_embedding_provider(settings: Settings) -> EmbeddingProvider | None:
    """Build a provider only when its configuration explicitly opts into embeddings."""
    if settings.embedding_provider == "fastembed":
        return FastEmbedEmbeddingProvider(settings)
    if (
        settings.embedding_api_key
        and settings.embedding_api_key.get_secret_value().strip()
        and settings.embedding_model
        and settings.embedding_dimensions
    ):
        return OpenAIEmbeddingProvider(settings)
    return None


def content_hash(title: str, text: str) -> str:
    return hashlib.sha256(f"{title}\0{text}".encode()).hexdigest()
