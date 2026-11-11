from __future__ import annotations

import hashlib
from collections.abc import Sequence
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


def content_hash(title: str, text: str) -> str:
    return hashlib.sha256(f"{title}\0{text}".encode()).hexdigest()
