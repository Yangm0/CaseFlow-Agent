from __future__ import annotations

from typing import Any, Protocol


class Reranker(Protocol):
    """Integration boundary for a configured rerank provider; no mock scorer is supplied."""

    def rerank(self, query: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]: ...
