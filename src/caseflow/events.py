from __future__ import annotations

import asyncio
import json
import threading
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


class ThreadEventHub:
    """Bounded in-process SSE journal; durable thread/report state remains in PostgreSQL."""

    def __init__(self, history_size: int = 200):
        self.history_size = history_size
        self._lock = threading.Lock()
        self._history: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        self._subscribers: dict[str, set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = (
            defaultdict(set)
        )

    def publish(self, thread_id: str, event: str, data: dict[str, Any]) -> dict[str, Any]:
        item = {
            "id": uuid4().hex,
            "event": event,
            "data": data,
            "created_at": datetime.now(UTC).isoformat(),
        }
        with self._lock:
            self._history[thread_id].append(item)
            subscribers = tuple(self._subscribers.get(thread_id, ()))
        for loop, queue in subscribers:
            if loop.is_closed():
                continue
            loop.call_soon_threadsafe(self._offer, queue, item)
        return item

    def latest_event_id(self, thread_id: str) -> str | None:
        with self._lock:
            history = self._history.get(thread_id)
            return history[-1]["id"] if history else None

    @staticmethod
    def _offer(queue: asyncio.Queue, item: dict[str, Any]) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(item)

    async def stream(
        self,
        thread_id: str,
        last_event_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any] | None]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        subscriber = (loop, queue)
        with self._lock:
            history = list(self._history.get(thread_id, ()))
            self._subscribers[thread_id].add(subscriber)
        try:
            if last_event_id:
                ids = [item["id"] for item in history]
                history = (
                    history[ids.index(last_event_id) + 1 :]
                    if last_event_id in ids
                    else history
                )
            seen: set[str] = set()
            for item in history:
                seen.add(item["id"])
                yield item
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield None
                    continue
                if item["id"] not in seen:
                    seen.add(item["id"])
                    yield item
        finally:
            with self._lock:
                self._subscribers.get(thread_id, set()).discard(subscriber)


def format_sse(item: dict[str, Any] | None) -> str:
    if item is None:
        return ": keep-alive\n\n"
    payload = json.dumps(item["data"], ensure_ascii=False, separators=(",", ":"))
    return f"id: {item['id']}\nevent: {item['event']}\ndata: {payload}\n\n"
