"""SharedState: cross-agent structured data store."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class StateEntry:
    """Single entry in SharedState."""

    key: str
    value: Any
    written_by: str
    written_at: datetime
    schema_hint: str = ""
    ttl_turns: Optional[int] = None


class SharedState:
    """Thread-safe cross-agent data store using asyncio.Lock."""

    def __init__(self):
        self._store: dict[str, StateEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            e = self._store.get(key)
            return e.value if e else None

    async def set(
        self,
        key: str,
        value: Any,
        agent_id: str,
        schema_hint: str = "",
        ttl_turns: int | None = None,
    ):
        async with self._lock:
            self._store[key] = StateEntry(
                key=key,
                value=value,
                written_by=agent_id,
                written_at=datetime.now(),
                schema_hint=schema_hint,
                ttl_turns=ttl_turns,
            )

    async def keys(self, prefix: str = "") -> list[str]:
        async with self._lock:
            return [k for k in self._store if k.startswith(prefix)]

    async def delete(self, key: str) -> bool:
        async with self._lock:
            return self._store.pop(key, None) is not None

    def snapshot(self) -> dict[str, str]:
        """Synchronous. Token-efficient summary for system prompt injection.

        Keys + schema hints only, no actual values.
        Safe without lock: read-only dict iteration in single-threaded async loop.
        """
        return {
            k: f"{e.schema_hint} (by {e.written_by})"
            for k, e in self._store.items()
        }
