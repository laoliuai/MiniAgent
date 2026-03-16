"""SharedState tool wrappers for LLM access."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Tool, ToolResult

if TYPE_CHECKING:
    from mini_agent.shared_state import SharedState


class StateReadTool(Tool):
    """Read a value from shared state."""

    def __init__(self, shared_state: SharedState):
        self._state = shared_state

    @property
    def name(self) -> str:
        return "state_read"

    @property
    def description(self) -> str:
        return "Read a value from the shared data store by key."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The key to read"},
            },
            "required": ["key"],
        }

    async def execute(self, key: str) -> ToolResult:  # type: ignore[override]
        try:
            value = await self._state.get(key)
            if value is None:
                return ToolResult(success=True, content=f"Key '{key}' not found (null).")
            return ToolResult(success=True, content=str(value))
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class StateWriteTool(Tool):
    """Write a value to shared state."""

    def __init__(self, shared_state: SharedState, agent_id: str):
        self._state = shared_state
        self._agent_id = agent_id

    @property
    def name(self) -> str:
        return "state_write"

    @property
    def description(self) -> str:
        return (
            "Write a value to the shared data store. "
            "Value is stored as a string. Use schema_hint to describe the data type."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The key to write"},
                "value": {"type": "string", "description": "The value to store"},
                "schema_hint": {
                    "type": "string",
                    "description": "Type hint for other agents, e.g. 'JSON array with 10 items'",
                    "default": "",
                },
            },
            "required": ["key", "value"],
        }

    async def execute(self, key: str, value: str, schema_hint: str = "") -> ToolResult:  # type: ignore[override]
        try:
            await self._state.set(
                key, value, agent_id=self._agent_id, schema_hint=schema_hint
            )
            return ToolResult(success=True, content=f"Stored '{key}' successfully.")
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class StateListTool(Tool):
    """List keys in shared state."""

    def __init__(self, shared_state: SharedState):
        self._state = shared_state

    @property
    def name(self) -> str:
        return "state_list"

    @property
    def description(self) -> str:
        return "List all keys in the shared data store, optionally filtered by prefix."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "prefix": {
                    "type": "string",
                    "description": "Optional prefix filter",
                    "default": "",
                },
            },
        }

    async def execute(self, prefix: str = "") -> ToolResult:  # type: ignore[override]
        try:
            keys = await self._state.keys(prefix=prefix)
            if not keys:
                return ToolResult(success=True, content="No keys found.")
            snapshot = self._state.snapshot()
            lines = [f"- {k}: {snapshot.get(k, '')}" for k in keys]
            return ToolResult(success=True, content="\n".join(lines))
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))
