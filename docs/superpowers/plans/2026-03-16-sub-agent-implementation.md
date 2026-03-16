# Sub-Agent Architecture Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sub-agent delegation, SharedState, and Session API to MiniAgent per the design spec.

**Architecture:** AgentConfig dataclass replaces scattered Agent params. DelegationTool and StateTools are standard Tool subclasses auto-registered by Agent. Session wraps Agent + SharedState with factory methods. CLI uses Session.create() with YAML-configured sub-agents.

**Tech Stack:** Python 3.10+, asyncio, dataclasses, pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-03-16-sub-agent-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|---|---|
| `mini_agent/agent_config.py` | AgentConfig dataclass, SubAgentRunner type alias |
| `mini_agent/shared_state.py` | SharedState store + StateEntry |
| `mini_agent/tools/delegation_tool.py` | DelegationTool (Tool subclass) |
| `mini_agent/tools/state_tools.py` | StateReadTool, StateWriteTool, StateListTool |
| `mini_agent/session.py` | Session API with create() and create_orchestrator() |
| `tests/test_agent_config.py` | AgentConfig unit tests |
| `tests/test_shared_state.py` | SharedState unit tests |
| `tests/test_state_tools.py` | State tool unit tests |
| `tests/test_delegation_tool.py` | DelegationTool unit tests |
| `tests/test_session.py` | Session factory unit tests |
| `tests/test_sub_agent_integration.py` | End-to-end delegation + SharedState tests |

### Modified Files
| File | Changes |
|---|---|
| `mini_agent/agent.py` | Constructor takes AgentConfig; add register_sub_agent, _register_auto_tools, _make_sub_agent_runner, _rebuild_system_block; dual step limits |
| `mini_agent/config.py` | Rename AgentConfig → AgentSettings; add SubAgentEntry; update from_yaml |
| `mini_agent/cli.py` | Use Session.create(); build sub-agent configs from YAML |
| `mini_agent/llm/base.py` | Add model param to generate() and generate_stream() |
| `mini_agent/llm/llm_wrapper.py` | Forward model param |
| `mini_agent/llm/anthropic_client.py` | Use model override in generate_stream() |
| `mini_agent/llm/openai_client.py` | Use model override in generate_stream() |

---

## Chunk 1: Foundation (AgentConfig, SharedState, LLMClient)

### Task 1: AgentConfig Dataclass

> **Note:** This creates `mini_agent/agent_config.py` with a new `AgentConfig` dataclass. The existing Pydantic `AgentConfig` in `config.py` is renamed to `AgentSettings` in Task 8. Until Task 8 is complete, both names coexist — they live in different modules (`agent_config.AgentConfig` vs `config.AgentConfig`) so there is no import collision, but be aware of the temporary naming overlap.

**Files:**
- Create: `mini_agent/agent_config.py`
- Test: `tests/test_agent_config.py`

- [ ] **Step 1: Write tests for AgentConfig**

```python
# tests/test_agent_config.py
"""Tests for AgentConfig dataclass."""
from mini_agent.agent_config import AgentConfig


class TestAgentConfigDefaults:
    def test_default_values(self):
        config = AgentConfig()
        assert config.agent_id == "main"
        assert config.name == "Assistant"
        assert config.description == ""
        assert config.model is None
        assert config.system_prompt == ""
        assert config.tools == []
        assert config.context_config is None
        assert config.can_delegate is False
        assert config.max_delegation_depth == 1
        assert config.max_steps_per_turn == 30
        assert config.max_steps_total == 50
        assert config.state_access == "readwrite"

    def test_custom_values(self):
        config = AgentConfig(
            agent_id="coder",
            name="Coder",
            description="Writes code",
            model="gpt-4",
            system_prompt="You are a coder.",
            can_delegate=True,
            max_delegation_depth=2,
            max_steps_per_turn=10,
            max_steps_total=20,
            state_access="read",
        )
        assert config.agent_id == "coder"
        assert config.name == "Coder"
        assert config.model == "gpt-4"
        assert config.can_delegate is True
        assert config.max_delegation_depth == 2
        assert config.state_access == "read"

    def test_tools_list_independence(self):
        """Each AgentConfig gets its own tools list (no shared mutable default)."""
        c1 = AgentConfig()
        c2 = AgentConfig()
        c1.tools.append("fake_tool")
        assert c2.tools == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_agent_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mini_agent.agent_config'`

- [ ] **Step 3: Implement AgentConfig**

```python
# mini_agent/agent_config.py
"""AgentConfig: runtime configuration describing what an agent IS."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


# Factory callback type for sub-agent execution
# (sub_config, task_message) -> result_text
SubAgentRunner = Callable[["AgentConfig", str], Awaitable[str]]


@dataclass
class AgentConfig:
    """Runtime configuration for an Agent instance.

    Describes identity, capabilities, and limits.
    Infrastructure dependencies (llm_client, workspace_dir, logger, etc.)
    are passed to the Agent constructor separately.
    """

    # Identity
    agent_id: str = "main"
    name: str = "Assistant"
    description: str = ""

    # LLM
    model: Optional[str] = None  # None = use LLMClient's default model

    # System prompt
    system_prompt: str = ""

    # Tools (Tool object list)
    tools: list[Any] = field(default_factory=list)

    # Context management
    context_config: Optional[Any] = None  # ContextConfig | None

    # Delegation
    can_delegate: bool = False
    max_delegation_depth: int = 1

    # Limits
    max_steps_per_turn: int = 30
    max_steps_total: int = 50

    # SharedState access
    state_access: str = "readwrite"  # "read" | "write" | "readwrite" | "none"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent_config.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mini_agent/agent_config.py tests/test_agent_config.py
git commit -m "feat: add AgentConfig dataclass"
```

---

### Task 2: SharedState

**Files:**
- Create: `mini_agent/shared_state.py`
- Test: `tests/test_shared_state.py`

- [ ] **Step 1: Write tests for SharedState**

```python
# tests/test_shared_state.py
"""Tests for SharedState cross-agent data store."""
import asyncio
from mini_agent.shared_state import SharedState, StateEntry


class TestSharedStateBasicOps:
    async def test_set_and_get(self):
        state = SharedState()
        await state.set("key1", "value1", agent_id="main")
        result = await state.get("key1")
        assert result == "value1"

    async def test_get_nonexistent_returns_none(self):
        state = SharedState()
        result = await state.get("missing")
        assert result is None

    async def test_overwrite_value(self):
        state = SharedState()
        await state.set("key1", "old", agent_id="main")
        await state.set("key1", "new", agent_id="sub1")
        result = await state.get("key1")
        assert result == "new"

    async def test_keys_all(self):
        state = SharedState()
        await state.set("a", 1, agent_id="main")
        await state.set("b", 2, agent_id="main")
        keys = await state.keys()
        assert sorted(keys) == ["a", "b"]

    async def test_keys_with_prefix(self):
        state = SharedState()
        await state.set("data.sales", 1, agent_id="main")
        await state.set("data.costs", 2, agent_id="main")
        await state.set("config.model", "x", agent_id="main")
        keys = await state.keys(prefix="data.")
        assert sorted(keys) == ["data.costs", "data.sales"]

    async def test_delete_existing(self):
        state = SharedState()
        await state.set("key1", "val", agent_id="main")
        result = await state.delete("key1")
        assert result is True
        assert await state.get("key1") is None

    async def test_delete_nonexistent(self):
        state = SharedState()
        result = await state.delete("missing")
        assert result is False


class TestSharedStateSnapshot:
    async def test_snapshot_empty(self):
        state = SharedState()
        assert state.snapshot() == {}

    async def test_snapshot_includes_schema_hint(self):
        state = SharedState()
        await state.set("df", "data", agent_id="analyst", schema_hint="DataFrame(200 rows)")
        snap = state.snapshot()
        assert "df" in snap
        assert "DataFrame(200 rows)" in snap["df"]
        assert "analyst" in snap["df"]

    async def test_snapshot_is_sync(self):
        """snapshot() should be callable without await."""
        state = SharedState()
        await state.set("k", "v", agent_id="main")
        # This should NOT be a coroutine
        result = state.snapshot()
        assert isinstance(result, dict)


class TestStateEntry:
    async def test_entry_metadata(self):
        state = SharedState()
        await state.set("key1", "val", agent_id="sub1", schema_hint="str", ttl_turns=5)
        # Access internal to verify metadata stored correctly
        entry = state._store["key1"]
        assert entry.written_by == "sub1"
        assert entry.schema_hint == "str"
        assert entry.ttl_turns == 5
        assert entry.written_at is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_shared_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mini_agent.shared_state'`

- [ ] **Step 3: Implement SharedState**

```python
# mini_agent/shared_state.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_shared_state.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add mini_agent/shared_state.py tests/test_shared_state.py
git commit -m "feat: add SharedState cross-agent data store"
```

---

### Task 3: LLMClient Per-Call Model Override

**Files:**
- Modify: `mini_agent/llm/base.py:49-108` (generate + generate_stream signatures)
- Modify: `mini_agent/llm/llm_wrapper.py:115-138` (forward model param in both generate + generate_stream)
- Modify: `mini_agent/llm/anthropic_client.py:201-221` (use model override)
- Modify: `mini_agent/llm/openai_client.py:205-225` (use model override)
- Test: `tests/test_llm_model_override.py`

- [ ] **Step 1: Write tests for model override**

```python
# tests/test_llm_model_override.py
"""Tests for per-call model override in LLM clients."""
import inspect
import pytest
from collections.abc import AsyncGenerator

from mini_agent.llm.base import LLMClientBase
from mini_agent.schema import LLMStreamChunk, LLMStreamChunkType, Message


class FakeClientWithModelCapture(LLMClientBase):
    """Fake client that captures the model parameter passed to generate_stream."""

    def __init__(self):
        super().__init__(api_key="fake", api_base="http://fake", model="default-model")
        self.captured_model = None

    async def generate_stream(self, messages, tools=None, model=None):
        self.captured_model = model or self.model
        yield LLMStreamChunk(type=LLMStreamChunkType.TEXT_DELTA, content="ok")
        yield LLMStreamChunk(type=LLMStreamChunkType.DONE, finish_reason="stop")

    def _prepare_request(self, messages, tools=None):
        return {}

    def _convert_messages(self, messages):
        return None, []


class TestModelOverrideSignature:
    def test_generate_stream_accepts_model_param(self):
        sig = inspect.signature(LLMClientBase.generate_stream)
        assert "model" in sig.parameters
        assert sig.parameters["model"].default is None

    def test_generate_accepts_model_param(self):
        sig = inspect.signature(LLMClientBase.generate)
        assert "model" in sig.parameters
        assert sig.parameters["model"].default is None


class TestModelOverrideBehavior:
    async def test_model_none_uses_default(self):
        client = FakeClientWithModelCapture()
        await client.generate([Message(role="user", content="hi")], model=None)
        assert client.captured_model == "default-model"

    async def test_model_override_forwarded(self):
        client = FakeClientWithModelCapture()
        await client.generate(
            [Message(role="user", content="hi")], model="custom-model"
        )
        assert client.captured_model == "custom-model"

    async def test_generate_stream_model_override(self):
        client = FakeClientWithModelCapture()
        chunks = []
        async for chunk in client.generate_stream(
            [Message(role="user", content="hi")], model="stream-model"
        ):
            chunks.append(chunk)
        assert client.captured_model == "stream-model"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm_model_override.py -v`
Expected: FAIL — `model` parameter not found in signature

- [ ] **Step 3: Add model param to LLMClientBase (base.py)**

Two changes in `base.py`:

**Change 1:** Modify `generate()` signature (line 49) and its body (line 60). Full updated method head:
```python
async def generate(
    self,
    messages: list[Message],
    tools: list[Any] | None = None,
    model: str | None = None,
) -> LLMResponse:
```
And in the body, forward model to generate_stream (line 60):
```python
# Change this line:
async for chunk in self.generate_stream(messages, tools):
# To:
async for chunk in self.generate_stream(messages, tools, model=model):
```

**Change 2:** Modify `generate_stream()` abstract signature (line 99):
```python
@abstractmethod
async def generate_stream(
    self,
    messages: list[Message],
    tools: list[Any] | None = None,
    model: str | None = None,
) -> AsyncGenerator[LLMStreamChunk, None]:
```

- [ ] **Step 4: Add model param to LLMClient wrapper (llm_wrapper.py)**

Modify both `generate()` (line 115) and `generate_stream()` (line 131):
```python
async def generate(self, messages, tools=None, model: str | None = None) -> LLMResponse:
    return await self._client.generate(messages, tools, model=model)

async def generate_stream(self, messages, tools=None, model: str | None = None):
    async for chunk in self._client.generate_stream(messages, tools, model=model):
        yield chunk
```

- [ ] **Step 5: Add model param to AnthropicClient (anthropic_client.py)**

Modify `generate_stream()` (line 201):
```python
async def generate_stream(self, messages, tools=None, model: str | None = None):
```

In the params dict (line 221), replace `"model": self.model` with:
```python
use_model = model or self.model
params: dict[str, Any] = {
    "model": use_model,
    ...
}
```

- [ ] **Step 6: Add model param to OpenAIClient (openai_client.py)**

Modify `generate_stream()` (line 205):
```python
async def generate_stream(self, messages, tools=None, model: str | None = None):
```

In the params dict (line 225), replace `"model": self.model` with:
```python
use_model = model or self.model
params: dict[str, Any] = {
    "model": use_model,
    ...
}
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm_model_override.py -v`
Expected: PASS

Also run existing LLM tests to check no regressions:
Run: `uv run pytest tests/test_llm.py tests/test_llm_clients.py tests/test_llm_base_generate.py tests/test_anthropic_stream.py tests/test_openai_stream.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add mini_agent/llm/base.py mini_agent/llm/llm_wrapper.py mini_agent/llm/anthropic_client.py mini_agent/llm/openai_client.py tests/test_llm_model_override.py
git commit -m "feat: add per-call model override to LLM clients"
```

---

## Chunk 2: Tools + Agent Refactor + Session

### Task 4: State Tools

**Files:**
- Create: `mini_agent/tools/state_tools.py`
- Test: `tests/test_state_tools.py`

- [ ] **Step 1: Write tests for state tools**

```python
# tests/test_state_tools.py
"""Tests for SharedState tool wrappers."""
import pytest
from mini_agent.shared_state import SharedState
from mini_agent.tools.state_tools import StateReadTool, StateWriteTool, StateListTool


class TestStateReadTool:
    async def test_read_existing_key(self):
        state = SharedState()
        await state.set("key1", "value1", agent_id="test")
        tool = StateReadTool(state)
        result = await tool.execute(key="key1")
        assert result.success is True
        assert "value1" in result.content

    async def test_read_nonexistent_key(self):
        state = SharedState()
        tool = StateReadTool(state)
        result = await tool.execute(key="missing")
        assert result.success is True
        assert "not found" in result.content.lower() or "null" in result.content.lower()

    async def test_schema(self):
        tool = StateReadTool(SharedState())
        schema = tool.to_schema()
        assert schema["name"] == "state_read"
        assert "key" in schema["input_schema"]["properties"]


class TestStateWriteTool:
    async def test_write_value(self):
        state = SharedState()
        tool = StateWriteTool(state, agent_id="main")
        result = await tool.execute(key="k1", value="v1", schema_hint="string")
        assert result.success is True
        stored = await state.get("k1")
        assert stored == "v1"

    async def test_write_records_agent_id(self):
        state = SharedState()
        tool = StateWriteTool(state, agent_id="coder")
        await tool.execute(key="k1", value="v1")
        entry = state._store["k1"]
        assert entry.written_by == "coder"

    async def test_schema(self):
        tool = StateWriteTool(SharedState(), agent_id="main")
        schema = tool.to_schema()
        assert schema["name"] == "state_write"
        assert "key" in schema["input_schema"]["properties"]
        assert "value" in schema["input_schema"]["properties"]


class TestStateListTool:
    async def test_list_all(self):
        state = SharedState()
        await state.set("a", 1, agent_id="main")
        await state.set("b", 2, agent_id="main")
        tool = StateListTool(state)
        result = await tool.execute()
        assert result.success is True
        assert "a" in result.content
        assert "b" in result.content

    async def test_list_with_prefix(self):
        state = SharedState()
        await state.set("data.x", 1, agent_id="main")
        await state.set("config.y", 2, agent_id="main")
        tool = StateListTool(state)
        result = await tool.execute(prefix="data.")
        assert result.success is True
        assert "data.x" in result.content
        assert "config.y" not in result.content

    async def test_schema(self):
        tool = StateListTool(SharedState())
        schema = tool.to_schema()
        assert schema["name"] == "state_list"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_state_tools.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement state tools**

```python
# mini_agent/tools/state_tools.py
"""SharedState tool wrappers for LLM access."""
from __future__ import annotations

import json
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_state_tools.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mini_agent/tools/state_tools.py tests/test_state_tools.py
git commit -m "feat: add SharedState tool wrappers (read/write/list)"
```

---

### Task 5: DelegationTool

**Files:**
- Create: `mini_agent/tools/delegation_tool.py`
- Test: `tests/test_delegation_tool.py`

- [ ] **Step 1: Write tests for DelegationTool**

```python
# tests/test_delegation_tool.py
"""Tests for DelegationTool."""
import pytest
from mini_agent.agent_config import AgentConfig
from mini_agent.tools.delegation_tool import DelegationTool


def make_runner(result: str = "done", raise_error: Exception | None = None):
    """Create a mock SubAgentRunner."""
    async def runner(config: AgentConfig, task: str) -> str:
        if raise_error:
            raise raise_error
        return result
    return runner


class TestDelegationToolSchema:
    def test_name(self):
        tool = DelegationTool(
            sub_agents={"coder": AgentConfig(name="Coder", description="Writes code")},
            runner=make_runner(),
        )
        assert tool.name == "delegate_to_agent"

    def test_description_lists_agents(self):
        tool = DelegationTool(
            sub_agents={
                "coder": AgentConfig(name="Coder", description="Writes code"),
                "analyst": AgentConfig(name="Analyst", description="Analyzes data"),
            },
            runner=make_runner(),
        )
        desc = tool.description
        assert "coder" in desc
        assert "Writes code" in desc
        assert "analyst" in desc

    def test_parameters_enum(self):
        tool = DelegationTool(
            sub_agents={
                "a": AgentConfig(),
                "b": AgentConfig(),
            },
            runner=make_runner(),
        )
        params = tool.parameters
        enum = params["properties"]["agent_name"]["enum"]
        assert sorted(enum) == ["a", "b"]

    def test_schema_format(self):
        tool = DelegationTool(
            sub_agents={"x": AgentConfig()},
            runner=make_runner(),
        )
        schema = tool.to_schema()
        assert schema["name"] == "delegate_to_agent"
        assert "input_schema" in schema


class TestDelegationToolExecute:
    async def test_successful_delegation(self):
        tool = DelegationTool(
            sub_agents={"coder": AgentConfig(name="Coder", description="Writes code")},
            runner=make_runner(result="Task completed successfully."),
        )
        result = await tool.execute(agent_name="coder", task="Write hello.py")
        assert result.success is True
        assert "Task completed successfully." in result.content

    async def test_unknown_agent(self):
        tool = DelegationTool(
            sub_agents={"coder": AgentConfig()},
            runner=make_runner(),
        )
        result = await tool.execute(agent_name="unknown", task="Do something")
        assert result.success is False
        assert "Unknown agent" in result.error

    async def test_runner_exception(self):
        tool = DelegationTool(
            sub_agents={"coder": AgentConfig(name="Coder")},
            runner=make_runner(raise_error=RuntimeError("LLM failed")),
        )
        result = await tool.execute(agent_name="coder", task="Do something")
        assert result.success is False
        assert "failed" in result.error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_delegation_tool.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement DelegationTool**

```python
# mini_agent/tools/delegation_tool.py
"""DelegationTool: delegate tasks to specialized sub-agents."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Tool, ToolResult

if TYPE_CHECKING:
    from mini_agent.agent_config import AgentConfig, SubAgentRunner


class DelegationTool(Tool):
    """Delegates a task to a specialized sub-agent.

    Receives a SubAgentRunner callback to avoid circular dependency with Agent.
    The runner is created by Agent and injected at registration time.
    """

    def __init__(
        self,
        sub_agents: dict[str, AgentConfig],
        runner: SubAgentRunner,
    ):
        self._sub_agents = sub_agents
        self._runner = runner

    @property
    def name(self) -> str:
        return "delegate_to_agent"

    @property
    def description(self) -> str:
        agent_list = "\n".join(
            f"- {name}: {cfg.description}"
            for name, cfg in self._sub_agents.items()
        )
        return (
            "Delegate a task to a specialized sub-agent. "
            "The task must be self-contained — sub-agents cannot ask "
            "for clarification.\n\n"
            f"Available agents:\n{agent_list}"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "enum": list(self._sub_agents.keys()),
                    "description": "Which agent to delegate to",
                },
                "task": {
                    "type": "string",
                    "description": "Complete, self-contained task description",
                },
            },
            "required": ["agent_name", "task"],
        }

    async def execute(self, agent_name: str, task: str) -> ToolResult:  # type: ignore[override]
        config = self._sub_agents.get(agent_name)
        if not config:
            return ToolResult(
                success=False,
                content="",
                error=f"Unknown agent: {agent_name}. Available: {list(self._sub_agents.keys())}",
            )
        try:
            result = await self._runner(config, task)
            return ToolResult(success=True, content=result)
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Sub-agent '{agent_name}' failed: {e}",
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_delegation_tool.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mini_agent/tools/delegation_tool.py tests/test_delegation_tool.py
git commit -m "feat: add DelegationTool with factory callback pattern"
```

---

### Task 6: Agent Refactor

This is the largest task — refactoring `Agent.__init__` to use `AgentConfig` and adding sub-agent lifecycle methods.

**Files:**
- Modify: `mini_agent/agent.py`
- Modify: `tests/test_agent.py` (update Agent construction calls)
- Modify: `tests/test_agent_stream.py` (update Agent construction calls)

- [ ] **Step 1: Write tests for new Agent constructor and sub-agent features**

Create a new test file for the sub-agent specific features. Existing agent tests will be updated in a later step.

```python
# tests/test_agent_sub_agent.py
"""Tests for Agent sub-agent features (AgentConfig, delegation, state tools)."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from mini_agent.agent import Agent
from mini_agent.agent_config import AgentConfig
from mini_agent.shared_state import SharedState


def make_mock_llm():
    """Create a mock LLM client."""
    llm = MagicMock()
    llm.generate_stream = AsyncMock(return_value=AsyncMock())
    return llm


class TestAgentConfigConstructor:
    def test_basic_construction(self, tmp_path):
        config = AgentConfig(system_prompt="You are helpful.")
        agent = Agent(
            llm_client=make_mock_llm(),
            config=config,
            workspace_dir=tmp_path,
        )
        assert agent.config.agent_id == "main"
        assert agent.config.system_prompt == "You are helpful."
        assert "Current Workspace" in agent.system_prompt

    def test_tools_registered(self, tmp_path):
        from mini_agent.tools.base import Tool, ToolResult

        class FakeTool(Tool):
            @property
            def name(self): return "fake"
            @property
            def description(self): return "A fake tool"
            @property
            def parameters(self): return {"type": "object", "properties": {}}
            async def execute(self) -> ToolResult:
                return ToolResult(success=True, content="ok")

        config = AgentConfig(tools=[FakeTool()])
        agent = Agent(llm_client=make_mock_llm(), config=config, workspace_dir=tmp_path)
        assert "fake" in agent.tools

    def test_path_policy_in_system_prompt(self, tmp_path):
        config = AgentConfig(system_prompt="Hello")
        agent = Agent(llm_client=make_mock_llm(), config=config, workspace_dir=tmp_path)
        assert "Path Access Policy" in agent.system_prompt


class TestSubAgentRegistration:
    def test_register_sub_agent(self, tmp_path):
        config = AgentConfig()
        agent = Agent(llm_client=make_mock_llm(), config=config, workspace_dir=tmp_path)
        sub_config = AgentConfig(agent_id="coder", name="Coder", description="Writes code")
        agent.register_sub_agent("coder", sub_config)
        assert "coder" in agent.sub_agent_names
        assert "delegate_to_agent" in agent.tools

    def test_register_enables_delegation(self, tmp_path):
        config = AgentConfig(can_delegate=False)
        agent = Agent(llm_client=make_mock_llm(), config=config, workspace_dir=tmp_path)
        assert "delegate_to_agent" not in agent.tools
        agent.register_sub_agent("helper", AgentConfig(description="Helps"))
        assert "delegate_to_agent" in agent.tools
        assert agent.config.can_delegate is True

    def test_system_prompt_updated_after_registration(self, tmp_path):
        config = AgentConfig(system_prompt="Base prompt")
        agent = Agent(llm_client=make_mock_llm(), config=config, workspace_dir=tmp_path)
        assert "Available Sub-Agents" not in agent.system_prompt
        agent.register_sub_agent("coder", AgentConfig(description="Writes code"))
        assert "Available Sub-Agents" in agent.system_prompt
        assert "coder" in agent.system_prompt


class TestStateToolsRegistration:
    def test_state_tools_registered_with_readwrite(self, tmp_path):
        config = AgentConfig(state_access="readwrite")
        state = SharedState()
        agent = Agent(
            llm_client=make_mock_llm(), config=config,
            workspace_dir=tmp_path, shared_state=state,
        )
        assert "state_read" in agent.tools
        assert "state_write" in agent.tools
        assert "state_list" in agent.tools

    def test_state_tools_read_only(self, tmp_path):
        config = AgentConfig(state_access="read")
        state = SharedState()
        agent = Agent(
            llm_client=make_mock_llm(), config=config,
            workspace_dir=tmp_path, shared_state=state,
        )
        assert "state_read" in agent.tools
        assert "state_list" in agent.tools
        assert "state_write" not in agent.tools

    def test_no_state_tools_without_shared_state(self, tmp_path):
        config = AgentConfig(state_access="readwrite")
        agent = Agent(llm_client=make_mock_llm(), config=config, workspace_dir=tmp_path)
        assert "state_read" not in agent.tools

    def test_state_tools_write_only(self, tmp_path):
        config = AgentConfig(state_access="write")
        state = SharedState()
        agent = Agent(
            llm_client=make_mock_llm(), config=config,
            workspace_dir=tmp_path, shared_state=state,
        )
        assert "state_write" in agent.tools
        assert "state_read" not in agent.tools
        assert "state_list" not in agent.tools

    def test_no_state_tools_with_none_access(self, tmp_path):
        config = AgentConfig(state_access="none")
        state = SharedState()
        agent = Agent(
            llm_client=make_mock_llm(), config=config,
            workspace_dir=tmp_path, shared_state=state,
        )
        assert "state_read" not in agent.tools


class TestCanDelegateWithoutSubAgents:
    def test_can_delegate_true_but_no_sub_agents(self, tmp_path):
        """can_delegate=True without registered sub-agents should NOT add delegation tool."""
        config = AgentConfig(can_delegate=True)
        agent = Agent(llm_client=make_mock_llm(), config=config, workspace_dir=tmp_path)
        assert "delegate_to_agent" not in agent.tools


class TestSubAgentNamesAccessor:
    def test_empty_by_default(self, tmp_path):
        agent = Agent(llm_client=make_mock_llm(), config=AgentConfig(), workspace_dir=tmp_path)
        assert agent.sub_agent_names == []

    def test_returns_registered_names(self, tmp_path):
        agent = Agent(llm_client=make_mock_llm(), config=AgentConfig(), workspace_dir=tmp_path)
        agent.register_sub_agent("a", AgentConfig())
        agent.register_sub_agent("b", AgentConfig())
        assert sorted(agent.sub_agent_names) == ["a", "b"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_agent_sub_agent.py -v`
Expected: FAIL — Agent constructor signature doesn't match

- [ ] **Step 3: Refactor Agent constructor**

Modify `mini_agent/agent.py`. Key changes:

1. **Add imports** at top of file (keeping all existing imports like `ContextConfig`, `ContextManager`, `LLMClient`, etc.):
   ```python
   from .agent_config import AgentConfig, SubAgentRunner
   from .shared_state import SharedState
   ```

2. **Replace `__init__` signature** (lines 31-44) with:
   ```python
   def __init__(
       self,
       llm_client: LLMClient,
       config: AgentConfig,
       workspace_dir: str | Path = "./workspace",
       shared_state: SharedState | None = None,
       logger: AgentLogger | None = None,
       path_guard=None,
   ):
   ```

3. **Replace init body** (lines 45-89) with:
   ```python
       self.config = config
       self.llm = llm_client
       self.workspace_dir = Path(workspace_dir)
       self.shared_state = shared_state
       self.path_guard = path_guard
       self.cancel_event: Optional[asyncio.Event] = None

       # Tools: from config
       self.tools = {tool.name: tool for tool in config.tools}

       # Sub-agent registry
       self._sub_agents: dict[str, AgentConfig] = {}
       self._delegation_depth: int = 0
       self._step_count: int = 0

       # Workspace setup
       self.workspace_dir.mkdir(parents=True, exist_ok=True)

       # System prompt
       self.system_prompt = self._build_system_prompt(config.system_prompt)

       # Logger
       from .config import LogLevel
       self.logger = logger or AgentLogger(
           file_level=LogLevel.STANDARD,
           console_level=LogLevel.MINIMAL,
       )

       # Auto-register delegation + state tools
       self._register_auto_tools()

       # ContextManager
       ctx = config.context_config or ContextConfig()
       self.context_manager = ContextManager(ctx, self.llm, self.tools)

       # Backward compat
       self.messages: list[Message] = []
   ```

4. **Add new methods** after `__init__`:

   ```python
   @property
   def sub_agent_names(self) -> list[str]:
       return list(self._sub_agents.keys())

   def register_sub_agent(self, name: str, config: AgentConfig):
       """Register a sub-agent configuration."""
       self._sub_agents[name] = config
       self.config.can_delegate = True
       self._register_auto_tools()
       self._rebuild_system_block()

   def _register_auto_tools(self):
       """Add/update delegation and state tools based on current config."""
       from .tools.delegation_tool import DelegationTool
       from .tools.state_tools import StateReadTool, StateWriteTool, StateListTool

       if self.config.can_delegate and self._sub_agents:
           runner = self._make_sub_agent_runner()
           self.tools["delegate_to_agent"] = DelegationTool(self._sub_agents, runner)

       if self.shared_state and self.config.state_access != "none":
           access = self.config.state_access
           agent_id = self.config.agent_id
           if "read" in access:
               self.tools["state_read"] = StateReadTool(self.shared_state)
               self.tools["state_list"] = StateListTool(self.shared_state)
           if "write" in access:
               self.tools["state_write"] = StateWriteTool(self.shared_state, agent_id)

   def _make_sub_agent_runner(self) -> SubAgentRunner:
       """Create a closure that runs sub-agents sharing this agent's infrastructure."""
       async def runner(config: AgentConfig, task: str) -> str:
           import copy

           if self._delegation_depth >= self.config.max_delegation_depth:
               return f"[Delegation blocked] Max depth ({self.config.max_delegation_depth}) reached."

           cfg = copy.deepcopy(config)
           # ContextConfig is already imported at module level
           cfg.context_config = cfg.context_config or ContextConfig.from_mode("claude_code")

           if self._delegation_depth >= self.config.max_delegation_depth - 1:
               cfg.can_delegate = False

           sub = Agent(
               llm_client=self.llm,
               config=cfg,
               workspace_dir=self.workspace_dir,
               shared_state=self.shared_state,
               logger=self.logger,
               path_guard=self.path_guard,
           )
           sub._delegation_depth = self._delegation_depth + 1

           sub.add_user_message(task)
           result = await sub.run()

           return (
               f"[Sub-agent: {cfg.name}]\n"
               f"Steps used: {sub._step_count}\n"
               f"Result:\n{result}"
           )
       return runner

   def _build_system_prompt(self, base_prompt: str) -> str:
       """Construct system prompt with workspace info, path policy, sub-agent list."""
       parts = [base_prompt]

       if "Current Workspace" not in base_prompt:
           parts.append(
               f"\n\n## Current Workspace\n"
               f"You are currently working in: `{self.workspace_dir.absolute()}`\n"
               f"All relative paths will be resolved relative to this directory."
           )

       parts.append(
           "\n\n## Path Access Policy\n"
           "You operate under file access restrictions:\n"
           "- Full read/write access within the workspace directory\n"
           "- Access outside the workspace is restricted\n"
           "- Your own source code is not accessible\n\n"
           "If a file operation is denied, inform the user about the restriction.\n"
           "Do NOT attempt to work around restrictions via bash, alternative paths, "
           "or encoded commands."
       )

       if self._sub_agents:
           agent_list = "\n".join(
               f"- **{name}**: {cfg.description}"
               for name, cfg in self._sub_agents.items()
           )
           parts.append(f"\n\n## Available Sub-Agents\n{agent_list}")

       if self.shared_state and self.shared_state.snapshot():
           state_lines = "\n".join(
               f"- {k}: {v}" for k, v in self.shared_state.snapshot().items()
           )
           parts.append(f"\n\n## Shared Data\n{state_lines}")

       return "".join(parts)

   def _rebuild_system_block(self):
       """Rebuild system prompt and update system block in BlockStore."""
       self.system_prompt = self._build_system_prompt(self.config.system_prompt)
       system_block = self.context_manager.store.get("system")
       if system_block:
           from .context.token_counter import count_tokens
           system_block.working_content = self.system_prompt
           system_block.token_count = count_tokens(self.system_prompt)
   ```

5. **Update run_stream()** step loop:
   - Change `for step in range(self.max_steps)` to `for step in range(self.config.max_steps_per_turn)`
   - Add total step check at loop top:
     ```python
     if self._step_count >= self.config.max_steps_total:
         yield StreamEvent(type=StreamEventType.ERROR,
                           content=f"Total step limit ({self.config.max_steps_total}) reached.")
         return
     ```
   - Increment `self._step_count` after each tool execution
   - Pass `model=self.config.model` to `self.llm.generate_stream()`
   - Update max-steps-exceeded error at end of loop:
     ```python
     yield StreamEvent(type=StreamEventType.ERROR,
                       content=f"Task couldn't be completed after {self.config.max_steps_per_turn} steps in this turn.")
     ```

- [ ] **Step 4: Run new sub-agent tests**

Run: `uv run pytest tests/test_agent_sub_agent.py -v`
Expected: All PASS

- [ ] **Step 5: Update existing agent tests**

Files that construct `Agent(...)` and need migration to `AgentConfig`:
- `tests/test_agent.py` — main agent tests
- `tests/test_agent_stream.py` — streaming tests
- `tests/test_acp.py` — ACP server tests (if any direct Agent construction)
- `tests/test_integration.py` — integration tests (if any)
- `mini_agent/acp/__init__.py` — ACP server may construct Agent

Pattern:
```python
# Before:
agent = Agent(llm_client=mock_llm, system_prompt="...", tools=[...], max_steps=5)

# After:
from mini_agent.agent_config import AgentConfig
config = AgentConfig(system_prompt="...", tools=[...], max_steps_per_turn=5)
agent = Agent(llm_client=mock_llm, config=config, workspace_dir=tmp_path)
```

Also update any references to `agent.max_steps` → `agent.config.max_steps_per_turn`.

Search for all references: `grep -rn "Agent(" tests/ mini_agent/acp/ mini_agent/cli.py` and update each site.

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS. If other files (examples, ACP) break, fix those too.

- [ ] **Step 7: Commit**

```bash
git add mini_agent/agent.py tests/test_agent_sub_agent.py tests/test_agent.py tests/test_agent_stream.py
git commit -m "feat: refactor Agent to use AgentConfig, add sub-agent lifecycle"
```

---

### Task 7: Session API

**Files:**
- Create: `mini_agent/session.py`
- Test: `tests/test_session.py`

- [ ] **Step 1: Write tests for Session**

```python
# tests/test_session.py
"""Tests for Session API."""
import pytest
from unittest.mock import MagicMock, AsyncMock
from mini_agent.agent_config import AgentConfig
from mini_agent.session import Session
from mini_agent.shared_state import SharedState


def make_mock_llm():
    llm = MagicMock()
    llm.generate_stream = AsyncMock()
    return llm


class TestSessionCreate:
    def test_creates_session_with_agent(self, tmp_path):
        session = Session.create(
            llm_client=make_mock_llm(),
            system_prompt="Hello",
            workspace_dir=tmp_path,
        )
        assert session.agent is not None
        assert session.shared_state is not None
        assert isinstance(session.shared_state, SharedState)
        assert session.agent.config.agent_id == "main"

    def test_creates_session_with_sub_agents(self, tmp_path):
        session = Session.create(
            llm_client=make_mock_llm(),
            sub_agents={
                "coder": AgentConfig(name="Coder", description="Writes code"),
            },
            workspace_dir=tmp_path,
        )
        assert session.agent.config.can_delegate is True
        assert "coder" in session.agent.sub_agent_names
        assert "delegate_to_agent" in session.agent.tools

    def test_no_delegation_without_sub_agents(self, tmp_path):
        session = Session.create(llm_client=make_mock_llm(), workspace_dir=tmp_path)
        assert session.agent.config.can_delegate is False
        assert "delegate_to_agent" not in session.agent.tools

    def test_state_tools_always_registered(self, tmp_path):
        """Session.create() always creates SharedState, so state tools should be registered."""
        session = Session.create(llm_client=make_mock_llm(), workspace_dir=tmp_path)
        assert session.shared_state is not None
        assert "state_read" in session.agent.tools
        assert "state_write" in session.agent.tools

    def test_custom_context_config(self, tmp_path):
        from mini_agent.context.config import ContextConfig
        ctx = ContextConfig.from_mode("claude_code")
        session = Session.create(llm_client=make_mock_llm(), context_config=ctx, workspace_dir=tmp_path)
        assert session.agent.context_manager is not None

    def test_workspace_dir_passed(self, tmp_path):
        session = Session.create(
            llm_client=make_mock_llm(),
            workspace_dir=tmp_path,
        )
        assert session.agent.workspace_dir == tmp_path


class TestSessionCreateOrchestrator:
    def test_creates_orchestrator(self, tmp_path):
        session = Session.create_orchestrator(
            llm_client=make_mock_llm(),
            workers={
                "coder": AgentConfig(name="Coder", description="Writes code"),
            },
            workspace_dir=tmp_path,
        )
        assert session.agent.config.agent_id == "orchestrator"
        assert session.agent.config.can_delegate is True
        assert "coder" in session.agent.sub_agent_names

    def test_orchestrator_default_prompt(self, tmp_path):
        session = Session.create_orchestrator(
            llm_client=make_mock_llm(),
            workers={"w": AgentConfig()},
            workspace_dir=tmp_path,
        )
        assert "orchestrator" in session.agent.config.system_prompt.lower()

    def test_orchestrator_custom_prompt(self, tmp_path):
        session = Session.create_orchestrator(
            llm_client=make_mock_llm(),
            workers={"w": AgentConfig()},
            orchestrator_prompt="Custom orchestrator.",
            workspace_dir=tmp_path,
        )
        assert session.agent.config.system_prompt == "Custom orchestrator."


class TestSessionDelegation:
    def test_add_user_message(self, tmp_path):
        session = Session.create(llm_client=make_mock_llm(), workspace_dir=tmp_path)
        session.add_user_message("Hello")
        status = session.get_status()
        assert status["turn"] == 1

    def test_get_status(self, tmp_path):
        session = Session.create(
            llm_client=make_mock_llm(),
            sub_agents={"coder": AgentConfig(description="Codes")},
            workspace_dir=tmp_path,
        )
        status = session.get_status()
        assert status["agent_id"] == "main"
        assert status["turn"] == 0
        assert status["shared_state_keys"] == []
        assert "coder" in status["sub_agents"]
        assert "context" in status
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement Session**

```python
# mini_agent/session.py
"""Session API: high-level entry point for agent creation and execution."""
from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .agent import Agent
from .agent_config import AgentConfig
from .shared_state import SharedState

if TYPE_CHECKING:
    from .context.config import ContextConfig
    from .llm import LLMClient
    from .logger import AgentLogger
    from .schema import StreamEvent
    from .tools.base import Tool
    from .tools.path_guard import PathGuard


class Session:
    """Wraps Agent + SharedState with convenient factory methods."""

    def __init__(self, agent: Agent, shared_state: SharedState):
        self.agent = agent
        self.shared_state = shared_state

    # -- Delegated to Agent --

    def add_user_message(self, content: str):
        self.agent.add_user_message(content)

    async def run(self, cancel_event=None) -> str:
        return await self.agent.run(cancel_event)

    async def run_stream(self, cancel_event=None) -> AsyncGenerator[StreamEvent, None]:
        async for event in self.agent.run_stream(cancel_event):
            yield event

    # -- Status --

    def get_status(self) -> dict:
        return {
            "agent_id": self.agent.config.agent_id,
            "turn": self.agent.context_manager.current_turn,
            "context": self.agent.context_manager.get_status(),
            "shared_state_keys": list(self.shared_state.snapshot().keys()),
            "sub_agents": list(self.agent.sub_agent_names),
        }

    # -- Factory Methods --

    @classmethod
    def create(
        cls,
        llm_client: LLMClient,
        system_prompt: str = "You are a helpful assistant.",
        tools: list[Tool] | None = None,
        sub_agents: dict[str, AgentConfig] | None = None,
        context_config: ContextConfig | None = None,
        workspace_dir: str | Path = "./workspace",
        logger: AgentLogger | None = None,
        path_guard: PathGuard | None = None,
    ) -> Session:
        """Primary entry point. Single agent with optional delegation."""
        shared_state = SharedState()

        main_config = AgentConfig(
            agent_id="main",
            system_prompt=system_prompt,
            tools=tools or [],
            context_config=context_config,
            can_delegate=bool(sub_agents),
        )

        agent = Agent(
            llm_client=llm_client,
            config=main_config,
            workspace_dir=workspace_dir,
            shared_state=shared_state,
            logger=logger,
            path_guard=path_guard,
        )

        for name, sub_cfg in (sub_agents or {}).items():
            agent.register_sub_agent(name, sub_cfg)

        return cls(agent=agent, shared_state=shared_state)

    @classmethod
    def create_orchestrator(
        cls,
        llm_client: LLMClient,
        workers: dict[str, AgentConfig],
        orchestrator_prompt: str | None = None,
        context_config: ContextConfig | None = None,
        workspace_dir: str | Path = "./workspace",
        logger: AgentLogger | None = None,
        path_guard: PathGuard | None = None,
    ) -> Session:
        """Advanced entry point. Agent identity = planner/coordinator."""
        from .context.config import ContextConfig as CtxCfg

        shared_state = SharedState()

        default_prompt = (
            "You are an orchestrator agent. Your job is to:\n"
            "1. Analyze the user's task and break it into subtasks\n"
            "2. Delegate subtasks to specialized worker agents\n"
            "3. Synthesize worker results into a coherent final answer\n\n"
            "Planning: break into 2-5 subtasks. Each must be self-contained.\n"
            "Efficiency: don't delegate trivial tasks - do them yourself.\n"
            "Synthesis: after workers complete, produce unified answer."
        )

        orch_config = AgentConfig(
            agent_id="orchestrator",
            name="Orchestrator",
            system_prompt=orchestrator_prompt or default_prompt,
            context_config=context_config or CtxCfg.from_mode("full_layering"),
            can_delegate=True,
            max_delegation_depth=1,
            state_access="readwrite",
        )

        agent = Agent(
            llm_client=llm_client,
            config=orch_config,
            workspace_dir=workspace_dir,
            shared_state=shared_state,
            logger=logger,
            path_guard=path_guard,
        )

        for name, w_cfg in workers.items():
            w_cfg.context_config = w_cfg.context_config or CtxCfg.from_mode("claude_code")
            agent.register_sub_agent(name, w_cfg)

        return cls(agent=agent, shared_state=shared_state)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_session.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mini_agent/session.py tests/test_session.py
git commit -m "feat: add Session API with create() and create_orchestrator()"
```

---

## Chunk 3: Config + CLI + Integration

### Task 8: Config Changes

**Files:**
- Modify: `mini_agent/config.py` (rename AgentConfig → AgentSettings, add SubAgentEntry, update from_yaml)
- Modify: `tests/test_context_config.py` or other tests that reference config.agent (if any)

- [ ] **Step 1: Write tests for config changes**

```python
# tests/test_config_sub_agent.py
"""Tests for sub-agent config parsing."""
import pytest
import yaml
import tempfile
from pathlib import Path
from mini_agent.config import Config, AgentSettings, SubAgentEntry


class TestAgentSettingsRename:
    def test_agent_settings_exists(self):
        """AgentSettings should exist (renamed from AgentConfig)."""
        settings = AgentSettings()
        assert settings.max_steps_per_turn == 30
        assert settings.max_steps_total == 50
        assert settings.workspace_dir == "./workspace"

    def test_backward_compat_max_steps(self):
        """If only max_steps is in YAML (flat top level), it maps to max_steps_per_turn."""
        yaml_content = """
api_key: test-key
api_base: https://api.test.com
model: test-model
max_steps: 25
workspace_dir: ./workspace
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = Config.from_yaml(f.name)
        assert config.agent.max_steps_per_turn == 25
        assert config.agent.max_steps_total == 50  # default

    def test_both_step_limits(self):
        """Both max_steps_per_turn and max_steps_total can be set."""
        yaml_content = """
api_key: test-key
api_base: https://api.test.com
model: test-model
max_steps_per_turn: 20
max_steps_total: 100
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = Config.from_yaml(f.name)
        assert config.agent.max_steps_per_turn == 20
        assert config.agent.max_steps_total == 100


class TestSubAgentEntry:
    def test_defaults(self):
        entry = SubAgentEntry()
        assert entry.description == ""
        assert entry.system_prompt == ""
        assert entry.system_prompt_path == ""
        assert entry.model is None
        assert entry.tools == []

    def test_custom_values(self):
        entry = SubAgentEntry(
            description="Writes code",
            system_prompt="You are a coder.",
            tools=["bash", "read", "write"],
        )
        assert entry.description == "Writes code"
        assert entry.tools == ["bash", "read", "write"]


class TestSubAgentConfigParsing:
    def test_parse_sub_agents_from_yaml(self):
        yaml_content = """
api_key: test-key
api_base: https://api.test.com
model: test-model
sub_agents:
  coder:
    description: "Writes code"
    system_prompt: "You are a coder."
    tools: ["bash", "read", "write"]
  researcher:
    description: "Researches"
    system_prompt: "You research."
    tools: ["bash", "read"]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = Config.from_yaml(f.name)
        assert "coder" in config.sub_agents
        assert "researcher" in config.sub_agents
        assert config.sub_agents["coder"].description == "Writes code"
        assert config.sub_agents["coder"].tools == ["bash", "read", "write"]

    def test_empty_sub_agents(self):
        yaml_content = """
api_key: test-key
api_base: https://api.test.com
model: test-model
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = Config.from_yaml(f.name)
        assert config.sub_agents == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_sub_agent.py -v`
Expected: FAIL — `AgentSettings` doesn't exist, `sub_agents` not in Config

- [ ] **Step 3: Update config.py**

> **Note:** The spec's Section 7 YAML example shows `agent:` as a nested section. The actual `from_yaml()` reads agent fields from the flat top level (e.g., `data.get("max_steps")`). Preserve this flat format for backward compatibility.

Complete changes to `config.py`:

**3a.** Rename `AgentConfig` class (line 41) and update fields:
```python
class AgentSettings(BaseModel):
    """Agent configuration (renamed from AgentConfig to avoid collision with runtime AgentConfig)."""

    max_steps_per_turn: int = 30
    max_steps_total: int = 50
    workspace_dir: str = "./workspace"
    system_prompt_path: str = "system_prompt.md"
```

**3b.** Add `SubAgentEntry` class after `AgentSettings`:
```python
class SubAgentEntry(BaseModel):
    """YAML config for a sub-agent definition."""
    description: str = ""
    system_prompt: str = ""
    system_prompt_path: str = ""
    model: str | None = None
    tools: list[str] = []
```

**3c.** Update `Config` class:
```python
class Config(BaseModel):
    """Main configuration class"""

    llm: LLMConfig
    agent: AgentSettings
    tools: ToolsConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    context: ContextConfig = ContextConfig()
    sub_agents: dict[str, SubAgentEntry] = {}
```

**3d.** Replace agent config parsing in `from_yaml()` (lines 164-169):
```python
        # Parse Agent configuration (flat top-level keys, NOT nested "agent:" section)
        max_steps_per_turn = data.get("max_steps_per_turn", 30)
        # Backward compat: old "max_steps" maps to max_steps_per_turn
        if "max_steps" in data and "max_steps_per_turn" not in data:
            max_steps_per_turn = data.get("max_steps", 30)
        agent_config = AgentSettings(
            max_steps_per_turn=max_steps_per_turn,
            max_steps_total=data.get("max_steps_total", 50),
            workspace_dir=data.get("workspace_dir", "./workspace"),
            system_prompt_path=data.get("system_prompt_path", "system_prompt.md"),
        )
```

**3e.** Add sub_agents parsing before the return statement (around line 217):
```python
        # Parse sub-agent definitions
        sub_agents_data = data.get("sub_agents", {})
        sub_agents = {
            name: SubAgentEntry(**entry)
            for name, entry in sub_agents_data.items()
        }
```

**3f.** Update the return statement to include sub_agents:
```python
        return cls(
            llm=llm_config,
            agent=agent_config,
            tools=tools_config,
            logging=logging_config,
            context=context_config,
            sub_agents=sub_agents,
        )
```

- [ ] **Step 4: Update all references to old AgentConfig name and max_steps**

All files that need changes (verify with `grep -rn "AgentConfig\|config\.agent\.max_steps" mini_agent/ tests/ examples/`):

**`mini_agent/cli.py`** (line ~723):
```python
# Before:
max_steps=config.agent.max_steps,
# After:
max_steps=config.agent.max_steps_per_turn,
```
(Note: This line will be replaced entirely by Session.create() in Task 9, but update it now for test compatibility.)

**`mini_agent/acp/__init__.py`** (line ~102):
```python
# Before:
max_steps=self._config.agent.max_steps,
# After:
max_steps=self._config.agent.max_steps_per_turn,
```

**`tests/test_acp.py`** (line ~8, ~64):
```python
# Before:
from mini_agent.config import AgentConfig
# After:
from mini_agent.config import AgentSettings

# Before:
agent=AgentConfig(max_steps=3, ...)
# After:
agent=AgentSettings(max_steps_per_turn=3, ...)
```

**`tests/test_integration.py`** (line ~93):
```python
# Before:
max_steps=config.agent.max_steps,
# After:
max_steps=config.agent.max_steps_per_turn,
```

**`examples/04_full_agent.py`** (line ~107):
```python
# Before:
max_steps=config.agent.max_steps,
# After:
max_steps=config.agent.max_steps_per_turn,
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_config_sub_agent.py -v`
Expected: All PASS

Run: `uv run pytest tests/ -v`
Expected: All PASS (check for broken references to old AgentConfig)

- [ ] **Step 6: Commit**

```bash
git add mini_agent/config.py mini_agent/cli.py mini_agent/acp/__init__.py tests/test_config_sub_agent.py tests/test_acp.py tests/test_integration.py examples/04_full_agent.py
git commit -m "feat: rename AgentConfig to AgentSettings, add SubAgentEntry for sub-agent YAML config"
```

---

### Task 9: CLI Adaptation

**Files:**
- Modify: `mini_agent/cli.py` (use Session.create, build sub-agent configs)

- [ ] **Step 1: Update add_workspace_tools to return tool name mapping**

Currently `add_workspace_tools()` adds tools to a list and returns PathGuard. Update it to also build a `dict[str, Tool]` mapping for sub-agent tool name resolution.

In `cli.py`, after all tools are added, build the mapping:

```python
# After all tools are initialized — map tool.name as-is (no lowering)
tool_registry: dict[str, Tool] = {}
for tool in tools:
    tool_registry[tool.name] = tool
```

- [ ] **Step 2: Build sub-agent AgentConfigs from YAML**

After tool initialization, before Agent creation:

```python
from mini_agent.agent_config import AgentConfig as RuntimeAgentConfig

sub_agent_configs = {}
for name, entry in config.sub_agents.items():
    sub_tools = [tool_registry[t] for t in entry.tools if t in tool_registry]
    sub_prompt = entry.system_prompt
    if not sub_prompt and entry.system_prompt_path:
        # find_config_file searches mini_agent/config/, ~/.mini-agent/config/, etc.
        # Sub-agent prompt files should be placed in those config directories.
        prompt_path = Config.find_config_file(entry.system_prompt_path)
        if prompt_path and prompt_path.exists():
            sub_prompt = prompt_path.read_text(encoding="utf-8")
    sub_agent_configs[name] = RuntimeAgentConfig(
        agent_id=name,
        name=name,
        description=entry.description,
        model=entry.model,
        system_prompt=sub_prompt or f"You are {name}.",
        tools=sub_tools,
        max_steps_total=config.agent.max_steps_total,
    )
```

- [ ] **Step 3: Create AgentLogger before Session.create()**

The CLI currently creates AgentLogger inline during Agent construction. Since Session.create() accepts an optional logger, create it before:

```python
from mini_agent.logger import AgentLogger
from mini_agent.config import LogLevel

agent_logger = AgentLogger(
    file_level=config.logging.file_level,
    console_level=config.logging.console_level,
    max_files=config.logging.max_files,
)
```

- [ ] **Step 4: Replace Agent construction with Session.create()**

Replace the current `Agent(...)` call (around line 719) with:

```python
from mini_agent.session import Session

session = Session.create(
    llm_client=llm_client,
    system_prompt=system_prompt,
    tools=tools,
    sub_agents=sub_agent_configs or None,
    context_config=config.context,
    workspace_dir=workspace_dir,
    logger=agent_logger,
    path_guard=path_guard,
)
agent = session.agent  # For backward compat with rest of CLI code
```

- [ ] **Step 5: Wire PathGuard logger**

After Session.create(), wire the logger as before:

```python
if path_guard:
    path_guard.logger = agent.logger
```

- [ ] **Step 6: Update Agent references in CLI**

The interactive loop and non-interactive mode reference `agent` directly. Since we set `agent = session.agent`, most code should work unchanged. Verify that:
- `agent.add_user_message()` still works
- `agent.run_stream()` still works
- `agent.context_manager` references still work (for /stats, /clear commands)
- `/clear` command resets properly

- [ ] **Step 7: Verify sub-agent config building**

Add a quick manual check — after building `sub_agent_configs`, print or log the count:
```python
if sub_agent_configs:
    logger.info(f"Configured {len(sub_agent_configs)} sub-agents: {list(sub_agent_configs.keys())}")
```

- [ ] **Step 8: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All PASS

- [ ] **Step 9: Manual smoke test**

Run: `uv run python -m mini_agent.cli --help`
Expected: CLI starts without errors

If a config.yaml with `sub_agents` is available, test:
Run: `uv run python -m mini_agent.cli`
Expected: Interactive mode starts, agent has delegation capability if sub_agents configured

- [ ] **Step 10: Commit**

```bash
git add mini_agent/cli.py
git commit -m "feat: CLI uses Session.create() with YAML sub-agent support"
```

---

### Task 10: Integration Tests

**Files:**
- Create: `tests/test_sub_agent_integration.py`

- [ ] **Step 1: Write integration tests**

```python
# tests/test_sub_agent_integration.py
"""Integration tests for sub-agent delegation and SharedState."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from mini_agent.agent import Agent
from mini_agent.agent_config import AgentConfig
from mini_agent.session import Session
from mini_agent.shared_state import SharedState
from mini_agent.schema import LLMStreamChunk, LLMStreamChunkType


def make_mock_llm_with_responses(responses: list[list[LLMStreamChunk]]):
    """Create a mock LLM that returns predefined response sequences.

    Each call to generate_stream returns the next response in the list.
    """
    llm = MagicMock()
    call_count = [0]

    async def mock_generate_stream(messages, tools, model=None):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        for chunk in responses[idx]:
            yield chunk

    llm.generate_stream = mock_generate_stream
    return llm


def text_response(text: str) -> list[LLMStreamChunk]:
    """Create a simple text-only response (no tool calls)."""
    return [
        LLMStreamChunk(type=LLMStreamChunkType.TEXT_DELTA, content=text),
        LLMStreamChunk(type=LLMStreamChunkType.DONE, finish_reason="stop"),
    ]


def tool_call_response(tool_name: str, tool_id: str, arguments: dict) -> list[LLMStreamChunk]:
    """Create a response with a single tool call."""
    import json
    return [
        LLMStreamChunk(
            type=LLMStreamChunkType.TOOL_CALL_START,
            tool_call_id=tool_id, tool_name=tool_name,
        ),
        LLMStreamChunk(
            type=LLMStreamChunkType.TOOL_CALL_DELTA,
            tool_call_id=tool_id, tool_arguments=json.dumps(arguments),
        ),
        LLMStreamChunk(
            type=LLMStreamChunkType.TOOL_CALL_END,
            tool_call_id=tool_id,
        ),
        LLMStreamChunk(type=LLMStreamChunkType.DONE, finish_reason="tool_use"),
    ]


class TestDelegationFlow:
    async def test_parent_delegates_to_sub_agent(self, tmp_path):
        """Parent calls delegate_to_agent, sub-agent runs and returns result."""
        # Sub-agent LLM: just returns text immediately
        sub_response = text_response("Sub-agent completed the task.")

        # Parent LLM call 1: calls delegate_to_agent tool
        # Parent LLM call 2: synthesizes final answer after delegation
        parent_call_1 = tool_call_response(
            "delegate_to_agent", "tc_1",
            {"agent_name": "coder", "task": "Write hello.py"},
        )
        parent_call_2 = text_response("Done! The coder wrote hello.py.")

        # The mock LLM serves responses in order: parent call 1, sub-agent call, parent call 2.
        # Both parent and sub-agent share the same LLM instance (as in real usage).
        llm = make_mock_llm_with_responses([parent_call_1, sub_response, parent_call_2])

        session = Session.create(
            llm_client=llm,
            system_prompt="You are helpful.",
            sub_agents={
                "coder": AgentConfig(
                    name="Coder",
                    description="Writes code",
                    system_prompt="You write code.",
                ),
            },
            workspace_dir=tmp_path,
        )

        session.add_user_message("Write hello.py")
        result = await session.run()
        assert "The coder wrote hello.py" in result

    async def test_depth_limit_enforced(self, tmp_path):
        """Sub-agent at max depth cannot re-delegate."""
        config = AgentConfig(
            system_prompt="Test",
            can_delegate=True,
            max_delegation_depth=1,
        )
        agent = Agent(llm_client=MagicMock(), config=config, workspace_dir=tmp_path)
        agent._delegation_depth = 1  # Already at max depth

        # The runner should block delegation
        runner = agent._make_sub_agent_runner()
        result = await runner(AgentConfig(name="sub"), "do something")
        assert "blocked" in result.lower() or "depth" in result.lower()


class TestSharedStateIntegration:
    async def test_state_tools_registered_via_session(self, tmp_path):
        session = Session.create(llm_client=MagicMock(), workspace_dir=tmp_path)
        # Session creates SharedState, so state tools should be registered
        assert "state_read" in session.agent.tools
        assert "state_write" in session.agent.tools
        assert "state_list" in session.agent.tools

    async def test_state_write_read_across_session(self, tmp_path):
        session = Session.create(llm_client=MagicMock(), workspace_dir=tmp_path)
        # Write via tool
        write_tool = session.agent.tools["state_write"]
        await write_tool.execute(key="test_key", value="test_value", schema_hint="string")

        # Read via tool
        read_tool = session.agent.tools["state_read"]
        result = await read_tool.execute(key="test_key")
        assert result.success is True
        assert "test_value" in result.content

    async def test_shared_state_visible_in_status(self, tmp_path):
        session = Session.create(llm_client=MagicMock(), workspace_dir=tmp_path)
        write_tool = session.agent.tools["state_write"]
        await write_tool.execute(key="data", value="123")
        status = session.get_status()
        assert "data" in status["shared_state_keys"]


class TestModelOverrideIntegration:
    async def test_sub_agent_uses_own_model(self, tmp_path):
        """Sub-agent with model override passes it to LLM generate_stream."""
        captured_models = []

        async def mock_generate_stream(messages, tools, model=None):
            captured_models.append(model)
            for chunk in text_response("done"):
                yield chunk

        llm = MagicMock()
        llm.generate_stream = mock_generate_stream

        session = Session.create(
            llm_client=llm,
            system_prompt="Parent",
            sub_agents={
                "coder": AgentConfig(
                    name="Coder",
                    description="Writes code",
                    system_prompt="Code.",
                    model="special-model-v2",
                ),
            },
            workspace_dir=tmp_path,
        )

        # Directly run the sub-agent via the runner to verify model is forwarded
        runner = session.agent._make_sub_agent_runner()
        sub_config = session.agent._sub_agents["coder"]
        await runner(sub_config, "Write hello.py")

        # The sub-agent's generate_stream call should have model="special-model-v2"
        assert "special-model-v2" in captured_models
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_sub_agent_integration.py -v`
Expected: All PASS (depends on Tasks 1-9 being complete)

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_sub_agent_integration.py
git commit -m "test: add sub-agent integration tests for delegation and SharedState"
```
