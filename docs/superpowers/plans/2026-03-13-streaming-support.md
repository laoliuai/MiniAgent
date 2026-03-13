# Streaming Support Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add streaming support across LLM clients, Agent core loop, and consumer layers (CLI, ACP), with `run_stream()` returning `AsyncGenerator[StreamEvent]` as the single source of truth.

**Architecture:** Bottom-up: new stream types in schema → LLM clients implement `generate_stream()` → Agent adds `run_stream()` (wrapping `run()`) → CLI/ACP consume events. Two layers of stream types: `LLMStreamChunk` (protocol-level) and `StreamEvent` (business-level).

**Tech Stack:** Python 3.10+, asyncio, anthropic SDK streaming, openai SDK streaming, pydantic, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-03-13-streaming-support-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `mini_agent/schema/schema.py` | Modify | Add `LLMStreamChunk`, `LLMStreamChunkType`, `StreamEvent`, `StreamEventType`; move `ToolResult` here |
| `mini_agent/schema/__init__.py` | Modify | Export new types + `ToolResult` |
| `mini_agent/tools/base.py` | Modify | Import `ToolResult` from `schema` instead of defining it |
| `mini_agent/llm/base.py` | Modify | `generate()` concrete (collects stream); add abstract `generate_stream()` |
| `mini_agent/llm/anthropic_client.py` | Modify | Implement `generate_stream()`, add `_make_stream_request()`; remove `generate()` and `_parse_response()` |
| `mini_agent/llm/openai_client.py` | Modify | Implement `generate_stream()`, add `_make_stream_request()`; remove `generate()` and `_parse_response()` |
| `mini_agent/llm/llm_wrapper.py` | Modify | Add `generate_stream()` passthrough |
| `mini_agent/agent.py` | Modify | Add `run_stream()`; `run()` wraps it; remove all `print()` calls |
| `mini_agent/cli.py` | Modify | Consume `run_stream()` events for terminal output |
| `mini_agent/acp/__init__.py` | Modify | `_run_turn()` consumes `run_stream()` events |
| `mini_agent/__init__.py` | Modify | Export new streaming types |
| `tests/test_stream_types.py` | Create | Unit tests for LLMStreamChunk, StreamEvent |
| `tests/test_llm_base_generate.py` | Create | Unit tests for base class `generate()` collecting stream |
| `tests/test_anthropic_stream.py` | Create | Unit tests for AnthropicClient.generate_stream() with mocks |
| `tests/test_openai_stream.py` | Create | Unit tests for OpenAIClient.generate_stream() with mocks |
| `tests/test_agent_stream.py` | Create | Unit tests for Agent.run_stream() and run() wrapper with mocks |

---

## Chunk 1: Schema Types & ToolResult Migration

### Task 1: Add streaming types to schema

**Files:**
- Modify: `mini_agent/schema/schema.py`
- Create: `tests/test_stream_types.py`

- [ ] **Step 1: Write failing tests for new schema types**

Create `tests/test_stream_types.py`:

```python
"""Tests for streaming schema types."""

from mini_agent.schema.schema import (
    LLMStreamChunk,
    LLMStreamChunkType,
    StreamEvent,
    StreamEventType,
    TokenUsage,
    ToolResult,
)


def test_llm_stream_chunk_text_delta():
    chunk = LLMStreamChunk(type=LLMStreamChunkType.TEXT_DELTA, content="hello")
    assert chunk.type == LLMStreamChunkType.TEXT_DELTA
    assert chunk.content == "hello"
    assert chunk.tool_call_id is None


def test_llm_stream_chunk_tool_call_start():
    chunk = LLMStreamChunk(
        type=LLMStreamChunkType.TOOL_CALL_START,
        tool_call_id="call_123",
        tool_name="read_file",
    )
    assert chunk.tool_call_id == "call_123"
    assert chunk.tool_name == "read_file"


def test_llm_stream_chunk_tool_call_delta():
    chunk = LLMStreamChunk(
        type=LLMStreamChunkType.TOOL_CALL_DELTA,
        tool_call_id="call_123",
        tool_arguments='{"path":',
    )
    assert chunk.tool_arguments == '{"path":'


def test_llm_stream_chunk_usage():
    usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chunk = LLMStreamChunk(type=LLMStreamChunkType.USAGE, usage=usage)
    assert chunk.usage.total_tokens == 15


def test_llm_stream_chunk_done():
    chunk = LLMStreamChunk(type=LLMStreamChunkType.DONE, finish_reason="stop")
    assert chunk.finish_reason == "stop"


def test_stream_event_text_delta():
    event = StreamEvent(type=StreamEventType.TEXT_DELTA, content="hi", step=1)
    assert event.type == StreamEventType.TEXT_DELTA
    assert event.step == 1


def test_stream_event_tool_call_start():
    event = StreamEvent(
        type=StreamEventType.TOOL_CALL_START,
        tool_name="bash",
        tool_call_id="call_1",
        tool_arguments={"command": "ls"},
        step=2,
    )
    assert event.tool_arguments == {"command": "ls"}


def test_stream_event_tool_call_result():
    result = ToolResult(success=True, content="file.txt")
    event = StreamEvent(
        type=StreamEventType.TOOL_CALL_RESULT,
        tool_name="bash",
        tool_call_id="call_1",
        tool_result=result,
        step=2,
    )
    assert event.tool_result.success is True


def test_stream_event_done():
    event = StreamEvent(type=StreamEventType.DONE, content="Final answer.")
    assert event.content == "Final answer."


def test_stream_event_error():
    event = StreamEvent(type=StreamEventType.ERROR, content="API failed")
    assert event.type == StreamEventType.ERROR


def test_stream_event_cancelled():
    event = StreamEvent(type=StreamEventType.CANCELLED)
    assert event.content is None


def test_tool_result_in_schema():
    """ToolResult should be importable from schema."""
    result = ToolResult(success=False, content="", error="not found")
    assert result.error == "not found"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stream_types.py -v`
Expected: FAIL with `ImportError` — types don't exist yet

- [ ] **Step 3: Add streaming types and move ToolResult to schema**

Add to the end of `mini_agent/schema/schema.py` (after existing `LLMResponse` class):

```python
class ToolResult(BaseModel):
    """Tool execution result."""
    success: bool
    content: str = ""
    error: str | None = None


class LLMStreamChunkType(str, Enum):
    """Types of chunks in an LLM stream."""
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    USAGE = "usage"
    DONE = "done"


class LLMStreamChunk(BaseModel):
    """A single chunk from an LLM streaming response (protocol-level)."""
    type: LLMStreamChunkType
    content: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments: str | None = None
    usage: TokenUsage | None = None
    finish_reason: str | None = None


class StreamEventType(str, Enum):
    """Types of events in an Agent stream."""
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    STEP_START = "step_start"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_RESULT = "tool_call_result"
    STEP_COMPLETE = "step_complete"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class StreamEvent(BaseModel):
    """A single event from Agent.run_stream() (business-level)."""
    type: StreamEventType
    content: str | None = None
    step: int | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_arguments: dict | None = None
    tool_result: ToolResult | None = None
    usage: TokenUsage | None = None
    finish_reason: str | None = None
```

- [ ] **Step 4: Update schema `__init__.py` to export new types**

Replace `mini_agent/schema/__init__.py` with:

```python
"""Schema definitions for Mini-Agent."""

from .schema import (
    FunctionCall,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    LLMStreamChunkType,
    Message,
    StreamEvent,
    StreamEventType,
    TokenUsage,
    ToolCall,
    ToolResult,
)

__all__ = [
    "FunctionCall",
    "LLMProvider",
    "LLMResponse",
    "LLMStreamChunk",
    "LLMStreamChunkType",
    "Message",
    "StreamEvent",
    "StreamEventType",
    "TokenUsage",
    "ToolCall",
    "ToolResult",
]
```

- [ ] **Step 5: Update `tools/base.py` to import ToolResult from schema**

Replace the `ToolResult` definition in `mini_agent/tools/base.py` with an import:

```python
"""Base tool classes."""

from typing import Any

from ..schema import ToolResult


class Tool:
    # ... rest stays the same
```

Remove the `from pydantic import BaseModel` import and the `ToolResult` class definition (lines 5-13).

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_stream_types.py -v`
Expected: All 12 tests PASS

- [ ] **Step 7: Run existing tests to verify nothing broke**

Run: `pytest tests/test_tool_schema.py tests/test_tools.py tests/test_note_tool.py tests/test_bash_tool.py -v`
Expected: All existing tests still PASS (ToolResult import change is transparent)

- [ ] **Step 8: Commit**

```bash
git add mini_agent/schema/schema.py mini_agent/schema/__init__.py mini_agent/tools/base.py tests/test_stream_types.py
git commit -m "feat: add streaming schema types and migrate ToolResult to schema"
```

---

## Chunk 2: LLM Client Base Class Refactor

### Task 2: Refactor LLMClientBase — generate() concrete, generate_stream() abstract

**Files:**
- Modify: `mini_agent/llm/base.py`
- Create: `tests/test_llm_base_generate.py`

- [ ] **Step 1: Write failing tests for base class generate()**

Create `tests/test_llm_base_generate.py`:

```python
"""Tests for LLMClientBase.generate() collecting stream chunks."""

import json
from collections.abc import AsyncGenerator

import pytest

from mini_agent.llm.base import LLMClientBase
from mini_agent.schema import (
    FunctionCall,
    LLMStreamChunk,
    LLMStreamChunkType,
    Message,
    TokenUsage,
    ToolCall,
)


class FakeStreamClient(LLMClientBase):
    """Test client that yields pre-defined chunks."""

    def __init__(self, chunks: list[LLMStreamChunk]):
        super().__init__(api_key="fake", api_base="http://fake", model="fake")
        self._chunks = chunks

    async def generate_stream(self, messages, tools=None):
        for chunk in self._chunks:
            yield chunk

    def _prepare_request(self, messages, tools=None):
        return {}

    def _convert_messages(self, messages):
        return None, []


@pytest.mark.asyncio
async def test_generate_collects_text():
    chunks = [
        LLMStreamChunk(type=LLMStreamChunkType.TEXT_DELTA, content="Hello "),
        LLMStreamChunk(type=LLMStreamChunkType.TEXT_DELTA, content="world"),
        LLMStreamChunk(type=LLMStreamChunkType.DONE, finish_reason="stop"),
    ]
    client = FakeStreamClient(chunks)
    response = await client.generate([Message(role="user", content="hi")])
    assert response.content == "Hello world"
    assert response.finish_reason == "stop"
    assert response.tool_calls is None


@pytest.mark.asyncio
async def test_generate_collects_thinking():
    chunks = [
        LLMStreamChunk(type=LLMStreamChunkType.THINKING_DELTA, content="Let me think"),
        LLMStreamChunk(type=LLMStreamChunkType.THINKING_DELTA, content="..."),
        LLMStreamChunk(type=LLMStreamChunkType.TEXT_DELTA, content="Answer"),
        LLMStreamChunk(type=LLMStreamChunkType.DONE, finish_reason="stop"),
    ]
    client = FakeStreamClient(chunks)
    response = await client.generate([Message(role="user", content="hi")])
    assert response.thinking == "Let me think..."
    assert response.content == "Answer"


@pytest.mark.asyncio
async def test_generate_collects_single_tool_call():
    chunks = [
        LLMStreamChunk(type=LLMStreamChunkType.TOOL_CALL_START,
                        tool_call_id="call_1", tool_name="read_file"),
        LLMStreamChunk(type=LLMStreamChunkType.TOOL_CALL_DELTA,
                        tool_call_id="call_1", tool_arguments='{"path":'),
        LLMStreamChunk(type=LLMStreamChunkType.TOOL_CALL_DELTA,
                        tool_call_id="call_1", tool_arguments='"test.txt"}'),
        LLMStreamChunk(type=LLMStreamChunkType.TOOL_CALL_END,
                        tool_call_id="call_1", tool_name="read_file"),
        LLMStreamChunk(type=LLMStreamChunkType.DONE, finish_reason="tool_use"),
    ]
    client = FakeStreamClient(chunks)
    response = await client.generate([Message(role="user", content="read test.txt")])
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].function.name == "read_file"
    assert response.tool_calls[0].function.arguments == {"path": "test.txt"}


@pytest.mark.asyncio
async def test_generate_collects_multiple_tool_calls():
    """Two tool calls interleaved (OpenAI-style)."""
    chunks = [
        LLMStreamChunk(type=LLMStreamChunkType.TOOL_CALL_START,
                        tool_call_id="call_1", tool_name="read_file"),
        LLMStreamChunk(type=LLMStreamChunkType.TOOL_CALL_START,
                        tool_call_id="call_2", tool_name="bash"),
        LLMStreamChunk(type=LLMStreamChunkType.TOOL_CALL_DELTA,
                        tool_call_id="call_1", tool_arguments='{"path":"a.txt"}'),
        LLMStreamChunk(type=LLMStreamChunkType.TOOL_CALL_DELTA,
                        tool_call_id="call_2", tool_arguments='{"command":"ls"}'),
        LLMStreamChunk(type=LLMStreamChunkType.TOOL_CALL_END,
                        tool_call_id="call_1", tool_name="read_file"),
        LLMStreamChunk(type=LLMStreamChunkType.TOOL_CALL_END,
                        tool_call_id="call_2", tool_name="bash"),
        LLMStreamChunk(type=LLMStreamChunkType.DONE, finish_reason="tool_use"),
    ]
    client = FakeStreamClient(chunks)
    response = await client.generate([Message(role="user", content="do stuff")])
    assert len(response.tool_calls) == 2
    assert response.tool_calls[0].function.name == "read_file"
    assert response.tool_calls[0].function.arguments == {"path": "a.txt"}
    assert response.tool_calls[1].function.name == "bash"
    assert response.tool_calls[1].function.arguments == {"command": "ls"}


@pytest.mark.asyncio
async def test_generate_collects_usage():
    usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    chunks = [
        LLMStreamChunk(type=LLMStreamChunkType.TEXT_DELTA, content="ok"),
        LLMStreamChunk(type=LLMStreamChunkType.USAGE, usage=usage),
        LLMStreamChunk(type=LLMStreamChunkType.DONE, finish_reason="stop"),
    ]
    client = FakeStreamClient(chunks)
    response = await client.generate([Message(role="user", content="hi")])
    assert response.usage.total_tokens == 150


@pytest.mark.asyncio
async def test_generate_empty_thinking_is_none():
    chunks = [
        LLMStreamChunk(type=LLMStreamChunkType.TEXT_DELTA, content="answer"),
        LLMStreamChunk(type=LLMStreamChunkType.DONE, finish_reason="stop"),
    ]
    client = FakeStreamClient(chunks)
    response = await client.generate([Message(role="user", content="hi")])
    assert response.thinking is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_base_generate.py -v`
Expected: FAIL — `generate_stream` not defined, `generate()` still abstract

- [ ] **Step 3: Rewrite `mini_agent/llm/base.py`**

Replace the entire file:

```python
"""Base class for LLM clients."""

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from ..retry import RetryConfig
from ..schema import (
    FunctionCall,
    LLMResponse,
    LLMStreamChunk,
    LLMStreamChunkType,
    Message,
    ToolCall,
)


class LLMClientBase(ABC):
    """Abstract base class for LLM clients.

    Subclasses implement generate_stream(). The non-streaming generate()
    is provided by the base class and collects stream chunks into an LLMResponse.
    """

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        retry_config: RetryConfig | None = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.retry_config = retry_config or RetryConfig()
        self.retry_callback = None

    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        """Collect generate_stream() into a complete LLMResponse."""
        text, thinking = "", ""
        tool_calls = []
        pending_tools: dict[str, dict] = {}
        usage, finish_reason = None, "stop"

        async for chunk in self.generate_stream(messages, tools):
            match chunk.type:
                case LLMStreamChunkType.TEXT_DELTA:
                    text += chunk.content
                case LLMStreamChunkType.THINKING_DELTA:
                    thinking += chunk.content
                case LLMStreamChunkType.TOOL_CALL_START:
                    pending_tools[chunk.tool_call_id] = {
                        "id": chunk.tool_call_id,
                        "name": chunk.tool_name,
                        "arguments_json": "",
                    }
                case LLMStreamChunkType.TOOL_CALL_DELTA:
                    pending_tools[chunk.tool_call_id]["arguments_json"] += chunk.tool_arguments
                case LLMStreamChunkType.TOOL_CALL_END:
                    info = pending_tools[chunk.tool_call_id]
                    tool_calls.append(
                        ToolCall(
                            id=info["id"],
                            type="function",
                            function=FunctionCall(
                                name=info["name"],
                                arguments=json.loads(info["arguments_json"]),
                            ),
                        )
                    )
                case LLMStreamChunkType.USAGE:
                    usage = chunk.usage
                case LLMStreamChunkType.DONE:
                    finish_reason = chunk.finish_reason or "stop"

        return LLMResponse(
            content=text,
            thinking=thinking if thinking else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason,
            usage=usage,
        )

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Subclasses must implement this to yield LLMStreamChunks."""
        ...  # pragma: no cover
        # yield is needed to make this a generator; the ... above is for the abstractmethod
        yield  # type: ignore  # pragma: no cover

    @abstractmethod
    def _prepare_request(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_base_generate.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mini_agent/llm/base.py tests/test_llm_base_generate.py
git commit -m "refactor: make LLMClientBase.generate() concrete, add abstract generate_stream()"
```

---

### Task 3: Implement AnthropicClient.generate_stream()

**Files:**
- Modify: `mini_agent/llm/anthropic_client.py`
- Create: `tests/test_anthropic_stream.py`

- [ ] **Step 1: Write failing test for Anthropic streaming**

Create `tests/test_anthropic_stream.py`:

```python
"""Tests for AnthropicClient.generate_stream() with mocked SDK."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent.llm.anthropic_client import AnthropicClient
from mini_agent.schema import LLMStreamChunkType, Message


def make_event(event_type, **kwargs):
    """Create a mock Anthropic stream event."""
    event = MagicMock()
    event.type = event_type
    for k, v in kwargs.items():
        # Support nested attrs like event.delta.type
        parts = k.split(".")
        obj = event
        for part in parts[:-1]:
            if not hasattr(obj, part) or not isinstance(getattr(obj, part), MagicMock):
                setattr(obj, part, MagicMock())
            obj = getattr(obj, part)
        setattr(obj, parts[-1], v)
    return event


class MockStream:
    """Mock async context manager for Anthropic stream."""
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


@pytest.mark.asyncio
async def test_anthropic_stream_text_only():
    client = AnthropicClient(api_key="fake", api_base="http://fake", model="test")

    events = [
        make_event("content_block_start", index=0, **{"content_block.type": "text"}),
        make_event("content_block_delta", index=0,
                   **{"delta.type": "text_delta", "delta.text": "Hello"}),
        make_event("content_block_delta", index=0,
                   **{"delta.type": "text_delta", "delta.text": " world"}),
        make_event("content_block_stop", index=0),
        make_event("message_delta", **{"delta.stop_reason": "end_turn",
                   "usage.output_tokens": 10}),
        make_event("message_stop"),
    ]
    mock_stream = MockStream(events)

    with patch.object(client, '_make_stream_request', new_callable=AsyncMock, return_value=mock_stream):
        chunks = []
        async for chunk in client.generate_stream(
            [Message(role="user", content="hi")]
        ):
            chunks.append(chunk)

    text_chunks = [c for c in chunks if c.type == LLMStreamChunkType.TEXT_DELTA]
    assert len(text_chunks) == 2
    assert text_chunks[0].content == "Hello"
    assert text_chunks[1].content == " world"
    assert any(c.type == LLMStreamChunkType.DONE for c in chunks)


@pytest.mark.asyncio
async def test_anthropic_stream_tool_call():
    client = AnthropicClient(api_key="fake", api_base="http://fake", model="test")

    events = [
        make_event("content_block_start", index=0,
                   **{"content_block.type": "tool_use", "content_block.id": "call_1",
                      "content_block.name": "read_file"}),
        make_event("content_block_delta", index=0,
                   **{"delta.type": "input_json_delta", "delta.partial_json": '{"path":'}),
        make_event("content_block_delta", index=0,
                   **{"delta.type": "input_json_delta", "delta.partial_json": '"test.txt"}'}),
        make_event("content_block_stop", index=0),
        make_event("message_delta", **{"delta.stop_reason": "tool_use",
                   "usage.output_tokens": 20}),
        make_event("message_stop"),
    ]
    mock_stream = MockStream(events)

    with patch.object(client, '_make_stream_request', new_callable=AsyncMock, return_value=mock_stream):
        chunks = []
        async for chunk in client.generate_stream(
            [Message(role="user", content="read file")]
        ):
            chunks.append(chunk)

    starts = [c for c in chunks if c.type == LLMStreamChunkType.TOOL_CALL_START]
    deltas = [c for c in chunks if c.type == LLMStreamChunkType.TOOL_CALL_DELTA]
    ends = [c for c in chunks if c.type == LLMStreamChunkType.TOOL_CALL_END]
    assert len(starts) == 1
    assert starts[0].tool_name == "read_file"
    assert len(deltas) == 2
    assert all(d.tool_call_id == "call_1" for d in deltas)
    assert len(ends) == 1
    assert ends[0].tool_call_id == "call_1"


@pytest.mark.asyncio
async def test_anthropic_stream_thinking():
    client = AnthropicClient(api_key="fake", api_base="http://fake", model="test")

    events = [
        make_event("content_block_start", index=0, **{"content_block.type": "thinking"}),
        make_event("content_block_delta", index=0,
                   **{"delta.type": "thinking_delta", "delta.thinking": "Let me think"}),
        make_event("content_block_delta", index=0,
                   **{"delta.type": "thinking_delta", "delta.thinking": "..."}),
        make_event("content_block_stop", index=0),
        make_event("content_block_start", index=1, **{"content_block.type": "text"}),
        make_event("content_block_delta", index=1,
                   **{"delta.type": "text_delta", "delta.text": "Answer"}),
        make_event("content_block_stop", index=1),
        make_event("message_stop"),
    ]
    mock_stream = MockStream(events)

    with patch.object(client, '_make_stream_request', new_callable=AsyncMock, return_value=mock_stream):
        chunks = []
        async for chunk in client.generate_stream(
            [Message(role="user", content="think")]
        ):
            chunks.append(chunk)

    thinking = [c for c in chunks if c.type == LLMStreamChunkType.THINKING_DELTA]
    assert len(thinking) == 2
    assert thinking[0].content == "Let me think"


@pytest.mark.asyncio
async def test_anthropic_stream_retry_on_connection_failure():
    """Retry wraps connection establishment only."""
    from mini_agent.retry import RetryConfig
    client = AnthropicClient(
        api_key="fake", api_base="http://fake", model="test",
        retry_config=RetryConfig(enabled=True, max_retries=2, initial_delay=0.01),
    )

    call_count = [0]
    async def failing_then_success(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 2:
            raise ConnectionError("connection refused")
        return MockStream([
            make_event("content_block_start", index=0, **{"content_block.type": "text"}),
            make_event("content_block_delta", index=0,
                       **{"delta.type": "text_delta", "delta.text": "ok"}),
            make_event("content_block_stop", index=0),
            make_event("message_stop"),
        ])

    with patch.object(client, '_make_stream_request', side_effect=failing_then_success):
        chunks = []
        async for chunk in client.generate_stream(
            [Message(role="user", content="hi")]
        ):
            chunks.append(chunk)

    assert call_count[0] == 2
    text_chunks = [c for c in chunks if c.type == LLMStreamChunkType.TEXT_DELTA]
    assert len(text_chunks) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_anthropic_stream.py -v`
Expected: FAIL — `generate_stream()` and `_make_stream_request()` not implemented

- [ ] **Step 3: Implement AnthropicClient streaming**

Modify `mini_agent/llm/anthropic_client.py`:

1. Add `_make_stream_request()` method — creates stream via `self.client.messages.stream(**params)` and returns the stream context manager.
2. Add `_make_stream_request_with_retry()` — wraps `_make_stream_request` with retry logic (connection phase only).
3. Add `generate_stream()` — calls `_make_stream_request_with_retry()`, iterates events, yields `LLMStreamChunk`. Tracks `content_blocks: dict[int, dict]` for block-type lookups on `content_block_stop`.
4. Remove `generate()` override (inherited from base class now), `_make_api_request()`, and `_parse_response()`.

See spec section "AnthropicClient.generate_stream()" for complete pseudocode.

Key implementation details:
- `_make_stream_request()` is a thin async wrapper — it calls `self.client.messages.stream(**params)` and returns the async context manager directly. No `await` on the stream itself; the caller uses `async with` to enter it. Retry wraps this call (connection establishment), not the iteration.
- Track `content_blocks[idx]` on `content_block_start` to resolve block type/id on `content_block_stop`
- On `message_delta`, extract `finish_reason` from `event.delta.stop_reason`. For USAGE, construct `TokenUsage` from `event.usage.output_tokens` (the Anthropic SDK puts the final output token count here). The `prompt_tokens` come from the initial `message_start` event's `event.message.usage.input_tokens` — track it at the top of the generator and combine on `message_delta`.
- Import `LLMStreamChunk`, `LLMStreamChunkType`, `TokenUsage` from `..schema`

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_anthropic_stream.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add mini_agent/llm/anthropic_client.py tests/test_anthropic_stream.py
git commit -m "feat: implement AnthropicClient.generate_stream()"
```

---

### Task 4: Implement OpenAIClient.generate_stream()

**Files:**
- Modify: `mini_agent/llm/openai_client.py`
- Create: `tests/test_openai_stream.py`

- [ ] **Step 1: Write failing test for OpenAI streaming**

Create `tests/test_openai_stream.py`:

```python
"""Tests for OpenAIClient.generate_stream() with mocked SDK."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent.llm.openai_client import OpenAIClient
from mini_agent.schema import LLMStreamChunkType, Message


def make_openai_chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    """Create a mock OpenAI stream chunk."""
    chunk = MagicMock()
    choice = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    # reasoning_content support
    delta.reasoning_content = None
    if not hasattr(delta, "reasoning_content"):
        delta.reasoning_content = None
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    chunk.usage = usage
    return chunk


def make_tool_call_delta(index, tc_id=None, name=None, arguments=None):
    tc = MagicMock()
    tc.index = index
    tc.id = tc_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


class MockAsyncIterator:
    def __init__(self, items):
        self._items = list(items)
    def __aiter__(self):
        return self
    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


@pytest.mark.asyncio
async def test_openai_stream_text_only():
    client = OpenAIClient(api_key="fake", api_base="http://fake", model="test")

    stream_chunks = [
        make_openai_chunk(content="Hello"),
        make_openai_chunk(content=" world"),
        make_openai_chunk(finish_reason="stop"),
    ]

    with patch.object(client, '_make_stream_request', new_callable=AsyncMock,
                      return_value=MockAsyncIterator(stream_chunks)):
        chunks = []
        async for chunk in client.generate_stream(
            [Message(role="user", content="hi")]
        ):
            chunks.append(chunk)

    text_chunks = [c for c in chunks if c.type == LLMStreamChunkType.TEXT_DELTA]
    assert len(text_chunks) == 2
    assert text_chunks[0].content == "Hello"
    assert any(c.type == LLMStreamChunkType.DONE for c in chunks)


@pytest.mark.asyncio
async def test_openai_stream_tool_call():
    client = OpenAIClient(api_key="fake", api_base="http://fake", model="test")

    stream_chunks = [
        make_openai_chunk(tool_calls=[make_tool_call_delta(0, "call_1", "bash", "")]),
        make_openai_chunk(tool_calls=[make_tool_call_delta(0, None, None, '{"command":')]),
        make_openai_chunk(tool_calls=[make_tool_call_delta(0, None, None, '"ls"}')]),
        make_openai_chunk(finish_reason="tool_calls"),
    ]

    with patch.object(client, '_make_stream_request', new_callable=AsyncMock,
                      return_value=MockAsyncIterator(stream_chunks)):
        chunks = []
        async for chunk in client.generate_stream(
            [Message(role="user", content="list files")]
        ):
            chunks.append(chunk)

    starts = [c for c in chunks if c.type == LLMStreamChunkType.TOOL_CALL_START]
    deltas = [c for c in chunks if c.type == LLMStreamChunkType.TOOL_CALL_DELTA]
    ends = [c for c in chunks if c.type == LLMStreamChunkType.TOOL_CALL_END]
    assert len(starts) == 1
    assert starts[0].tool_name == "bash"
    assert len(deltas) == 2
    assert all(d.tool_call_id == "call_1" for d in deltas)
    assert len(ends) == 1


@pytest.mark.asyncio
async def test_openai_stream_multiple_tool_calls():
    """Two tool calls in one response."""
    client = OpenAIClient(api_key="fake", api_base="http://fake", model="test")

    stream_chunks = [
        make_openai_chunk(tool_calls=[make_tool_call_delta(0, "c1", "read_file", "")]),
        make_openai_chunk(tool_calls=[make_tool_call_delta(1, "c2", "bash", "")]),
        make_openai_chunk(tool_calls=[make_tool_call_delta(0, None, None, '{"path":"a.txt"}')]),
        make_openai_chunk(tool_calls=[make_tool_call_delta(1, None, None, '{"command":"ls"}')]),
        make_openai_chunk(finish_reason="tool_calls"),
    ]

    with patch.object(client, '_make_stream_request', new_callable=AsyncMock,
                      return_value=MockAsyncIterator(stream_chunks)):
        chunks = []
        async for chunk in client.generate_stream(
            [Message(role="user", content="do both")]
        ):
            chunks.append(chunk)

    starts = [c for c in chunks if c.type == LLMStreamChunkType.TOOL_CALL_START]
    ends = [c for c in chunks if c.type == LLMStreamChunkType.TOOL_CALL_END]
    assert len(starts) == 2
    assert len(ends) == 2


@pytest.mark.asyncio
async def test_openai_stream_thinking():
    """OpenAI-compatible thinking via delta.reasoning_content."""
    client = OpenAIClient(api_key="fake", api_base="http://fake", model="test")

    stream_chunks = [
        make_openai_chunk(content=None),  # reasoning_content set below
        make_openai_chunk(content=None),
        make_openai_chunk(content="Answer"),
        make_openai_chunk(finish_reason="stop"),
    ]
    # Set reasoning_content on first two chunks
    stream_chunks[0].choices[0].delta.reasoning_content = "Let me think"
    stream_chunks[1].choices[0].delta.reasoning_content = "..."

    with patch.object(client, '_make_stream_request', new_callable=AsyncMock,
                      return_value=MockAsyncIterator(stream_chunks)):
        chunks = []
        async for chunk in client.generate_stream(
            [Message(role="user", content="think")]
        ):
            chunks.append(chunk)

    thinking = [c for c in chunks if c.type == LLMStreamChunkType.THINKING_DELTA]
    assert len(thinking) == 2
    assert thinking[0].content == "Let me think"
    text = [c for c in chunks if c.type == LLMStreamChunkType.TEXT_DELTA]
    assert len(text) == 1
    assert text[0].content == "Answer"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_openai_stream.py -v`
Expected: FAIL

- [ ] **Step 3: Implement OpenAIClient streaming**

Modify `mini_agent/llm/openai_client.py`:

1. Add `_make_stream_request()` — calls `self.client.chat.completions.create(**params, stream=True)` and returns the async iterator.
2. Add `_make_stream_request_with_retry()` — wraps with retry.
3. Add `generate_stream()` — tracks `current_tool_calls: dict[int, dict]` indexed by position. All `TOOL_CALL_DELTA` chunks include `tool_call_id`. On `finish_reason`, emit `TOOL_CALL_END` for each tracked tool call, then `USAGE` and `DONE`.
4. Remove `generate()` override, `_make_api_request()`, and `_parse_response()`.

See spec section "OpenAIClient.generate_stream()" for complete pseudocode.

Key implementation details:
- OpenAI streams `delta.tool_calls` as a list of partial dicts with `index`, optional `id`/`function.name` (on first appearance), and `function.arguments` (incremental)
- Track by index, always emit `tool_call_id` on `TOOL_CALL_DELTA` via `current_tool_calls[idx]["id"]`
- Thinking content via `delta.reasoning_content` (MiniMax/OpenAI-compatible)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_openai_stream.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add mini_agent/llm/openai_client.py tests/test_openai_stream.py
git commit -m "feat: implement OpenAIClient.generate_stream()"
```

---

### Task 5: Add generate_stream() to LLMClient wrapper

**Files:**
- Modify: `mini_agent/llm/llm_wrapper.py`

- [ ] **Step 1: Add generate_stream() passthrough to LLMClient**

Add to `mini_agent/llm/llm_wrapper.py` after the existing `generate()` method:

```python
async def generate_stream(
    self,
    messages: list[Message],
    tools: list | None = None,
) -> AsyncGenerator[LLMStreamChunk, None]:
    """Stream response from LLM."""
    async for chunk in self._client.generate_stream(messages, tools):
        yield chunk
```

Add imports at the top:
```python
from collections.abc import AsyncGenerator
from ..schema import LLMStreamChunk
```

- [ ] **Step 2: Run all LLM tests to verify nothing broke**

Run: `pytest tests/test_llm_base_generate.py tests/test_anthropic_stream.py tests/test_openai_stream.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add mini_agent/llm/llm_wrapper.py
git commit -m "feat: add generate_stream() passthrough to LLMClient wrapper"
```

---

## Chunk 3: Agent Layer Streaming

### Task 6: Implement Agent.run_stream() and rewrite run()

**Files:**
- Modify: `mini_agent/agent.py`
- Create: `tests/test_agent_stream.py`

- [ ] **Step 1: Write failing tests for Agent.run_stream()**

Create `tests/test_agent_stream.py`:

```python
"""Tests for Agent.run_stream() and run() wrapper with mocked LLM."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent.agent import Agent
from mini_agent.schema import (
    LLMStreamChunk,
    LLMStreamChunkType,
    StreamEvent,
    StreamEventType,
    ToolResult,
)


def make_mock_llm(chunks_sequence):
    """Create a mock LLM client that yields chunks from a sequence.

    chunks_sequence is a list of lists — each inner list is one generate_stream() call.
    """
    call_count = [0]

    async def mock_generate_stream(messages, tools=None):
        idx = min(call_count[0], len(chunks_sequence) - 1)
        call_count[0] += 1
        for chunk in chunks_sequence[idx]:
            yield chunk

    llm = MagicMock()
    llm.generate_stream = mock_generate_stream
    # generate() also needed for _create_summary — mock it simply
    llm.generate = AsyncMock(return_value=MagicMock(content="summary"))
    return llm


def make_text_chunks(text, finish_reason="stop"):
    """Helper: create chunks for a simple text response."""
    chunks = []
    for char in text:
        chunks.append(LLMStreamChunk(type=LLMStreamChunkType.TEXT_DELTA, content=char))
    chunks.append(LLMStreamChunk(type=LLMStreamChunkType.DONE, finish_reason=finish_reason))
    return chunks


@pytest.mark.asyncio
async def test_run_stream_simple_text():
    llm = make_mock_llm([make_text_chunks("Hi")])
    agent = Agent(llm_client=llm, system_prompt="test", tools=[], max_steps=5)
    agent.add_user_message("hello")

    events = []
    async for event in agent.run_stream():
        events.append(event)

    types = [e.type for e in events]
    assert StreamEventType.STEP_START in types
    assert StreamEventType.TEXT_DELTA in types
    assert StreamEventType.DONE in types
    assert StreamEventType.STEP_COMPLETE in types

    text_events = [e for e in events if e.type == StreamEventType.TEXT_DELTA]
    assert "".join(e.content for e in text_events) == "Hi"

    done_event = next(e for e in events if e.type == StreamEventType.DONE)
    assert done_event.content == "Hi"


@pytest.mark.asyncio
async def test_run_returns_final_text():
    llm = make_mock_llm([make_text_chunks("Hello world")])
    agent = Agent(llm_client=llm, system_prompt="test", tools=[], max_steps=5)
    agent.add_user_message("hi")

    result = await agent.run()
    assert result == "Hello world"


@pytest.mark.asyncio
async def test_run_stream_with_tool_call():
    """Agent calls a tool, then returns final text."""
    # First LLM call: tool call
    tool_chunks = [
        LLMStreamChunk(type=LLMStreamChunkType.TOOL_CALL_START,
                        tool_call_id="c1", tool_name="test_tool"),
        LLMStreamChunk(type=LLMStreamChunkType.TOOL_CALL_DELTA,
                        tool_call_id="c1", tool_arguments='{"arg":"val"}'),
        LLMStreamChunk(type=LLMStreamChunkType.TOOL_CALL_END,
                        tool_call_id="c1", tool_name="test_tool"),
        LLMStreamChunk(type=LLMStreamChunkType.DONE, finish_reason="tool_use"),
    ]
    # Second LLM call: final text after tool result
    final_chunks = make_text_chunks("Done!")

    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.execute = AsyncMock(return_value=ToolResult(success=True, content="tool output"))

    llm = make_mock_llm([tool_chunks, final_chunks])
    agent = Agent(llm_client=llm, system_prompt="test", tools=[mock_tool], max_steps=5)
    agent.add_user_message("use the tool")

    events = []
    async for event in agent.run_stream():
        events.append(event)

    types = [e.type for e in events]
    assert StreamEventType.TOOL_CALL_START in types
    assert StreamEventType.TOOL_CALL_RESULT in types
    assert StreamEventType.DONE in types

    tool_result_event = next(e for e in events if e.type == StreamEventType.TOOL_CALL_RESULT)
    assert tool_result_event.tool_result.success is True
    assert tool_result_event.tool_result.content == "tool output"


@pytest.mark.asyncio
async def test_run_stream_cancellation():
    llm = make_mock_llm([make_text_chunks("Hi")])
    agent = Agent(llm_client=llm, system_prompt="test", tools=[], max_steps=5)
    agent.add_user_message("hello")

    cancel_event = asyncio.Event()
    cancel_event.set()  # Pre-cancel

    events = []
    async for event in agent.run_stream(cancel_event=cancel_event):
        events.append(event)

    assert any(e.type == StreamEventType.CANCELLED for e in events)


@pytest.mark.asyncio
async def test_run_stream_error():
    """LLM stream raises an exception."""
    async def failing_stream(messages, tools=None):
        raise ConnectionError("stream failed")
        yield  # make it a generator

    llm = MagicMock()
    llm.generate_stream = failing_stream
    llm.generate = AsyncMock(return_value=MagicMock(content="summary"))

    agent = Agent(llm_client=llm, system_prompt="test", tools=[], max_steps=5)
    agent.add_user_message("hello")

    events = []
    async for event in agent.run_stream():
        events.append(event)

    error_events = [e for e in events if e.type == StreamEventType.ERROR]
    assert len(error_events) == 1
    assert "stream failed" in error_events[0].content


@pytest.mark.asyncio
async def test_run_stream_max_steps():
    """Agent exceeds max_steps."""
    # Always return a tool call so the loop never ends naturally
    tool_chunks = [
        LLMStreamChunk(type=LLMStreamChunkType.TOOL_CALL_START,
                        tool_call_id="c1", tool_name="test_tool"),
        LLMStreamChunk(type=LLMStreamChunkType.TOOL_CALL_DELTA,
                        tool_call_id="c1", tool_arguments='{}'),
        LLMStreamChunk(type=LLMStreamChunkType.TOOL_CALL_END,
                        tool_call_id="c1", tool_name="test_tool"),
        LLMStreamChunk(type=LLMStreamChunkType.DONE, finish_reason="tool_use"),
    ]

    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.execute = AsyncMock(return_value=ToolResult(success=True, content="ok"))

    llm = make_mock_llm([tool_chunks] * 5)
    agent = Agent(llm_client=llm, system_prompt="test", tools=[mock_tool], max_steps=3)
    agent.add_user_message("loop forever")

    events = []
    async for event in agent.run_stream():
        events.append(event)

    error_events = [e for e in events if e.type == StreamEventType.ERROR]
    assert len(error_events) == 1
    assert "3 steps" in error_events[0].content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_stream.py -v`
Expected: FAIL — `run_stream()` not implemented

- [ ] **Step 3: Implement run_stream() and rewrite run()**

Modify `mini_agent/agent.py`:

1. Add `run_stream()` method — the full agent loop as an `AsyncGenerator[StreamEvent, None]`. See spec section "Agent.run_stream()" for the complete implementation. Key: uses `pending_tools: dict[str, dict]` for multi-tool-call accumulation, yields all event types, handles cancellation, RetryExhaustedError.

2. Rewrite `run()` to wrap `run_stream()`:
```python
async def run(self, cancel_event=None) -> str:
    final_text = ""
    async for event in self.run_stream(cancel_event):
        if event.type == StreamEventType.DONE:
            final_text = event.content or ""
        elif event.type == StreamEventType.ERROR:
            return event.content or "Unknown error"
        elif event.type == StreamEventType.CANCELLED:
            return "Task cancelled by user."
    return final_text
```

3. Remove all `print()` calls from the class (step headers, thinking output, tool calls display, timing, summarization feedback, cleanup feedback).

4. Remove the `Colors` class from `agent.py` (no longer needed).

5. Add imports:
```python
from collections.abc import AsyncGenerator
from .schema import LLMStreamChunk, LLMStreamChunkType, StreamEvent, StreamEventType, ToolResult
```

6. Remove `from .tools.base import Tool, ToolResult` — import `ToolResult` from schema instead. Keep `from .tools.base import Tool`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_stream.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run existing agent tests to check backward compatibility**

Run: `pytest tests/test_agent.py -v`
Expected: PASS (run() still returns str, same behavior)

- [ ] **Step 6: Commit**

```bash
git add mini_agent/agent.py tests/test_agent_stream.py
git commit -m "feat: add Agent.run_stream(), rewrite run() as wrapper, remove print logic"
```

---

## Chunk 4: Consumer Layer Adaptation & Public API

### Task 7: Update CLI to consume run_stream() events

**Files:**
- Modify: `mini_agent/cli.py`

- [ ] **Step 1: Replace agent execution in CLI with event-driven rendering**

In `mini_agent/cli.py`, the agent execution happens in two places:
1. Non-interactive mode (around line 621): `await agent.run()`
2. Interactive mode (around line 810): `agent_task = asyncio.create_task(agent.run())`

Replace both with `agent.run_stream()` consumption. See spec section "CLI (cli.py)" for the event matching pattern.

Key changes:
- Import `sys`, `json`, `StreamEventType` at the top
- Non-interactive: `async for event in agent.run_stream(): ...`
- Interactive: Create an async helper that iterates `run_stream()` with the cancel_event, then wrap that in `asyncio.create_task()`
- Use `sys.stdout.write()` + `sys.stdout.flush()` for `TEXT_DELTA` and `THINKING_DELTA` (character-by-character streaming)
- Keep all existing box-drawing and color formatting from the current CLI for `STEP_START`, `TOOL_CALL_START`, etc.
- Track step start times in the CLI for `STEP_COMPLETE` timing display

- [ ] **Step 2: Smoke test the CLI manually (if config.yaml available)**

Run: `uv run python -m mini_agent.cli --task "Say hello" 2>&1 | head -20`
Expected: See streaming text output, step headers, and clean exit

- [ ] **Step 3: Commit**

```bash
git add mini_agent/cli.py
git commit -m "feat: CLI consumes run_stream() events for streaming terminal output"
```

---

### Task 8: Update ACP server to consume run_stream() events

**Files:**
- Modify: `mini_agent/acp/__init__.py`

- [ ] **Step 1: Rewrite _run_turn() to consume run_stream()**

Replace the current `_run_turn()` method that directly calls `agent.llm.generate()` with an event-driven version that consumes `agent.run_stream()`. See spec section "ACP Server" for the complete implementation.

Key changes:
- Remove the manual agent loop (the `for _ in range(agent.max_steps)` loop)
- Replace with `async for event in state.agent.run_stream():`
- Map `StreamEventType` to ACP protocol calls
- Import `StreamEventType` from `mini_agent.schema`

- [ ] **Step 2: Run existing ACP tests**

Run: `pytest tests/test_acp.py -v`
Expected: PASS (if tests use mocks) or skip (if they need a real server)

- [ ] **Step 3: Commit**

```bash
git add mini_agent/acp/__init__.py
git commit -m "feat: ACP server consumes run_stream() events for incremental updates"
```

---

### Task 9: Update public SDK exports

**Files:**
- Modify: `mini_agent/__init__.py`

- [ ] **Step 1: Add new types to __init__.py exports**

Update `mini_agent/__init__.py`:

```python
"""Mini Agent - Minimal single agent with basic tools and MCP support."""

from .agent import Agent
from .llm import LLMClient
from .schema import (
    FunctionCall,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    LLMStreamChunkType,
    Message,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "LLMClient",
    "LLMProvider",
    "Message",
    "LLMResponse",
    "LLMStreamChunk",
    "LLMStreamChunkType",
    "StreamEvent",
    "StreamEventType",
    "ToolCall",
    "FunctionCall",
    "ToolResult",
]
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "from mini_agent import Agent, StreamEvent, StreamEventType, LLMStreamChunk, LLMStreamChunkType, ToolResult; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add mini_agent/__init__.py
git commit -m "feat: export streaming types from public SDK surface"
```

---

### Task 10: Final verification and cleanup

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS, no import errors, no regressions

- [ ] **Step 2: Verify no remaining print() in agent.py**

Run: `grep -n "print(" mini_agent/agent.py`
Expected: No matches (all print logic moved to CLI)

- [ ] **Step 3: Verify ToolResult is no longer defined in tools/base.py**

Run: `grep -n "class ToolResult" mini_agent/tools/base.py`
Expected: No matches

- [ ] **Step 4: Final commit (if any cleanup needed)**

```bash
git add -A
git commit -m "chore: streaming support cleanup and final verification"
```
