"""Tests for AnthropicClient.generate_stream() with mocked SDK."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


def _make_client(**kwargs):
    """Create an AnthropicClient with the SDK client constructor mocked out."""
    with patch("mini_agent.llm.anthropic_client.anthropic.AsyncAnthropic"):
        from mini_agent.llm.anthropic_client import AnthropicClient
        defaults = {"api_key": "fake", "api_base": "http://fake", "model": "test"}
        defaults.update(kwargs)
        return AnthropicClient(**defaults)


@pytest.mark.asyncio
async def test_anthropic_stream_text_only():
    client = _make_client()

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
    client = _make_client()

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
    client = _make_client()

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
    client = _make_client(
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
