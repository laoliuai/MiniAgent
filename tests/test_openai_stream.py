"""Tests for OpenAIClient.generate_stream() with mocked SDK."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent.schema import LLMStreamChunkType, Message


def make_openai_chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    """Create a mock OpenAI stream chunk."""
    chunk = MagicMock()
    choice = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
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


def _make_client(**kwargs):
    """Create an OpenAIClient with the SDK client constructor mocked out."""
    with patch("mini_agent.llm.openai_client.AsyncOpenAI"):
        from mini_agent.llm.openai_client import OpenAIClient
        defaults = {"api_key": "fake", "api_base": "http://fake", "model": "test"}
        defaults.update(kwargs)
        return OpenAIClient(**defaults)


@pytest.mark.asyncio
async def test_openai_stream_text_only():
    client = _make_client()

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
    client = _make_client()

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
    client = _make_client()

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
    client = _make_client()

    stream_chunks = [
        make_openai_chunk(content=None),
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
