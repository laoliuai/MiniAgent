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
