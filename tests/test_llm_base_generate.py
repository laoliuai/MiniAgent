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
