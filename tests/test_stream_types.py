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
