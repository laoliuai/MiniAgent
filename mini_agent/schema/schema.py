from enum import Enum
from typing import Any

from pydantic import BaseModel


class LLMProvider(str, Enum):
    """LLM provider types."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class FunctionCall(BaseModel):
    """Function call details."""

    name: str
    arguments: dict[str, Any]  # Function arguments as dict


class ToolCall(BaseModel):
    """Tool call structure."""

    id: str
    type: str  # "function"
    function: FunctionCall


class Message(BaseModel):
    """Chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | list[dict[str, Any]]  # Can be string or list of content blocks
    thinking: str | None = None  # Extended thinking content for assistant messages
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # For tool role


class TokenUsage(BaseModel):
    """Token usage statistics from LLM API response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """LLM response."""

    content: str
    thinking: str | None = None  # Extended thinking blocks
    tool_calls: list[ToolCall] | None = None
    finish_reason: str
    usage: TokenUsage | None = None  # Token usage from API response


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
