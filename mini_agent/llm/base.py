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
        """Initialize the LLM client.

        Args:
            api_key: API key for authentication
            api_base: Base URL for the API
            model: Model name to use
            retry_config: Optional retry configuration
        """
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.retry_config = retry_config or RetryConfig()

        # Callback for tracking retry count
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
        """Prepare the request payload for the API.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            Dictionary containing the request payload
        """
        pass

    @abstractmethod
    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert internal message format to API-specific format.

        Args:
            messages: List of internal Message objects

        Returns:
            Tuple of (system_message, api_messages)
        """
        pass
