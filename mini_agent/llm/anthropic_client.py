"""Anthropic LLM client implementation."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

import anthropic

from ..retry import RetryConfig, async_retry
from ..schema import LLMStreamChunk, LLMStreamChunkType, Message, TokenUsage
from .base import LLMClientBase

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClientBase):
    """LLM client using Anthropic's protocol.

    This client uses the official Anthropic SDK and supports:
    - Streaming responses via generate_stream()
    - Extended thinking content
    - Tool calling
    - Retry logic (connection phase only)
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.minimaxi.com/anthropic",
        model: str = "MiniMax-M2.5",
        retry_config: RetryConfig | None = None,
    ):
        """Initialize Anthropic client.

        Args:
            api_key: API key for authentication
            api_base: Base URL for the API (default: MiniMax Anthropic endpoint)
            model: Model name to use (default: MiniMax-M2.5)
            retry_config: Optional retry configuration
        """
        super().__init__(api_key, api_base, model, retry_config)

        # Initialize Anthropic async client
        self.client = anthropic.AsyncAnthropic(
            base_url=api_base,
            api_key=api_key,
            default_headers={"Authorization": f"Bearer {api_key}"},
        )

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to Anthropic format.

        Anthropic tool format:
        {
            "name": "tool_name",
            "description": "Tool description",
            "input_schema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }

        Args:
            tools: List of Tool objects or dicts

        Returns:
            List of tools in Anthropic dict format
        """
        result = []
        for tool in tools:
            if isinstance(tool, dict):
                result.append(tool)
            elif hasattr(tool, "to_schema"):
                # Tool object with to_schema method
                result.append(tool.to_schema())
            else:
                raise TypeError(f"Unsupported tool type: {type(tool)}")
        return result

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert internal messages to Anthropic format.

        Args:
            messages: List of internal Message objects

        Returns:
            Tuple of (system_message, api_messages)
        """
        system_message = None
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
                continue

            # For user and assistant messages
            if msg.role in ["user", "assistant"]:
                # Handle assistant messages with thinking or tool calls
                if msg.role == "assistant" and (msg.thinking or msg.tool_calls):
                    # Build content blocks for assistant with thinking and/or tool calls
                    content_blocks = []

                    # Add thinking block if present
                    if msg.thinking:
                        content_blocks.append({"type": "thinking", "thinking": msg.thinking})

                    # Add text content if present
                    if msg.content:
                        content_blocks.append({"type": "text", "text": msg.content})

                    # Add tool use blocks
                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            content_blocks.append(
                                {
                                    "type": "tool_use",
                                    "id": tool_call.id,
                                    "name": tool_call.function.name,
                                    "input": tool_call.function.arguments,
                                }
                            )

                    api_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    api_messages.append({"role": msg.role, "content": msg.content})

            # For tool result messages
            elif msg.role == "tool":
                # Anthropic uses user role with tool_result content blocks
                api_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )

        return system_message, api_messages

    def _prepare_request(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare the request for Anthropic API.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            Dictionary containing request parameters
        """
        system_message, api_messages = self._convert_messages(messages)

        return {
            "system_message": system_message,
            "api_messages": api_messages,
            "tools": tools,
        }

    async def _make_stream_request(self, params: dict[str, Any]) -> Any:
        """Create a streaming request via the Anthropic SDK.

        This is a thin wrapper that calls self.client.messages.stream(**params)
        and returns the stream context manager.

        Args:
            params: Request parameters for the Anthropic messages API

        Returns:
            Async context manager for the stream
        """
        return self.client.messages.stream(**params)

    async def _make_stream_request_with_retry(self, params: dict[str, Any]) -> Any:
        """Wrap _make_stream_request with retry logic (connection phase only).

        Args:
            params: Request parameters for the Anthropic messages API

        Returns:
            Async context manager for the stream
        """
        if self.retry_config.enabled:
            retry_decorator = async_retry(
                config=self.retry_config, on_retry=self.retry_callback
            )
            retryable_request = retry_decorator(self._make_stream_request)
            return await retryable_request(params)
        else:
            return await self._make_stream_request(params)

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Stream response chunks from Anthropic LLM.

        Yields LLMStreamChunk objects as events arrive from the Anthropic SDK
        streaming API. Handles text, thinking, and tool_use content blocks.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Yields:
            LLMStreamChunk for each meaningful event in the stream
        """
        # Build API params
        system_message, api_messages = self._convert_messages(messages)
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16384,
            "messages": api_messages,
        }
        if system_message:
            params["system"] = system_message
        if tools:
            params["tools"] = self._convert_tools(tools)

        # Get stream context manager (with retry on connection)
        stream_cm = await self._make_stream_request_with_retry(params)

        content_blocks: dict[int, dict] = {}
        input_tokens = 0

        async with stream_cm as stream:
            async for event in stream:
                if event.type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        input_tokens = getattr(event.message.usage, "input_tokens", 0) or 0

                elif event.type == "content_block_start":
                    idx = event.index
                    block_type = event.content_block.type
                    content_blocks[idx] = {"type": block_type}

                    if block_type == "tool_use":
                        content_blocks[idx]["id"] = event.content_block.id
                        content_blocks[idx]["name"] = event.content_block.name
                        yield LLMStreamChunk(
                            type=LLMStreamChunkType.TOOL_CALL_START,
                            tool_call_id=event.content_block.id,
                            tool_name=event.content_block.name,
                        )

                elif event.type == "content_block_delta":
                    delta_type = event.delta.type

                    if delta_type == "text_delta":
                        yield LLMStreamChunk(
                            type=LLMStreamChunkType.TEXT_DELTA,
                            content=event.delta.text,
                        )
                    elif delta_type == "thinking_delta":
                        yield LLMStreamChunk(
                            type=LLMStreamChunkType.THINKING_DELTA,
                            content=event.delta.thinking,
                        )
                    elif delta_type == "input_json_delta":
                        block_info = content_blocks.get(event.index, {})
                        yield LLMStreamChunk(
                            type=LLMStreamChunkType.TOOL_CALL_DELTA,
                            tool_call_id=block_info.get("id"),
                            tool_arguments=event.delta.partial_json,
                        )

                elif event.type == "content_block_stop":
                    block_info = content_blocks.get(event.index, {})
                    if block_info.get("type") == "tool_use":
                        yield LLMStreamChunk(
                            type=LLMStreamChunkType.TOOL_CALL_END,
                            tool_call_id=block_info.get("id"),
                            tool_name=block_info.get("name"),
                        )

                elif event.type == "message_delta":
                    output_tokens = getattr(event.usage, "output_tokens", 0) or 0
                    total = input_tokens + output_tokens
                    yield LLMStreamChunk(
                        type=LLMStreamChunkType.USAGE,
                        usage=TokenUsage(
                            prompt_tokens=input_tokens,
                            completion_tokens=output_tokens,
                            total_tokens=total,
                        ),
                    )
                    yield LLMStreamChunk(
                        type=LLMStreamChunkType.DONE,
                        finish_reason=event.delta.stop_reason or "stop",
                    )
