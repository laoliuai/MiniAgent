"""OpenAI LLM client implementation."""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from ..retry import RetryConfig, async_retry
from ..schema import LLMStreamChunk, LLMStreamChunkType, Message, TokenUsage
from .base import LLMClientBase

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClientBase):
    """LLM client using OpenAI's protocol.

    This client uses the official OpenAI SDK and supports:
    - Streaming responses via generate_stream()
    - Reasoning content (via reasoning_split=True)
    - Tool calling
    - Retry logic (connection phase only)
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.minimaxi.com/v1",
        model: str = "MiniMax-M2.5",
        retry_config: RetryConfig | None = None,
    ):
        """Initialize OpenAI client.

        Args:
            api_key: API key for authentication
            api_base: Base URL for the API (default: MiniMax OpenAI endpoint)
            model: Model name to use (default: MiniMax-M2.5)
            retry_config: Optional retry configuration
        """
        super().__init__(api_key, api_base, model, retry_config)

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to OpenAI format.

        Args:
            tools: List of Tool objects or dicts

        Returns:
            List of tools in OpenAI dict format
        """
        result = []
        for tool in tools:
            if isinstance(tool, dict):
                # If already a dict, check if it's in OpenAI format
                if "type" in tool and tool["type"] == "function":
                    result.append(tool)
                else:
                    # Assume it's in Anthropic format, convert to OpenAI
                    result.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "description": tool["description"],
                                "parameters": tool["input_schema"],
                            },
                        }
                    )
            elif hasattr(tool, "to_openai_schema"):
                # Tool object with to_openai_schema method
                result.append(tool.to_openai_schema())
            else:
                raise TypeError(f"Unsupported tool type: {type(tool)}")
        return result

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert internal messages to OpenAI format.

        Args:
            messages: List of internal Message objects

        Returns:
            Tuple of (system_message, api_messages)
            Note: OpenAI includes system message in the messages array
        """
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                # OpenAI includes system message in messages array
                api_messages.append({"role": "system", "content": msg.content})
                continue

            # For user messages
            if msg.role == "user":
                api_messages.append({"role": "user", "content": msg.content})

            # For assistant messages
            elif msg.role == "assistant":
                assistant_msg = {"role": "assistant"}

                # Add content if present
                if msg.content:
                    assistant_msg["content"] = msg.content

                # Add tool calls if present
                if msg.tool_calls:
                    tool_calls_list = []
                    for tool_call in msg.tool_calls:
                        tool_calls_list.append(
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": json.dumps(tool_call.function.arguments),
                                },
                            }
                        )
                    assistant_msg["tool_calls"] = tool_calls_list

                # IMPORTANT: Add reasoning_details if thinking is present
                # This is CRITICAL for Interleaved Thinking to work properly!
                # The complete response_message (including reasoning_details) must be
                # preserved in Message History and passed back to the model in the next turn.
                # This ensures the model's chain of thought is not interrupted.
                if msg.thinking:
                    assistant_msg["reasoning_details"] = [{"text": msg.thinking}]

                api_messages.append(assistant_msg)

            # For tool result messages
            elif msg.role == "tool":
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )

        return None, api_messages

    def _prepare_request(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare the request for OpenAI API.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            Dictionary containing request parameters
        """
        _, api_messages = self._convert_messages(messages)

        return {
            "api_messages": api_messages,
            "tools": tools,
        }

    async def _make_stream_request(self, params: dict[str, Any]) -> Any:
        """Create a streaming request via the OpenAI SDK.

        Calls self.client.chat.completions.create(**params, stream=True)
        and returns the async iterator.

        Args:
            params: Request parameters for the OpenAI chat completions API

        Returns:
            Async iterator of stream chunks
        """
        return await self.client.chat.completions.create(**params, stream=True)

    async def _make_stream_request_with_retry(self, params: dict[str, Any]) -> Any:
        """Wrap _make_stream_request with retry logic (connection phase only).

        Args:
            params: Request parameters for the OpenAI chat completions API

        Returns:
            Async iterator of stream chunks
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
        """Stream response chunks from OpenAI LLM.

        Yields LLMStreamChunk objects as events arrive from the OpenAI SDK
        streaming API. Handles text, thinking (reasoning_content), and tool calls.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Yields:
            LLMStreamChunk for each meaningful event in the stream
        """
        # Build API params
        _, api_messages = self._convert_messages(messages)
        params: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "extra_body": {"reasoning_split": True},
        }
        if tools:
            params["tools"] = self._convert_tools(tools)

        # Get stream (with retry on connection)
        stream = await self._make_stream_request_with_retry(params)

        # Track tool calls by index position
        current_tool_calls: dict[int, dict] = {}

        async for openai_chunk in stream:
            # Skip chunks with no choices
            if not openai_chunk.choices:
                continue

            choice = openai_chunk.choices[0]
            delta = choice.delta

            # Check for thinking content (reasoning_content)
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                yield LLMStreamChunk(
                    type=LLMStreamChunkType.THINKING_DELTA,
                    content=reasoning_content,
                )

            # Check for text content
            if delta.content:
                yield LLMStreamChunk(
                    type=LLMStreamChunkType.TEXT_DELTA,
                    content=delta.content,
                )

            # Check for tool call deltas
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index

                    # First appearance of this tool call (has id and name)
                    if tc.id:
                        current_tool_calls[idx] = {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": "",
                        }
                        yield LLMStreamChunk(
                            type=LLMStreamChunkType.TOOL_CALL_START,
                            tool_call_id=tc.id,
                            tool_name=tc.function.name,
                        )

                    # Accumulate arguments
                    if tc.function and tc.function.arguments:
                        current_tool_calls[idx]["arguments"] += tc.function.arguments
                        yield LLMStreamChunk(
                            type=LLMStreamChunkType.TOOL_CALL_DELTA,
                            tool_call_id=current_tool_calls[idx]["id"],
                            tool_arguments=tc.function.arguments,
                        )

            # Check for finish reason
            if choice.finish_reason:
                # Emit TOOL_CALL_END for each tracked tool call
                for tool_info in current_tool_calls.values():
                    yield LLMStreamChunk(
                        type=LLMStreamChunkType.TOOL_CALL_END,
                        tool_call_id=tool_info["id"],
                        tool_name=tool_info["name"],
                    )

                # Check for usage info
                if openai_chunk.usage:
                    yield LLMStreamChunk(
                        type=LLMStreamChunkType.USAGE,
                        usage=TokenUsage(
                            prompt_tokens=openai_chunk.usage.prompt_tokens or 0,
                            completion_tokens=openai_chunk.usage.completion_tokens or 0,
                            total_tokens=openai_chunk.usage.total_tokens or 0,
                        ),
                    )

                yield LLMStreamChunk(
                    type=LLMStreamChunkType.DONE,
                    finish_reason=choice.finish_reason,
                )
