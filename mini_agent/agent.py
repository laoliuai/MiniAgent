"""Core Agent implementation."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Optional

import tiktoken

from .llm import LLMClient
from .logger import AgentLogger
from .schema import (
    FunctionCall,
    LLMStreamChunk,
    LLMStreamChunkType,
    Message,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
)
from .tools.base import Tool

logger = logging.getLogger(__name__)


class Agent:
    """Single agent with basic tools and MCP support."""

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        tools: list[Tool],
        max_steps: int = 50,
        workspace_dir: str = "./workspace",
        token_limit: int = 80000,  # Summary triggered when tokens exceed this value
    ):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.token_limit = token_limit
        self.workspace_dir = Path(workspace_dir)
        # Cancellation event for interrupting agent execution (set externally, e.g., by Esc key)
        self.cancel_event: Optional[asyncio.Event] = None

        # Ensure workspace exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Inject workspace information into system prompt if not already present
        if "Current Workspace" not in system_prompt:
            workspace_info = f"\n\n## Current Workspace\nYou are currently working in: `{self.workspace_dir.absolute()}`\nAll relative paths will be resolved relative to this directory."
            system_prompt = system_prompt + workspace_info

        self.system_prompt = system_prompt

        # Initialize message history
        self.messages: list[Message] = [Message(role="system", content=system_prompt)]

        # Initialize logger
        self.logger = AgentLogger()

        # Token usage from last API response (updated after each LLM call)
        self.api_total_tokens: int = 0
        # Flag to skip token check right after summary (avoid consecutive triggers)
        self._skip_next_token_check: bool = False

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))

    def _check_cancelled(self) -> bool:
        """Check if agent execution has been cancelled.

        Returns:
            True if cancelled, False otherwise.
        """
        if self.cancel_event is not None and self.cancel_event.is_set():
            return True
        return False

    def _cleanup_incomplete_messages(self):
        """Remove the incomplete assistant message and its partial tool results.

        This ensures message consistency after cancellation by removing
        only the current step's incomplete messages, preserving completed steps.
        """
        # Find the index of the last assistant message
        last_assistant_idx = -1
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].role == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx == -1:
            # No assistant message found, nothing to clean
            return

        # Remove the last assistant message and all tool results after it
        removed_count = len(self.messages) - last_assistant_idx
        if removed_count > 0:
            self.messages = self.messages[:last_assistant_idx]
            logger.info(f"Cleaned up {removed_count} incomplete message(s)")

    def _estimate_tokens(self) -> int:
        """Accurately calculate token count for message history using tiktoken

        Uses cl100k_base encoder (GPT-4/Claude/M2 compatible)
        """
        try:
            # Use cl100k_base encoder (used by GPT-4 and most modern models)
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback: if tiktoken initialization fails, use simple estimation
            return self._estimate_tokens_fallback()

        total_tokens = 0

        for msg in self.messages:
            # Count text content
            if isinstance(msg.content, str):
                total_tokens += len(encoding.encode(msg.content))
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        # Convert dict to string for calculation
                        total_tokens += len(encoding.encode(str(block)))

            # Count thinking
            if msg.thinking:
                total_tokens += len(encoding.encode(msg.thinking))

            # Count tool_calls
            if msg.tool_calls:
                total_tokens += len(encoding.encode(str(msg.tool_calls)))

            # Metadata overhead per message (approximately 4 tokens)
            total_tokens += 4

        return total_tokens

    def _estimate_tokens_fallback(self) -> int:
        """Fallback token estimation method (when tiktoken is unavailable)"""
        total_chars = 0
        for msg in self.messages:
            if isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        total_chars += len(str(block))

            if msg.thinking:
                total_chars += len(msg.thinking)

            if msg.tool_calls:
                total_chars += len(str(msg.tool_calls))

        # Rough estimation: average 2.5 characters = 1 token
        return int(total_chars / 2.5)

    async def _summarize_messages(self):
        """Message history summarization: summarize conversations between user messages when tokens exceed limit

        Strategy (Agent mode):
        - Keep all user messages (these are user intents)
        - Summarize content between each user-user pair (agent execution process)
        - If last round is still executing (has agent/tool messages but no next user), also summarize
        - Structure: system -> user1 -> summary1 -> user2 -> summary2 -> user3 -> summary3 (if executing)

        Summary is triggered when EITHER:
        - Local token estimation exceeds limit
        - API reported total_tokens exceeds limit
        """
        # Skip check if we just completed a summary (wait for next LLM call to update api_total_tokens)
        if self._skip_next_token_check:
            self._skip_next_token_check = False
            return

        estimated_tokens = self._estimate_tokens()

        # Check both local estimation and API reported tokens
        should_summarize = estimated_tokens > self.token_limit or self.api_total_tokens > self.token_limit

        # If neither exceeded, no summary needed
        if not should_summarize:
            return

        logger.info(
            f"Token usage - Local estimate: {estimated_tokens}, API reported: {self.api_total_tokens}, Limit: {self.token_limit}"
        )
        logger.info("Triggering message history summarization...")

        # Find all user message indices (skip system prompt)
        user_indices = [i for i, msg in enumerate(self.messages) if msg.role == "user" and i > 0]

        # Need at least 1 user message to perform summary
        if len(user_indices) < 1:
            logger.warning("Insufficient messages, cannot summarize")
            return

        # Build new message list
        new_messages = [self.messages[0]]  # Keep system prompt
        summary_count = 0

        # Iterate through each user message and summarize the execution process after it
        for i, user_idx in enumerate(user_indices):
            # Add current user message
            new_messages.append(self.messages[user_idx])

            # Determine message range to summarize
            # If last user, go to end of message list; otherwise to before next user
            if i < len(user_indices) - 1:
                next_user_idx = user_indices[i + 1]
            else:
                next_user_idx = len(self.messages)

            # Extract execution messages for this round
            execution_messages = self.messages[user_idx + 1 : next_user_idx]

            # If there are execution messages in this round, summarize them
            if execution_messages:
                summary_text = await self._create_summary(execution_messages, i + 1)
                if summary_text:
                    summary_message = Message(
                        role="user",
                        content=f"[Assistant Execution Summary]\n\n{summary_text}",
                    )
                    new_messages.append(summary_message)
                    summary_count += 1

        # Replace message list
        self.messages = new_messages

        # Skip next token check to avoid consecutive summary triggers
        # (api_total_tokens will be updated after next LLM call)
        self._skip_next_token_check = True

        new_tokens = self._estimate_tokens()
        logger.info(f"Summary completed, local tokens: {estimated_tokens} -> {new_tokens}")
        logger.info(f"Structure: system + {len(user_indices)} user messages + {summary_count} summaries")
        logger.info("Note: API token count will update on next LLM call")

    async def _create_summary(self, messages: list[Message], round_num: int) -> str:
        """Create summary for one execution round

        Args:
            messages: List of messages to summarize
            round_num: Round number

        Returns:
            Summary text
        """
        if not messages:
            return ""

        # Build summary content
        summary_content = f"Round {round_num} execution process:\n\n"
        for msg in messages:
            if msg.role == "assistant":
                content_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                summary_content += f"Assistant: {content_text}\n"
                if msg.tool_calls:
                    tool_names = [tc.function.name for tc in msg.tool_calls]
                    summary_content += f"  -> Called tools: {', '.join(tool_names)}\n"
            elif msg.role == "tool":
                result_preview = msg.content if isinstance(msg.content, str) else str(msg.content)
                summary_content += f"  <- Tool returned: {result_preview}...\n"

        # Call LLM to generate concise summary
        try:
            summary_prompt = f"""Please provide a concise summary of the following Agent execution process:

{summary_content}

Requirements:
1. Focus on what tasks were completed and which tools were called
2. Keep key execution results and important findings
3. Be concise and clear, within 1000 words
4. Use English
5. Do not include "user" related content, only summarize the Agent's execution process"""

            summary_msg = Message(role="user", content=summary_prompt)
            response = await self.llm.generate(
                messages=[
                    Message(
                        role="system",
                        content="You are an assistant skilled at summarizing Agent execution processes.",
                    ),
                    summary_msg,
                ]
            )

            summary_text = response.content
            logger.info(f"Summary for round {round_num} generated successfully")
            return summary_text

        except Exception as e:
            logger.error(f"Summary generation failed for round {round_num}: {e}")
            # Use simple text summary on failure
            return summary_content

    async def run_stream(self, cancel_event: Optional[asyncio.Event] = None) -> AsyncGenerator[StreamEvent, None]:
        """Execute agent loop as an async generator, yielding StreamEvent objects.

        This is the core agent loop. All output is delivered via events —
        no print statements, no direct terminal output.

        Args:
            cancel_event: Optional asyncio.Event that can be set to cancel execution.
                          When set, the agent will stop at the next safe checkpoint.

        Yields:
            StreamEvent objects representing each phase of the agent loop.
        """
        if cancel_event is not None:
            self.cancel_event = cancel_event
        self.logger.start_new_run()

        for step in range(self.max_steps):
            # Cancellation check
            if self._check_cancelled():
                self._cleanup_incomplete_messages()
                yield StreamEvent(type=StreamEventType.CANCELLED)
                return

            yield StreamEvent(type=StreamEventType.STEP_START, step=step + 1)

            # Token management
            await self._summarize_messages()

            # Stream LLM response
            text, thinking = "", ""
            tool_calls = []
            pending_tools: dict[str, dict] = {}
            tool_list = list(self.tools.values())
            self.logger.log_request(messages=self.messages, tools=tool_list)

            try:
                async for chunk in self.llm.generate_stream(self.messages, tool_list):
                    match chunk.type:
                        case LLMStreamChunkType.TEXT_DELTA:
                            text += chunk.content
                            yield StreamEvent(type=StreamEventType.TEXT_DELTA,
                                              content=chunk.content, step=step + 1)
                        case LLMStreamChunkType.THINKING_DELTA:
                            thinking += chunk.content
                            yield StreamEvent(type=StreamEventType.THINKING_DELTA,
                                              content=chunk.content, step=step + 1)
                        case LLMStreamChunkType.TOOL_CALL_START:
                            pending_tools[chunk.tool_call_id] = {
                                "id": chunk.tool_call_id,
                                "name": chunk.tool_name,
                                "arguments_json": ""
                            }
                        case LLMStreamChunkType.TOOL_CALL_DELTA:
                            pending_tools[chunk.tool_call_id]["arguments_json"] += chunk.tool_arguments
                        case LLMStreamChunkType.TOOL_CALL_END:
                            info = pending_tools[chunk.tool_call_id]
                            tc = ToolCall(id=info["id"], type="function",
                                          function=FunctionCall(
                                              name=info["name"],
                                              arguments=json.loads(info["arguments_json"])))
                            tool_calls.append(tc)
                        case LLMStreamChunkType.USAGE:
                            if chunk.usage:
                                self.api_total_tokens = chunk.usage.total_tokens
            except Exception as e:
                from .retry import RetryExhaustedError
                if isinstance(e, RetryExhaustedError):
                    msg = f"LLM call failed after {e.attempts} retries. Last error: {e.last_exception}"
                else:
                    msg = str(e)
                yield StreamEvent(type=StreamEventType.ERROR, content=msg)
                return

            # Append complete assistant message to history
            assistant_msg = Message(role="assistant", content=text,
                                    thinking=thinking or None,
                                    tool_calls=tool_calls or None)
            self.messages.append(assistant_msg)
            self.logger.log_response(content=text, thinking=thinking or None,
                                      tool_calls=tool_calls or None, finish_reason="stop")

            # No tool calls -> task complete
            if not tool_calls:
                yield StreamEvent(type=StreamEventType.STEP_COMPLETE, step=step + 1)
                yield StreamEvent(type=StreamEventType.DONE, content=text)
                return

            # Cancellation check before tool execution
            if self._check_cancelled():
                self._cleanup_incomplete_messages()
                yield StreamEvent(type=StreamEventType.CANCELLED)
                return

            # Execute tool calls
            for tool_call in tool_calls:
                yield StreamEvent(type=StreamEventType.TOOL_CALL_START,
                                  tool_name=tool_call.function.name,
                                  tool_call_id=tool_call.id,
                                  tool_arguments=tool_call.function.arguments,
                                  step=step + 1)

                if tool_call.function.name not in self.tools:
                    result = ToolResult(success=False, content="",
                                        error=f"Unknown tool: {tool_call.function.name}")
                else:
                    try:
                        tool = self.tools[tool_call.function.name]
                        result = await tool.execute(**tool_call.function.arguments)
                    except Exception as e:
                        import traceback
                        result = ToolResult(success=False, content="",
                                            error=f"Tool execution failed: {e}\n{traceback.format_exc()}")

                self.logger.log_tool_result(tool_name=tool_call.function.name,
                                             arguments=tool_call.function.arguments,
                                             result_success=result.success,
                                             result_content=result.content if result.success else None,
                                             result_error=result.error if not result.success else None)

                yield StreamEvent(type=StreamEventType.TOOL_CALL_RESULT,
                                  tool_name=tool_call.function.name,
                                  tool_call_id=tool_call.id,
                                  tool_result=result,
                                  step=step + 1)

                self.messages.append(Message(
                    role="tool",
                    content=result.content if result.success else f"Error: {result.error}",
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name))

                # Cancellation check after each tool
                if self._check_cancelled():
                    self._cleanup_incomplete_messages()
                    yield StreamEvent(type=StreamEventType.CANCELLED)
                    return

            yield StreamEvent(type=StreamEventType.STEP_COMPLETE, step=step + 1)

        # Max steps exceeded
        yield StreamEvent(type=StreamEventType.ERROR,
                          content=f"Task couldn't be completed after {self.max_steps} steps.")

    async def run(self, cancel_event: Optional[asyncio.Event] = None) -> str:
        """Execute agent loop until task is complete or max steps reached.

        This is a convenience wrapper around run_stream() that collects events
        and returns the final text result.

        Args:
            cancel_event: Optional asyncio.Event that can be set to cancel execution.
                          When set, the agent will stop at the next safe checkpoint
                          (after completing the current step to keep messages consistent).

        Returns:
            The final response content, or error message (including cancellation message).
        """
        final_text = ""
        async for event in self.run_stream(cancel_event):
            if event.type == StreamEventType.DONE:
                final_text = event.content or ""
            elif event.type == StreamEventType.ERROR:
                return event.content or "Unknown error"
            elif event.type == StreamEventType.CANCELLED:
                return "Task cancelled by user."
        return final_text

    def get_history(self) -> list[Message]:
        """Get message history."""
        return self.messages.copy()
