"""Core Agent implementation."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Optional

from .context import ContextConfig, ContextManager
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
        context_config: ContextConfig | None = None,
        log_file_level: str = "standard",
        log_console_level: str = "minimal",
        log_max_files: int = 50,
        # Deprecated: kept for backward compatibility, ignored if context_config is provided
        token_limit: int = 80000,
    ):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
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

        # Initialize logger
        from .config import LogLevel
        self.logger = AgentLogger(
            file_level=LogLevel(log_file_level),
            console_level=LogLevel(log_console_level),
            max_files=log_max_files,
        )

        # Initialize ContextManager
        if context_config is None:
            context_config = ContextConfig(total_token_budget=token_limit)
        self.context_manager = ContextManager(context_config, self.llm, self.tools)

        # Keep self.messages for backward compatibility (populated from ContextManager)
        self.messages: list[Message] = []

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.context_manager.add_user_message(content)

    def _dicts_to_messages(self, dicts: list[dict]) -> list[Message]:
        """Convert BudgetAssembler output dicts to Message objects for LLM client."""
        messages = []
        for d in dicts:
            # Handle tool_use blocks (assistant with tool_use)
            if "tool_use" in d:
                tu = d["tool_use"]
                messages.append(Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(
                        id=tu["id"], type="function",
                        function=FunctionCall(name=tu["name"],
                                              arguments=tu["input"]),
                    )],
                ))
                continue
            # Handle tool_result blocks
            content = d.get("content", "")
            if isinstance(content, list) and content and content[0].get("type") == "tool_result":
                tr = content[0]
                messages.append(Message(
                    role="tool", content=tr["content"],
                    tool_call_id=tr["tool_use_id"], name="",
                ))
                continue
            # Normal message
            messages.append(Message(role=d["role"], content=content or ""))
        return messages

    def _check_cancelled(self) -> bool:
        """Check if agent execution has been cancelled.

        Returns:
            True if cancelled, False otherwise.
        """
        if self.cancel_event is not None and self.cancel_event.is_set():
            return True
        return False

    def _cleanup_incomplete_messages(self):
        """Remove incomplete blocks from current turn on cancellation."""
        from .context.models import BlockType, BlockStatus
        current = self.context_manager.current_turn
        blocks_to_remove = []
        for block in self.context_manager.store.all():
            if (block.turn_id == current
                    and block.block_type == BlockType.TOOL_CALL
                    and block.status == BlockStatus.ACTIVE):
                blocks_to_remove.append(block.id)
        for block_id in blocks_to_remove:
            self.context_manager.store.remove(block_id)
            logger.info(f"Cleaned up incomplete block: {block_id}")

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

        # Init system on first call
        if not self.context_manager.store.get("system"):
            self.context_manager.init_system(self.system_prompt)

        for step in range(self.max_steps):
            # Cancellation check
            if self._check_cancelled():
                self._cleanup_incomplete_messages()
                yield StreamEvent(type=StreamEventType.CANCELLED)
                return

            yield StreamEvent(type=StreamEventType.STEP_START, step=step + 1)

            # Context management: process and assemble messages
            message_dicts, ctx_events = await self.context_manager.process_and_assemble()
            for event in ctx_events:
                self.logger.log_context_event(*event.split(": ", 1))
            messages = self._dicts_to_messages(message_dicts)
            # Update self.messages for backward compatibility / logging
            self.messages = messages

            # Stream LLM response
            text, thinking = "", ""
            tool_calls = []
            pending_tools: dict[str, dict] = {}
            finish_reason = "stop"
            tool_list = list(self.tools.values())
            # Include context editing tools if enabled
            context_tools = self.context_manager.get_context_tools()
            all_tools = tool_list + context_tools  # Mix Tool objects + dicts (both supported)
            self.logger.log_request(messages=messages, tools=tool_list)

            try:
                async for chunk in self.llm.generate_stream(messages, all_tools):
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
                            pass  # ContextManager tracks tokens internally
                        case LLMStreamChunkType.DONE:
                            finish_reason = chunk.finish_reason or "stop"
            except Exception as e:
                from .retry import RetryExhaustedError
                if isinstance(e, RetryExhaustedError):
                    msg = f"LLM call failed after {e.attempts} retries. Last error: {e.last_exception}"
                else:
                    msg = str(e)
                yield StreamEvent(type=StreamEventType.ERROR, content=msg)
                return

            self.logger.log_response(content=text, thinking=thinking or None,
                                      tool_calls=tool_calls or None, finish_reason=finish_reason,
                                      usage=None)

            # No tool calls -> task complete
            if not tool_calls:
                # Record assistant reply in ContextManager
                self.context_manager.add_assistant_reply(text, thinking or None)
                yield StreamEvent(type=StreamEventType.STEP_COMPLETE, step=step + 1)
                yield StreamEvent(type=StreamEventType.DONE, content=text)
                return

            # Record assistant text (if any) alongside tool calls
            if text:
                self.context_manager.add_assistant_reply(text, thinking or None)

            # Cancellation check before tool execution
            if self._check_cancelled():
                self._cleanup_incomplete_messages()
                yield StreamEvent(type=StreamEventType.CANCELLED)
                return

            # Execute tool calls
            for tool_call in tool_calls:
                # Route context_* tools to ContextEditor (invisible to user)
                if tool_call.function.name.startswith("context_"):
                    edit_result = self.context_manager.handle_context_tool(
                        tool_call.function.name,
                        tool_call.function.arguments,
                    )
                    self.logger.log_context_event(
                        "context_edit",
                        f"{tool_call.function.name} → {edit_result}",
                    )
                    continue

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

                # Record tool call in ContextManager
                self.context_manager.add_tool_call(
                    tool_call.function.name,
                    tool_call.function.arguments,
                    result.content if result.success else f"Error: {result.error}",
                    tool_call_id=tool_call.id,
                )

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

    def get_history(self) -> list[dict]:
        """Return current context blocks as summary for external inspection."""
        return [{"id": b.id, "type": b.block_type.value, "status": b.status.value,
                 "turn": b.turn_id, "tokens": b.token_count}
                for b in self.context_manager.store.active_blocks()]
