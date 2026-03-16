"""Core Agent implementation."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Optional

from .agent_config import AgentConfig, SubAgentRunner
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
from .shared_state import SharedState
from .tools.base import Tool

logger = logging.getLogger(__name__)


class Agent:
    """Single agent with basic tools and MCP support."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: AgentConfig | None = None,
        workspace_dir: str | Path = "./workspace",
        shared_state: SharedState | None = None,
        logger: AgentLogger | None = None,
        path_guard=None,
        # Legacy params for backward compatibility
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        max_steps: int | None = None,
        context_config: ContextConfig | None = None,
        log_file_level: str = "standard",
        log_console_level: str = "minimal",
        log_max_files: int = 50,
        token_limit: int = 80000,
    ):
        # Handle backward compatibility: construct AgentConfig from legacy params
        if config is None:
            config = AgentConfig(
                system_prompt=system_prompt or "",
                tools=list(tools) if tools else [],
                max_steps_per_turn=max_steps or 50,
                max_steps_total=max_steps or 50,
                context_config=context_config,
            )

        self.config = config
        self.llm = llm_client
        self.workspace_dir = Path(workspace_dir)
        self.shared_state = shared_state
        self.path_guard = path_guard
        self.cancel_event: Optional[asyncio.Event] = None

        # Tools: from config
        self.tools = {tool.name: tool for tool in config.tools}

        # Sub-agent registry
        self._sub_agents: dict[str, AgentConfig] = {}
        self._delegation_depth: int = 0
        self._step_count: int = 0

        # Workspace setup
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # System prompt
        self.system_prompt = self._build_system_prompt(config.system_prompt)

        # Logger
        from .config import LogLevel
        self.logger = logger or AgentLogger(
            file_level=LogLevel(log_file_level),
            console_level=LogLevel(log_console_level),
            max_files=log_max_files,
        )

        # Auto-register delegation + state tools
        self._register_auto_tools()

        # ContextManager
        ctx = config.context_config or context_config
        if ctx is None:
            ctx = ContextConfig(total_token_budget=token_limit)
        self.context_manager = ContextManager(ctx, self.llm, self.tools)

        # Backward compat
        self.messages: list[Message] = []

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.context_manager.add_user_message(content)

    @property
    def sub_agent_names(self) -> list[str]:
        return list(self._sub_agents.keys())

    def register_sub_agent(self, name: str, config: AgentConfig):
        """Register a sub-agent configuration."""
        self._sub_agents[name] = config
        self.config.can_delegate = True
        self._register_auto_tools()
        self._rebuild_system_block()

    def _register_auto_tools(self):
        """Add/update delegation and state tools based on current config."""
        from .tools.delegation_tool import DelegationTool
        from .tools.state_tools import StateReadTool, StateWriteTool, StateListTool

        if self.config.can_delegate and self._sub_agents:
            runner = self._make_sub_agent_runner()
            self.tools["delegate_to_agent"] = DelegationTool(self._sub_agents, runner)

        if self.shared_state and self.config.state_access != "none":
            access = self.config.state_access
            agent_id = self.config.agent_id
            if "read" in access:
                self.tools["state_read"] = StateReadTool(self.shared_state)
                self.tools["state_list"] = StateListTool(self.shared_state)
            if "write" in access:
                self.tools["state_write"] = StateWriteTool(self.shared_state, agent_id)

    def _make_sub_agent_runner(self) -> SubAgentRunner:
        """Create a closure that runs sub-agents sharing this agent's infrastructure."""
        async def runner(config: AgentConfig, task: str) -> str:
            import copy

            if self._delegation_depth >= self.config.max_delegation_depth:
                return f"[Delegation blocked] Max depth ({self.config.max_delegation_depth}) reached."

            cfg = copy.deepcopy(config)
            cfg.context_config = cfg.context_config or ContextConfig.from_mode("claude_code")

            if self._delegation_depth >= self.config.max_delegation_depth - 1:
                cfg.can_delegate = False

            sub = Agent(
                llm_client=self.llm,
                config=cfg,
                workspace_dir=self.workspace_dir,
                shared_state=self.shared_state,
                logger=self.logger,
                path_guard=self.path_guard,
            )
            sub._delegation_depth = self._delegation_depth + 1

            sub.add_user_message(task)
            result = await sub.run()

            return (
                f"[Sub-agent: {cfg.name}]\n"
                f"Steps used: {sub._step_count}\n"
                f"Result:\n{result}"
            )
        return runner

    def _build_system_prompt(self, base_prompt: str) -> str:
        """Construct system prompt with workspace info, path policy, sub-agent list."""
        parts = [base_prompt]

        if "Current Workspace" not in base_prompt:
            parts.append(
                f"\n\n## Current Workspace\n"
                f"You are currently working in: `{self.workspace_dir.absolute()}`\n"
                f"All relative paths will be resolved relative to this directory."
            )

        parts.append(
            "\n\n## Path Access Policy\n"
            "You operate under file access restrictions:\n"
            "- Full read/write access within the workspace directory\n"
            "- Access outside the workspace is restricted\n"
            "- Your own source code is not accessible\n\n"
            "If a file operation is denied, inform the user about the restriction.\n"
            "Do NOT attempt to work around restrictions via bash, alternative paths, "
            "or encoded commands."
        )

        if self._sub_agents:
            agent_list = "\n".join(
                f"- **{name}**: {cfg.description}"
                for name, cfg in self._sub_agents.items()
            )
            parts.append(f"\n\n## Available Sub-Agents\n{agent_list}")

        if self.shared_state and self.shared_state.snapshot():
            state_lines = "\n".join(
                f"- {k}: {v}" for k, v in self.shared_state.snapshot().items()
            )
            parts.append(f"\n\n## Shared Data\n{state_lines}")

        return "".join(parts)

    def _rebuild_system_block(self):
        """Rebuild system prompt and update system block in BlockStore.

        Note: If called before run_stream(), the system block may not exist yet.
        That's OK -- run_stream() calls init_system() with the updated self.system_prompt.
        """
        self.system_prompt = self._build_system_prompt(self.config.system_prompt)
        system_block = self.context_manager.store.get("system")
        if system_block:
            from .context.token_counter import count_tokens
            system_block.working_content = self.system_prompt
            system_block.token_count = count_tokens(self.system_prompt)

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

        for step in range(self.config.max_steps_per_turn):
            # Refresh system prompt if SharedState has changed
            if self.shared_state and self.shared_state.snapshot():
                self._rebuild_system_block()

            # Total step limit check
            if self._step_count >= self.config.max_steps_total:
                yield StreamEvent(type=StreamEventType.ERROR,
                                  content=f"Total step limit ({self.config.max_steps_total}) reached.")
                return

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
                async for chunk in self.llm.generate_stream(messages, all_tools, model=self.config.model):
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
            self._step_count += 1

        # Max steps exceeded
        yield StreamEvent(type=StreamEventType.ERROR,
                          content=f"Task couldn't be completed after {self.config.max_steps_per_turn} steps.")

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
