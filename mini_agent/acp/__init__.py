"""ACP (Agent Client Protocol) bridge for Mini-Agent."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    AgentSideConnection,
    CancelNotification,
    InitializeRequest,
    InitializeResponse,
    NewSessionRequest,
    NewSessionResponse,
    PromptRequest,
    PromptResponse,
    session_notification,
    start_tool_call,
    stdio_streams,
    text_block,
    tool_content,
    update_agent_message,
    update_agent_thought,
    update_tool_call,
)
from pydantic import field_validator
from acp.schema import AgentCapabilities, Implementation, McpCapabilities

from mini_agent.agent import Agent
from mini_agent.cli import add_workspace_tools, initialize_base_tools
from mini_agent.config import Config
from mini_agent.llm import LLMClient
from mini_agent.retry import RetryConfig as RetryConfigBase
from mini_agent.schema import Message, StreamEventType

logger = logging.getLogger(__name__)


try:
    class InitializeRequestPatch(InitializeRequest):
        @field_validator("protocolVersion", mode="before")
        @classmethod
        def normalize_protocol_version(cls, value: Any) -> int:
            if isinstance(value, str):
                try:
                    return int(value.split(".")[0])
                except Exception:
                    return 1
            if isinstance(value, (int, float)):
                return int(value)
            return 1

    InitializeRequest = InitializeRequestPatch
    InitializeRequest.model_rebuild(force=True)
except Exception:  # pragma: no cover - defensive
    logger.debug("ACP schema patch skipped")


@dataclass
class SessionState:
    agent: Agent
    cancelled: bool = False


class MiniMaxACPAgent:
    """Minimal ACP adapter wrapping the existing Agent runtime."""

    def __init__(
        self,
        conn: AgentSideConnection,
        config: Config,
        llm: LLMClient,
        base_tools: list,
        system_prompt: str,
    ):
        self._conn = conn
        self._config = config
        self._llm = llm
        self._base_tools = base_tools
        self._system_prompt = system_prompt
        self._sessions: dict[str, SessionState] = {}

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:  # noqa: ARG002
        return InitializeResponse(
            protocolVersion=PROTOCOL_VERSION,
            agentCapabilities=AgentCapabilities(loadSession=False),
            agentInfo=Implementation(name="mini-agent", title="Mini-Agent", version="0.1.0"),
        )

    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse:
        session_id = f"sess-{len(self._sessions)}-{uuid4().hex[:8]}"
        workspace = Path(params.cwd or self._config.agent.workspace_dir).expanduser()
        if not workspace.is_absolute():
            workspace = workspace.resolve()
        tools = list(self._base_tools)
        add_workspace_tools(tools, self._config, workspace)
        agent = Agent(llm_client=self._llm, system_prompt=self._system_prompt, tools=tools, max_steps=self._config.agent.max_steps, workspace_dir=str(workspace))
        self._sessions[session_id] = SessionState(agent=agent)
        return NewSessionResponse(sessionId=session_id)

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        state = self._sessions.get(params.sessionId)
        if not state:
            # Auto-create session if not found (compatibility with clients that skip newSession)
            logger.warning(f"Session '{params.sessionId}' not found, auto-creating new session")
            new_session = await self.newSession(NewSessionRequest(cwd=None))
            state = self._sessions.get(new_session.sessionId)
            if not state:
                logger.error("Failed to auto-create session")
                return PromptResponse(stopReason="refusal")
        state.cancelled = False
        user_text = "\n".join(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "") for block in params.prompt)
        state.agent.messages.append(Message(role="user", content=user_text))
        stop_reason = await self._run_turn(state, params.sessionId)
        return PromptResponse(stopReason=stop_reason)

    async def cancel(self, params: CancelNotification) -> None:
        state = self._sessions.get(params.sessionId)
        if state:
            state.cancelled = True

    async def _run_turn(self, state: SessionState, session_id: str) -> str:
        """Run one turn by consuming agent.run_stream() events."""
        cancel_event = asyncio.Event()
        # Wire ACP cancellation to the asyncio event
        if state.cancelled:
            cancel_event.set()

        async for event in state.agent.run_stream(cancel_event):
            # Check ACP-level cancellation each iteration
            if state.cancelled and not cancel_event.is_set():
                cancel_event.set()

            match event.type:
                case StreamEventType.TEXT_DELTA:
                    await self._send(session_id, update_agent_message(text_block(event.content)))
                case StreamEventType.THINKING_DELTA:
                    await self._send(session_id, update_agent_thought(text_block(event.content)))
                case StreamEventType.TOOL_CALL_START:
                    args = event.tool_arguments or {}
                    args_preview = ", ".join(f"{k}={repr(v)[:50]}" for k, v in list(args.items())[:2]) if isinstance(args, dict) else ""
                    label = f"🔧 {event.tool_name}({args_preview})" if args_preview else f"🔧 {event.tool_name}()"
                    await self._send(session_id, start_tool_call(event.tool_call_id, label, kind="execute", raw_input=args))
                case StreamEventType.TOOL_CALL_RESULT:
                    if event.tool_result:
                        status = "completed" if event.tool_result.success else "failed"
                        prefix = "[OK]" if event.tool_result.success else "[ERROR]"
                        text = f"{prefix} {event.tool_result.content if event.tool_result.success else event.tool_result.error or 'Tool execution failed'}"
                    else:
                        status, text = "failed", "[ERROR] No result"
                    await self._send(session_id, update_tool_call(event.tool_call_id, status=status, content=[tool_content(text_block(text))], raw_output=text))
                case StreamEventType.ERROR:
                    await self._send(session_id, update_agent_message(text_block(f"Error: {event.content}")))
                    return "refusal"
                case StreamEventType.CANCELLED:
                    return "cancelled"
                case StreamEventType.DONE:
                    return "end_turn"

        return "max_turn_requests"

    async def _send(self, session_id: str, update: Any) -> None:
        await self._conn.sessionUpdate(session_notification(session_id, update))


async def run_acp_server(config: Config | None = None) -> None:
    """Run Mini-Agent as an ACP-compatible stdio server."""
    config = config or Config.load()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    base_tools, skill_loader = await initialize_base_tools(config)
    prompt_path = Config.find_config_file(config.agent.system_prompt_path)
    if prompt_path and prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = "You are a helpful AI assistant."
    if skill_loader:
        meta = skill_loader.get_skills_metadata_prompt()
        if meta:
            system_prompt = f"{system_prompt.rstrip()}\n\n{meta}"
    rcfg = config.llm.retry
    llm = LLMClient(api_key=config.llm.api_key, api_base=config.llm.api_base, model=config.llm.model, retry_config=RetryConfigBase(enabled=rcfg.enabled, max_retries=rcfg.max_retries, initial_delay=rcfg.initial_delay, max_delay=rcfg.max_delay, exponential_base=rcfg.exponential_base))
    reader, writer = await stdio_streams()
    AgentSideConnection(lambda conn: MiniMaxACPAgent(conn, config, llm, base_tools, system_prompt), writer, reader)
    logger.info("Mini-Agent ACP server running")
    await asyncio.Event().wait()


def main() -> None:
    asyncio.run(run_acp_server())


__all__ = ["MiniMaxACPAgent", "run_acp_server", "main"]
