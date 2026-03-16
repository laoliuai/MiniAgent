# mini_agent/session.py
"""Session API: high-level entry point for agent creation and execution."""
from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .agent import Agent
from .agent_config import AgentConfig
from .shared_state import SharedState

if TYPE_CHECKING:
    from .context.config import ContextConfig
    from .llm import LLMClient
    from .logger import AgentLogger
    from .schema import StreamEvent
    from .tools.base import Tool
    from .tools.path_guard import PathGuard


class Session:
    """Wraps Agent + SharedState with convenient factory methods."""

    def __init__(self, agent: Agent, shared_state: SharedState):
        self.agent = agent
        self.shared_state = shared_state

    # -- Delegated to Agent --

    def add_user_message(self, content: str):
        self.agent.add_user_message(content)

    async def run(self, cancel_event=None) -> str:
        return await self.agent.run(cancel_event)

    async def run_stream(self, cancel_event=None) -> AsyncGenerator[StreamEvent, None]:
        async for event in self.agent.run_stream(cancel_event):
            yield event

    # -- Status --

    def get_status(self) -> dict:
        return {
            "agent_id": self.agent.config.agent_id,
            "turn": self.agent.context_manager.current_turn,
            "context": self.agent.context_manager.get_status(),
            "shared_state_keys": list(self.shared_state.snapshot().keys()),
            "sub_agents": list(self.agent.sub_agent_names),
        }

    # -- Factory Methods --

    @classmethod
    def create(
        cls,
        llm_client: LLMClient,
        system_prompt: str = "You are a helpful assistant.",
        tools: list[Tool] | None = None,
        sub_agents: dict[str, AgentConfig] | None = None,
        context_config: ContextConfig | None = None,
        workspace_dir: str | Path = "./workspace",
        logger: AgentLogger | None = None,
        path_guard: PathGuard | None = None,
    ) -> Session:
        """Primary entry point. Single agent with optional delegation."""
        shared_state = SharedState()

        main_config = AgentConfig(
            agent_id="main",
            system_prompt=system_prompt,
            tools=tools or [],
            context_config=context_config,
            can_delegate=bool(sub_agents),
        )

        agent = Agent(
            llm_client=llm_client,
            config=main_config,
            workspace_dir=workspace_dir,
            shared_state=shared_state,
            logger=logger,
            path_guard=path_guard,
        )

        for name, sub_cfg in (sub_agents or {}).items():
            agent.register_sub_agent(name, sub_cfg)

        return cls(agent=agent, shared_state=shared_state)

    @classmethod
    def create_orchestrator(
        cls,
        llm_client: LLMClient,
        workers: dict[str, AgentConfig],
        orchestrator_prompt: str | None = None,
        context_config: ContextConfig | None = None,
        workspace_dir: str | Path = "./workspace",
        logger: AgentLogger | None = None,
        path_guard: PathGuard | None = None,
    ) -> Session:
        """Advanced entry point. Agent identity = planner/coordinator."""
        from .context.config import ContextConfig as CtxCfg

        shared_state = SharedState()

        default_prompt = (
            "You are an orchestrator agent. Your job is to:\n"
            "1. Analyze the user's task and break it into subtasks\n"
            "2. Delegate subtasks to specialized worker agents\n"
            "3. Synthesize worker results into a coherent final answer\n\n"
            "Planning: break into 2-5 subtasks. Each must be self-contained.\n"
            "Efficiency: don't delegate trivial tasks - do them yourself.\n"
            "Synthesis: after workers complete, produce unified answer."
        )

        orch_config = AgentConfig(
            agent_id="orchestrator",
            name="Orchestrator",
            system_prompt=orchestrator_prompt or default_prompt,
            context_config=context_config or CtxCfg.from_mode("full_layering"),
            can_delegate=True,
            max_delegation_depth=1,
            state_access="readwrite",
        )

        agent = Agent(
            llm_client=llm_client,
            config=orch_config,
            workspace_dir=workspace_dir,
            shared_state=shared_state,
            logger=logger,
            path_guard=path_guard,
        )

        for name, w_cfg in workers.items():
            w_cfg.context_config = w_cfg.context_config or CtxCfg.from_mode("claude_code")
            agent.register_sub_agent(name, w_cfg)

        return cls(agent=agent, shared_state=shared_state)
