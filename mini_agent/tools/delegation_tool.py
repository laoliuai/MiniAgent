"""DelegationTool: delegate tasks to specialized sub-agents."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Tool, ToolResult

if TYPE_CHECKING:
    from mini_agent.agent_config import AgentConfig, SubAgentRunner


class DelegationTool(Tool):
    """Delegates a task to a specialized sub-agent.

    Receives a SubAgentRunner callback to avoid circular dependency with Agent.
    The runner is created by Agent and injected at registration time.
    """

    def __init__(
        self,
        sub_agents: dict[str, AgentConfig],
        runner: SubAgentRunner,
    ):
        self._sub_agents = sub_agents
        self._runner = runner

    @property
    def name(self) -> str:
        return "delegate_to_agent"

    @property
    def description(self) -> str:
        agent_list = "\n".join(
            f"- {name}: {cfg.description}"
            for name, cfg in self._sub_agents.items()
        )
        return (
            "Delegate a task to a specialized sub-agent. "
            "The task must be self-contained — sub-agents cannot ask "
            "for clarification.\n\n"
            f"Available agents:\n{agent_list}"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "enum": list(self._sub_agents.keys()),
                    "description": "Which agent to delegate to",
                },
                "task": {
                    "type": "string",
                    "description": "Complete, self-contained task description",
                },
            },
            "required": ["agent_name", "task"],
        }

    async def execute(self, agent_name: str, task: str) -> ToolResult:  # type: ignore[override]
        config = self._sub_agents.get(agent_name)
        if not config:
            return ToolResult(
                success=False,
                content="",
                error=f"Unknown agent: {agent_name}. Available: {list(self._sub_agents.keys())}",
            )
        try:
            result = await self._runner(config, task)
            return ToolResult(success=True, content=result)
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Sub-agent '{agent_name}' failed: {e}",
            )
