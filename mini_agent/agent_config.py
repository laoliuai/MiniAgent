"""AgentConfig: runtime configuration describing what an agent IS."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


# Factory callback type for sub-agent execution
# (sub_config, task_message) -> result_text
SubAgentRunner = Callable[["AgentConfig", str], Awaitable[str]]


@dataclass
class AgentConfig:
    """Runtime configuration for an Agent instance.

    Describes identity, capabilities, and limits.
    Infrastructure dependencies (llm_client, workspace_dir, logger, etc.)
    are passed to the Agent constructor separately.
    """

    # Identity
    agent_id: str = "main"
    name: str = "Assistant"
    description: str = ""

    # LLM
    model: Optional[str] = None  # None = use LLMClient's default model

    # System prompt
    system_prompt: str = ""

    # Tools (Tool object list)
    tools: list[Any] = field(default_factory=list)

    # Context management
    context_config: Optional[Any] = None  # ContextConfig | None

    # Delegation
    can_delegate: bool = False
    max_delegation_depth: int = 1

    # Limits
    max_steps_per_turn: int = 30
    max_steps_total: int = 50

    # SharedState access
    state_access: str = "readwrite"  # "read" | "write" | "readwrite" | "none"
