# Sub-Agent Architecture Design

## Background

MiniAgent currently operates as a single-agent system. The agent framework spec (`agent-framework-final-design-v2.md`, Sections 4-6) defines a sub-agent architecture enabling task delegation, cross-agent data sharing, and orchestrator patterns. This document specifies the concrete design for implementing that capability on top of the existing async-first codebase.

### Goals

1. **Delegation** — Main agent can delegate self-contained tasks to specialized sub-agents via a `delegate_to_agent` tool
2. **SharedState** — Structured data store shared across all agents in a session
3. **Session API** — Two factory methods: `Session.create()` (delegation) and `Session.create_orchestrator()` (planning-first coordinator)
4. **CLI support** — Minimal config.yaml integration for sub-agent delegation
5. **Per-agent model** — LLMClient supports per-call model override (default: all agents share one model)

### Non-Goals

- CLI orchestrator mode (future: add `mode: orchestrator` config or `--orchestrator` flag)
- Parallel sub-agent execution (sub-agents run sequentially)
- Sub-agent asking parent for clarification (tasks must be self-contained)
- Adversarial security between agents (same trust boundary)
- EventBus for cross-agent observability (framework spec Section 2.2; future work)
- Preset agent template factory functions for SDK (framework spec Section 7; future work)
- `max_turns` field in AgentConfig (the caller drives the multi-turn loop externally; the framework spec's `max_turns` is not needed at the Agent level)

---

## 1. AgentConfig

Runtime dataclass describing what an agent IS — identity, capabilities, and limits. Replaces the current Agent constructor's scattered parameters.

**File:** `mini_agent/agent_config.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from .context.config import ContextConfig
    from .tools.base import Tool


# Factory callback type: (sub_config, task_message) -> result_text
SubAgentRunner = Callable[["AgentConfig", str], Awaitable[str]]


@dataclass
class AgentConfig:
    # Identity
    agent_id: str = "main"
    name: str = "Assistant"
    description: str = ""          # Shown to LLM for delegation target selection

    # LLM
    model: str | None = None       # None = use LLMClient's default model

    # System prompt
    system_prompt: str = ""

    # Tools (Tool object list)
    tools: list[Any] = field(default_factory=list)  # list[Tool], Any to avoid import

    # Context management
    context_config: Optional[Any] = None  # ContextConfig | None, None = hybrid default

    # Delegation
    can_delegate: bool = False
    max_delegation_depth: int = 1   # 1 = can delegate, but subs cannot re-delegate

    # Limits
    max_steps_per_turn: int = 30    # Max tool calls within a single turn
    max_steps_total: int = 50       # Total step budget (meaningful for sub-agents)

    # SharedState access
    state_access: str = "readwrite"  # "read" | "write" | "readwrite" | "none"
```

### Parameter Split Rationale

AgentConfig holds what differs per agent (identity, capabilities, limits). Infrastructure dependencies are passed to Agent constructor directly:

| In AgentConfig | In Agent constructor |
|---|---|
| agent_id, name, description | llm_client (shared service) |
| model, system_prompt | workspace_dir (environment) |
| tools, context_config | shared_state (shared resource) |
| can_delegate, max_delegation_depth | logger (infrastructure) |
| max_steps_per_turn, max_steps_total | path_guard (security) |
| state_access | |

### Config Rename

The existing Pydantic `AgentConfig` in `config.py` (YAML config parsing) is renamed to `AgentSettings` to avoid naming collision:

```python
# config.py
class AgentSettings(BaseModel):
    """YAML config section for agent settings."""
    max_steps_per_turn: int = 30
    max_steps_total: int = 50
    workspace_dir: str = "./workspace"
    system_prompt_path: str = "system_prompt.md"
```

**Migration note:** The existing `AgentConfig.max_steps: int = 50` field is split into two fields. `Config.from_yaml()` must be updated to parse both `max_steps_per_turn` and `max_steps_total` from the YAML `agent` section, replacing the old `max_steps` parsing. For backward compatibility, if only `max_steps` is present in YAML, it maps to `max_steps_per_turn` with `max_steps_total` defaulting to 50.

---

## 2. Agent Constructor Refactor

**File:** `mini_agent/agent.py`

### New Constructor Signature

```python
class Agent:
    def __init__(
        self,
        llm_client: LLMClient,
        config: AgentConfig,
        workspace_dir: Path | str = "./workspace",
        shared_state: SharedState | None = None,
        logger: AgentLogger | None = None,
        path_guard: PathGuard | None = None,
    ):
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
        self._step_count: int = 0  # Total steps across all turns (for sub-agent limit)

        # Workspace setup
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # System prompt construction (workspace info + path policy + sub-agent descriptions)
        self.system_prompt = self._build_system_prompt(config.system_prompt)

        # Logger
        self.logger = logger or AgentLogger(...)

        # ContextManager
        ctx = config.context_config or ContextConfig()
        self.context_manager = ContextManager(ctx, self.llm, self.tools)

        # Backward compatibility
        self.messages: list[Message] = []
```

### Public Accessors

```python
@property
def sub_agent_names(self) -> list[str]:
    """Public accessor for registered sub-agent names."""
    return list(self._sub_agents.keys())
```

### register_sub_agent

```python
def register_sub_agent(self, name: str, config: AgentConfig):
    """Register a sub-agent configuration. Rebuilds delegation tool and system prompt."""
    self._sub_agents[name] = config
    self.config.can_delegate = True
    self._register_auto_tools()
    self._rebuild_system_block()  # Update system prompt with new sub-agent descriptions
```

### _register_auto_tools

Called after sub-agent registration and during init. Adds/updates delegation and state tools.

```python
def _register_auto_tools(self):
    # Delegation tool
    if self.config.can_delegate and self._sub_agents:
        runner = self._make_sub_agent_runner()
        self.tools["delegate_to_agent"] = DelegationTool(self._sub_agents, runner)

    # State tools
    if self.shared_state and self.config.state_access != "none":
        access = self.config.state_access
        agent_id = self.config.agent_id
        if "read" in access:
            self.tools["state_read"] = StateReadTool(self.shared_state)
            self.tools["state_list"] = StateListTool(self.shared_state)
        if "write" in access:
            self.tools["state_write"] = StateWriteTool(self.shared_state, agent_id)
```

### run_stream() Step Limit Changes

The loop condition changes from `range(self.max_steps)` to dual limits:

```python
async def run_stream(self, cancel_event=None):
    ...
    for step in range(self.config.max_steps_per_turn):
        # Also check total steps (meaningful for sub-agents)
        if self._step_count >= self.config.max_steps_total:
            yield StreamEvent(type=StreamEventType.ERROR,
                              content=f"Total step limit ({self.config.max_steps_total}) reached.")
            return

        # ... existing LLM call + tool execution ...

        # Increment total step counter
        self._step_count += 1
```

### _build_system_prompt

Constructs system prompt by appending workspace info, path policy, sub-agent descriptions, and SharedState snapshot:

```python
def _build_system_prompt(self, base_prompt: str) -> str:
    parts = [base_prompt]

    # Workspace info (existing logic)
    if "Current Workspace" not in base_prompt:
        parts.append(f"\n\n## Current Workspace\n...")

    # Path access policy (existing logic)
    parts.append("\n\n## Path Access Policy\n...")

    # Sub-agent descriptions (new)
    if self._sub_agents:
        agent_list = "\n".join(
            f"- **{name}**: {cfg.description}" for name, cfg in self._sub_agents.items()
        )
        parts.append(f"\n\n## Available Sub-Agents\n{agent_list}")

    # SharedState snapshot (new)
    if self.shared_state and self.shared_state.snapshot():
        state_lines = "\n".join(
            f"- {k}: {v}" for k, v in self.shared_state.snapshot().items()
        )
        parts.append(f"\n\n## Shared Data\n{state_lines}")

    return "".join(parts)
```

### _rebuild_system_block

Called after `register_sub_agent()` and before each turn (when SharedState changes). Regenerates the system prompt and updates the system block in ContextManager's BlockStore.

```python
def _rebuild_system_block(self):
    """Rebuild system prompt and update system block in BlockStore."""
    self.system_prompt = self._build_system_prompt(self.config.system_prompt)
    system_block = self.context_manager.store.get("system")
    if system_block:
        system_block.working_content = self.system_prompt
        system_block.token_count = count_tokens(self.system_prompt)
```

### Tool Sharing: Stateful Tool Considerations

When sub-agents reference tools from the parent's tool registry (via CLI config), most tools are effectively stateless per-call (ReadTool, WriteTool, EditTool, GrepTool) and safe to share.

**Stateful tools** (BashTool's background process registry, SessionNoteTool's memory file, TodoTool's task list) could interfere if shared between concurrent agents. Since sub-agents run **sequentially** (not in parallel), sharing is safe in practice. However, the spec notes this as a known limitation: if parallel sub-agent execution is added in the future, stateful tools must be cloned per sub-agent instance.

---

## 3. DelegationTool

**File:** `mini_agent/tools/delegation_tool.py`

A standard Tool subclass. Receives a `SubAgentRunner` callback to avoid circular dependency with Agent.

```python
class DelegationTool(Tool):
    """Delegates a task to a specialized sub-agent."""

    def __init__(self, sub_agents: dict[str, AgentConfig], runner: SubAgentRunner):
        self._sub_agents = sub_agents
        self._runner = runner

    @property
    def name(self) -> str:
        return "delegate_to_agent"

    @property
    def description(self) -> str:
        agent_list = "\n".join(
            f"- {name}: {cfg.description}" for name, cfg in self._sub_agents.items()
        )
        return (
            "Delegate a task to a specialized sub-agent. "
            "The task must be self-contained — sub-agents cannot ask for clarification.\n\n"
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

    async def execute(self, agent_name: str, task: str) -> ToolResult:
        config = self._sub_agents.get(agent_name)
        if not config:
            return ToolResult(
                success=False,
                error=f"Unknown agent: {agent_name}. Available: {list(self._sub_agents.keys())}",
            )
        try:
            result = await self._runner(config, task)
            return ToolResult(success=True, content=result)
        except Exception as e:
            return ToolResult(success=False, error=f"Sub-agent '{agent_name}' failed: {e}")
```

### Circular Dependency Solution

DelegationTool does NOT import Agent. Instead, Agent creates a runner closure and injects it:

```python
# In Agent
def _make_sub_agent_runner(self) -> SubAgentRunner:
    async def runner(config: AgentConfig, task: str) -> str:
        import copy

        # Depth check
        if self._delegation_depth >= self.config.max_delegation_depth:
            return f"[Delegation blocked] Max depth ({self.config.max_delegation_depth}) reached."

        # Deep copy config to prevent mutation
        cfg = copy.deepcopy(config)
        # Sub-agents default to claude_code mode (single task, no layering needed)
        cfg.context_config = cfg.context_config or ContextConfig.from_mode("claude_code")

        # Prevent further delegation if at depth limit
        if self._delegation_depth >= self.config.max_delegation_depth - 1:
            cfg.can_delegate = False

        # Create sub-agent sharing infrastructure
        sub = Agent(
            llm_client=self.llm,
            config=cfg,
            workspace_dir=self.workspace_dir,
            shared_state=self.shared_state,
            logger=self.logger,
            path_guard=self.path_guard,
        )
        sub._delegation_depth = self._delegation_depth + 1

        # Execute: sub-agent has exactly 1 turn
        sub.add_user_message(task)
        result = await sub.run()

        return (
            f"[Sub-agent: {cfg.name}]\n"
            f"Steps used: {sub._step_count}\n"
            f"Result:\n{result}"
        )
    return runner
```

### Sub-Agent Execution Model

```
Parent Agent                          Sub-Agent
    |                                     |
    |--- delegate_to_agent -------------->|
    |    (agent_name, task)               |
    |                                     |-- add_user_message(task)
    |                                     |-- run()
    |                                     |     +- step 1: tool call
    |                                     |     +- step 2: tool call
    |                                     |     +- ... (max_steps_total limit)
    |                                     |-- return final text
    |<-- ToolResult(content=result) ------|
    |                                     | (sub-agent destroyed, context freed)
```

Key constraints:
- Sub-agent **cannot** ask parent for clarification — task must be self-contained
- Sub-agent has exactly **1 turn** (1 user message), limited by `max_steps_total`
- Sub-agent is destroyed after execution — context is not preserved

---

## 4. SharedState

**File:** `mini_agent/shared_state.py`

Cross-agent structured data store. Uses `asyncio.Lock` (matching async-first architecture).

```python
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class StateEntry:
    key: str
    value: Any
    written_by: str        # agent_id
    written_at: datetime
    schema_hint: str = ""  # Type hint for LLM: "DataFrame(200 rows, 5 cols)"
    ttl_turns: Optional[int] = None  # Auto-expire by parent agent turns


class SharedState:
    def __init__(self):
        self._store: dict[str, StateEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            e = self._store.get(key)
            return e.value if e else None

    async def set(self, key: str, value: Any, agent_id: str,
                  schema_hint: str = "", ttl_turns: int | None = None):
        async with self._lock:
            self._store[key] = StateEntry(
                key=key, value=value, written_by=agent_id,
                written_at=datetime.now(), schema_hint=schema_hint,
                ttl_turns=ttl_turns,
            )

    async def keys(self, prefix: str = "") -> list[str]:
        async with self._lock:
            return [k for k in self._store if k.startswith(prefix)]

    async def delete(self, key: str) -> bool:
        async with self._lock:
            return self._store.pop(key, None) is not None

    def snapshot(self) -> dict[str, str]:
        """Synchronous. Token-efficient summary for system prompt injection.
        Keys + schema hints only, no actual values.
        Safe without lock: read-only dict iteration in single-threaded async loop."""
        return {
            k: f"{e.schema_hint} (by {e.written_by})"
            for k, e in self._store.items()
        }
```

### State Tools

**File:** `mini_agent/tools/state_tools.py`

Three Tool subclasses, registered based on `AgentConfig.state_access`:

```python
class StateReadTool(Tool):
    """Read a value from shared state."""
    name = "state_read"
    # execute(key: str) -> ToolResult

class StateWriteTool(Tool):
    """Write a value to shared state."""
    name = "state_write"
    # execute(key: str, value: str, schema_hint: str = "") -> ToolResult
    # Note: value is stored as string (LLM can only produce text).
    # Programmatic callers can store any type via SharedState.set() directly.

class StateListTool(Tool):
    """List keys in shared state."""
    name = "state_list"
    # execute(prefix: str = "") -> ToolResult
```

### Access Control Matrix

| state_access | state_read | state_list | state_write |
|---|---|---|---|
| `"readwrite"` | yes | yes | yes |
| `"read"` | yes | yes | no |
| `"write"` | no | no | yes |
| `"none"` | no | no | no |

---

## 5. Session API

**File:** `mini_agent/session.py`

Thin wrapper over Agent + SharedState with two factory methods.

```python
class Session:
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
            "sub_agents": list(self.agent.sub_agent_names),  # Public accessor
        }
```

### Factory 1: Session.create()

Single agent with optional delegation. Covers 95% of use cases.

```python
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
) -> "Session":
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
```

### Factory 2: Session.create_orchestrator()

Planning-first coordinator. Agent identity = orchestrator.

```python
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
) -> "Session":
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
        context_config=context_config or ContextConfig.from_mode("full_layering"),
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
        w_cfg.context_config = w_cfg.context_config or ContextConfig.from_mode("claude_code")
        agent.register_sub_agent(name, w_cfg)

    return cls(agent=agent, shared_state=shared_state)
```

### SDK Usage Examples

```python
# Simple: no delegation
session = Session.create(llm_client=client, tools=[bash, read, write])
session.add_user_message("Analyze sales.csv")
result = await session.run()

# With delegation
session = Session.create(
    llm_client=client,
    tools=[read, write],
    sub_agents={
        "coder": AgentConfig(name="Coder", description="Write code", tools=[bash, read, write, edit]),
        "analyst": AgentConfig(name="Analyst", description="Data analysis", tools=[bash, read]),
    },
)
session.add_user_message("Research industry data, then write an analysis script")
result = await session.run()

# Orchestrator
session = Session.create_orchestrator(
    llm_client=client,
    workers={
        "coder": AgentConfig(name="Coder", description="Write code", tools=[bash, read, write]),
        "researcher": AgentConfig(name="Researcher", description="Search and analyze", tools=[bash, read]),
    },
)
session.add_user_message("Create a competitive analysis report with data backing")
result = await session.run()
```

---

## 6. LLMClient Per-Call Model Override

**Files:** `mini_agent/llm/base.py`, `mini_agent/llm/llm_wrapper.py`, `anthropic_client.py`, `openai_client.py`

Add optional `model` parameter to both `generate()` and `generate_stream()` across the full interface chain:

```python
# LLMClientBase (base.py) - abstract method signatures
async def generate_stream(self, messages, tools, model: str | None = None) -> AsyncGenerator[...]:
    ...
async def generate(self, messages, tools, model: str | None = None) -> LLMResponse:
    ...  # Forwards model to generate_stream()

# LLMClient (llm_wrapper.py) - wrapper, forwards model param
async def generate_stream(self, messages, tools, model: str | None = None):
    ...

# AnthropicClient / OpenAIClient
async def generate_stream(self, messages, tools, model: str | None = None):
    use_model = model or self.model
    response = await self.client.messages.create(model=use_model, ...)
```

Agent passes `self.config.model` to LLM:

```python
# Agent.run_stream()
async for chunk in self.llm.generate_stream(
    messages, all_tools, model=self.config.model
):
```

Default `AgentConfig.model = None` means all agents share the LLMClient's default model. Only explicit override changes it.

---

## 7. CLI Adaptation

### config.py Changes

```python
# Rename existing AgentConfig -> AgentSettings
class AgentSettings(BaseModel):
    """YAML config section."""
    max_steps_per_turn: int = 30
    max_steps_total: int = 50
    workspace_dir: str = "./workspace"
    system_prompt_path: str = "system_prompt.md"

# New: sub-agent YAML entry
class SubAgentEntry(BaseModel):
    description: str = ""
    system_prompt: str = ""
    system_prompt_path: str = ""   # Alternative to system_prompt
    model: str | None = None
    tools: list[str] = []          # Tool name references: ["bash", "read", "write", ...]

class Config(BaseModel):
    llm: LLMConfig
    agent: AgentSettings           # Renamed
    tools: ToolsConfig
    logging: LoggingConfig
    context: ContextConfig
    sub_agents: dict[str, SubAgentEntry] = {}   # New
```

### config.yaml Example

```yaml
agent:
  max_steps_per_turn: 30
  max_steps_total: 50
  workspace_dir: "./workspace"

sub_agents:
  coder:
    description: "Write and debug code"
    system_prompt: "You are a coding assistant. Write clean, tested code."
    tools: ["bash", "read", "write", "edit", "grep"]
  researcher:
    description: "Search and analyze information"
    system_prompt: "You are a research assistant. Find and summarize information."
    tools: ["bash", "read", "grep"]
```

### CLI run_agent() Flow

```python
async def run_agent():
    config = Config.load()
    llm_client = create_llm_client(config)

    # 1. Build full tool set (dict[str, Tool])
    all_tools = initialize_tools(config, workspace_dir)

    # 2. Main agent tools = all tools
    main_tools = list(all_tools.values())

    # 3. Build sub-agent configs from YAML
    sub_agents = {}
    for name, entry in config.sub_agents.items():
        sub_tools = [all_tools[t] for t in entry.tools if t in all_tools]
        sub_prompt = entry.system_prompt or load_prompt(entry.system_prompt_path)
        sub_agents[name] = AgentConfig(
            agent_id=name,
            name=name,
            description=entry.description,
            model=entry.model,
            system_prompt=sub_prompt,
            tools=sub_tools,
            max_steps_total=config.agent.max_steps_total,
        )

    # 4. Create Session (replaces direct Agent construction)
    session = Session.create(
        llm_client=llm_client,
        system_prompt=system_prompt,
        tools=main_tools,
        sub_agents=sub_agents or None,
        context_config=config.context,
        workspace_dir=workspace_dir,
        logger=logger,
        path_guard=path_guard,
    )

    # 5. Interactive loop (session.agent replaces old agent reference)
    session.add_user_message(user_input)
    async for event in session.run_stream():
        render(event)
```

### Tool Name Reference

Sub-agents reference tools by string name in config.yaml. CLI resolves them against the initialized tool registry:

| Name | Tool Class |
|---|---|
| `"bash"` | BashTool |
| `"read"` | ReadTool |
| `"write"` | WriteTool |
| `"edit"` | EditTool |
| `"grep"` | GrepTool |
| `"note"` | SessionNoteTool |
| `"todo"` | TodoTool |

MCP tools and Skill tools can also be referenced by their registered names.

---

## 8. Error Handling

All sub-agent errors are captured by DelegationTool and returned as ToolResult. The parent agent never crashes due to sub-agent failure.

| Scenario | Handling | Parent sees |
|---|---|---|
| Unknown agent_name | DelegationTool returns error | `ToolResult(success=False)` |
| Depth limit exceeded | Runner returns descriptive text | `ToolResult(success=True, content="[blocked]...")` |
| Sub-agent tool error | Sub-agent's run() handles normally | Error info in result text |
| Sub-agent max_steps | Sub-agent run() returns truncation msg | `ToolResult(success=True, content="[max steps]...")` |
| Sub-agent LLM failure | Retry in sub-agent, then bubble up | `ToolResult(success=False, error="...")` |
| Uncaught exception | Runner try/except catches all | `ToolResult(success=False, error="...")` |

SharedState errors: State tool execute() wraps all exceptions into `ToolResult(success=False)`.

---

## 9. File Structure

### New Files

| File | Content |
|---|---|
| `mini_agent/agent_config.py` | AgentConfig dataclass, SubAgentRunner type alias |
| `mini_agent/shared_state.py` | SharedState, StateEntry |
| `mini_agent/session.py` | Session API (create, create_orchestrator) |
| `mini_agent/tools/delegation_tool.py` | DelegationTool |
| `mini_agent/tools/state_tools.py` | StateReadTool, StateWriteTool, StateListTool |

### Modified Files

| File | Changes |
|---|---|
| `mini_agent/agent.py` | Constructor takes AgentConfig, register_sub_agent, _register_auto_tools, _make_sub_agent_runner, dual step limits in run_stream |
| `mini_agent/config.py` | AgentConfig -> AgentSettings rename, SubAgentEntry, sub_agents parsing |
| `mini_agent/cli.py` | Use Session.create(), build sub-agent configs from YAML |
| `mini_agent/llm/base.py` | Abstract generate/generate_stream add model parameter |
| `mini_agent/llm/llm_wrapper.py` | generate/generate_stream forward model parameter |
| `mini_agent/llm/anthropic_client.py` | generate_stream uses model override |
| `mini_agent/llm/openai_client.py` | generate_stream uses model override |

---

## 10. Testing Strategy

### Unit Tests (mock dependencies)

| Test File | Coverage |
|---|---|
| `tests/test_agent_config.py` | AgentConfig defaults, field validation |
| `tests/test_shared_state.py` | get/set/keys/delete/snapshot, concurrent safety |
| `tests/test_delegation_tool.py` | Schema generation, unknown agent, normal delegation (mock runner), exception handling |
| `tests/test_state_tools.py` | read/write/list scenarios, access control |
| `tests/test_session.py` | create() and create_orchestrator() factories, parameter passing |

### Integration Tests (mock LLM)

| Test File | Coverage |
|---|---|
| `tests/test_sub_agent_integration.py` | Full delegation flow: parent calls delegate_to_agent -> sub-agent executes -> result returns |
| | SharedState read/write across agents |
| | Depth limiting enforcement |
| | Per-agent model override |
