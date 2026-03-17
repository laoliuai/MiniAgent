# Workflow Feature — Implementation Design Spec

> **Reference**: `docs/superpowers/specs/workflow-design-document.md`
> **Scope**: Structured multi-stage human-AI collaboration
> **Principle**: Workflow is a peer to Session, not a replacement. Zero impact on existing CLI/SDK.
> **Implementation approach**: Adapt design doc to fit existing codebase architecture (Agent/Session/ContextManager/Tool patterns).

---

## 1. Problem Statement

The existing framework handles free-form agent conversation well. However, structured human-AI collaboration — where multiple stages have distinct AI roles, human gates between stages, and artifacts flowing from one stage to the next — cannot be expressed with the current `Session.create()` API.

Use cases: product design (idea → brainstorm → draft → review → finalize), code review (submit → review → fix), report writing, onboarding flows.

---

## 2. Scope

### In scope
- Workflow core engine (`workflow.py`) — Stage, WorkflowStatus, WorkflowResponse, StageCompleteTool, Workflow class
- YAML loading and validation (`workflow_loader.py`) — parse, validate, discover, build Workflow from YAML
- CLI integration — `mini-agent workflow <name>` and `mini-agent workflows` commands
- 2 built-in workflows — `product_design.yaml` (6 stages) and `code_review.yaml` (3 stages)
- Custom workflow guide documentation

### Out of scope
- Web UI integration (future iteration)
- Workflow persistence/resumption (exit = state lost)
- Per-stage sub-agent activation (current Agent doesn't support selective activation)

---

## 3. Architecture

### Layer diagram

```
┌──────────────────────────────────────────────────────┐
│                     Agent                             │
│   run_stream() + tools + context pipeline             │
│                  (UNCHANGED)                          │
└──────────────────┬───────────────────────────────────┘
                   │
          ┌────────┴────────┐
          │                 │
┌─────────┴──────┐  ┌──────┴──────────┐
│    Session     │  │    Workflow      │
│  (UNCHANGED)   │  │    (NEW)        │
└────────────────┘  └──────────────────┘
                           │
                    ┌──────┴───────┐
                    │ workflow_    │
                    │ loader.py   │
                    │ (NEW)       │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         CLI commands   YAML files   Tool mapping
```

### Key boundary

Workflow is Session's peer. Both consume Agent. They never coexist in the same interaction. Agent is completely unaware it's inside a Workflow — Workflow communicates through:
- Swapping the system block content (same pattern as `_rebuild_system_block`)
- Injecting `StageCompleteTool` into the tool list (a regular Tool subclass)
- Checking a flag after `run_stream()` completes

### Files added/modified

| File | Status | Purpose |
|---|---|---|
| `mini_agent/workflow.py` | **NEW** | Workflow, Stage, WorkflowStatus, WorkflowResponse, StageCompleteTool |
| `mini_agent/workflow_loader.py` | **NEW** | YAML parsing, validation, discovery, `build_workflow()` |
| `mini_agent/workflows/product_design.yaml` | **NEW** | Built-in product design workflow (6 stages) |
| `mini_agent/workflows/code_review.yaml` | **NEW** | Built-in code review workflow (3 stages) |
| `mini_agent/cli.py` | **MODIFIED** | Add `workflow` and `workflows` subcommands |
| `mini_agent/agent.py` | **UNCHANGED** | — |
| `mini_agent/session.py` | **UNCHANGED** | — |
| `mini_agent/context/` | **UNCHANGED** | Used by Workflow but not modified |

---

## 4. Data Model

### 4.1 Stage

```python
@dataclass
class Stage:
    name: str                          # "intake", "brainstorm", "deep_dive"
    goal: str                          # Required. Shown to user: "理解原始需求"
    system_prompt: str                 # AI personality for this stage

    gate: str = "human"                # "human" | "auto" | "none"
    max_turns: int = 100               # Safety limit. Default 100 (raised from reference design's 20)
                                       # to support complex multi-user product discussions.
                                       # Individual workflows can lower this per-stage.
    tools: list[str] | None = None     # Override tools (None = inherit defaults)
    input_artifacts: list[str] = field(default_factory=list)
    output_artifacts: list[str] = field(default_factory=list)
```

### 4.2 WorkflowStatus

```python
@dataclass
class WorkflowStatus:
    stage_name: str
    stage_index: int                   # 0-based
    total_stages: int
    stage_goal: str
    gate_pending: bool
    turns_in_stage: int
    progress: float                    # 0.0 → 1.0
    is_complete: bool
    artifacts: dict[str, str]          # SharedState snapshot
    all_stages: list[dict]             # [{"name": ..., "goal": ..., "status": "completed|active|pending"}]
```

### 4.3 WorkflowResponse

```python
@dataclass
class WorkflowResponse:
    reply: str                         # Agent's text response
    status: WorkflowStatus
    events: list[StreamEvent]          # Full stream events for CLI rendering
```

`events` is included so CLI can reuse existing StreamEvent rendering logic (thinking, tool calls, text deltas).

---

## 5. StageCompleteTool

A regular Tool subclass. Only injected by Workflow — never registered in normal Session.

```python
class StageCompleteTool(Tool):
    """Uses @property pattern consistent with all other Tool subclasses in the codebase."""

    @property
    def name(self) -> str:
        return "stage_complete"

    @property
    def description(self) -> str:
        return (
            "Signal that the current stage's goal is achieved and you are ready "
            "for human review. Call this when you have produced all expected outputs. "
            "Do NOT call prematurely — the human will review and may ask for revisions."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was accomplished in this stage",
                },
            },
            "required": ["summary"],
        }

    def __init__(self):
        self._signaled = False
        self._summary = ""

    async def execute(self, summary: str) -> ToolResult:
        self._signaled = True
        self._summary = summary
        return ToolResult(success=True, content="Stage completion noted. Waiting for review.")

    def reset(self):
        self._signaled = False
        self._summary = ""
```

Integration mechanism:
1. `Workflow.create()` creates `StageCompleteTool` and adds it to Agent's tool list
2. Agent calls it like any other tool through normal dispatch
3. After `run_stream()` completes, Workflow checks `_signaled` flag
4. If signaled → trigger gate logic based on stage's gate type

---

## 6. Workflow Class

All public Workflow methods are `async def` — the codebase is async-first, and Workflow needs to `async for` over `agent.run_stream()` and `await` SharedState operations.

### 6.1 Constructor (internal)

```python
class Workflow:
    def __init__(self, agent, stages, shared_state, default_tools, stage_complete_tool,
                 output_dir=None):
        self._agent = agent
        self._stages = stages
        self._shared_state = shared_state
        self._default_tools = default_tools        # list[Tool]
        self._stage_complete_tool = stage_complete_tool
        self._current_stage_index = 0
        self._gate_pending = False
        self._stage_turn_count = 0
        self._stage_start_turn = 0                 # ContextManager turn at stage start
        self._last_stage_summary = ""
        self._is_complete = False
        self._output_dir = output_dir              # Resolved absolute path or None
```

### 6.2 Factory: Workflow.create()

```python
@classmethod
async def create(
    cls,
    llm_client: LLMClient,
    stages: list[Stage],
    tools: list[Tool] | None = None,
    context_mode: str = "hybrid",
    token_budget: int = 100_000,
    workspace_dir: str | Path = "./workspace",
    output_dir: str | None = None,
    logger: AgentLogger | None = None,
    path_guard: PathGuard | None = None,
) -> "Workflow":
```

Steps:
1. Create `SharedState`
2. Create `StageCompleteTool`
3. Resolve first stage's tools (or use defaults) + append `StageCompleteTool`
4. Build `AgentConfig` with first stage's system_prompt
5. Create `Agent` instance
6. If `output_dir` specified, resolve against `workspace_dir` and store for prompt injection
7. Return `Workflow` instance

### 6.3 async chat(message) -> WorkflowResponse

```
1. If gate_pending → return prompt to approve/reject, no agent call
2. Reset stage_complete_tool flag
3. agent.add_user_message(message)
4. Collect all events from agent.run_stream() via `async for`
5. Extract final reply text from DONE event
6. Increment _stage_turn_count
7. Check stage_complete_tool._signaled:
   - gate == "human" → set gate_pending = True
   - gate == "auto"  → await _auto_advance() (compress + advance)
   - gate == "none"  → set _is_complete = True (workflow done)
8. Check turn limit (gate != "none"): force gate_pending if exceeded
   Check turn limit (gate == "none"): set _is_complete = True
9. Return WorkflowResponse(reply, status, events)
```

### 6.4 async approve_gate(feedback="") -> WorkflowStatus

```
1. If not gate_pending → return current status (no-op)
2. Compress current stage via context_compress_to_conclusion
   - stage_turn_ids = list(range(_stage_start_turn, agent.context_manager.current_turn + 1))
3. Store stage summary in SharedState via await shared_state.set():
   key = "_stage/{name}/summary", value = summary text, agent_id = "workflow"
4. Advance stage index, reset _stage_turn_count = 0,
   set _stage_start_turn = agent.context_manager.current_turn + 1
5. If no more stages → set _is_complete = True, return
6. Apply new stage config (await _apply_stage_config)
7. If feedback provided → inject as first message of new stage
   (agent.add_user_message + await run to get response)
8. Return new status
```

### 6.5 async reject_gate(reason) -> WorkflowStatus

`reject_gate()` only injects the rejection reason as a user message and clears the gate. The CLI should follow up with a normal `chat()` call to drive the agent's response.

```
1. If not gate_pending → return current status (no-op)
2. Clear gate_pending
3. Inject rejection reason as user message (agent.add_user_message)
4. Return status (same stage, ready for more chat)
```

Note: `reject_gate()` returns `WorkflowStatus`, not `WorkflowResponse`. The agent does not run immediately — the CLI calls `chat()` next to get the agent's revised response with full events.

### 6.6 async _apply_stage_config(stage)

```
1. Build new system prompt:
   - stage.system_prompt
   - Stage completion guidance (stage name, goal, when to call stage_complete)
   - Context editing guidance (if enabled)
   - Input artifacts context: use `await shared_state.get(key)` to read actual
     artifact values (not snapshot() which only returns schema hints)
   - Output directory path (if configured)
2. Update Agent for new stage:
   a. Set agent.config.system_prompt = new_prompt
      (This ensures _rebuild_system_block() in run_stream() produces the correct
       content, since run_stream() rebuilds the system block when SharedState changes)
   b. Also directly update the system block if it exists:
      system_block.working_content = built_prompt (with workspace info appended)
      system_block.token_count = count_tokens(...)
3. Update Agent's tool dict directly (agent.tools is the runtime dispatch dict):
   - stage.tools (filtered from default_tools by name) or default_tools
   - Always include stage_complete_tool
   - self._agent.tools = {t.name: t for t in new_tool_list}
   (Do NOT just update agent.config.tools — the agent.tools dict is built once
    at construction and is the actual dispatch registry used by run_stream())
```

### 6.7 _compress_completed_stage()

Uses existing `context_compress_to_conclusion` from ContextEditor:

```python
conclusion = (
    f"[Stage \"{stage.name}\" completed]\n"
    f"Goal: {stage.goal}\n"
    f"Summary: {self._last_stage_summary}\n"
    f"Artifacts: {', '.join(stage.output_artifacts) or 'none'}"
)
self._agent.context_manager.editor.execute("context_compress_to_conclusion", {
    "turn_ids": stage_turn_ids,
    "conclusion": conclusion,
})
```

### 6.8 get_status() -> WorkflowStatus

Pure read. Builds status from current state. No side effects.

---

## 7. YAML Workflow Definition Format

### 7.1 Schema

```yaml
# Required
name: string               # Unique identifier, used in CLI
stages: list               # Ordered list, at least 1

# Optional
description: string        # Human-readable description
version: string
output_dir: string         # Default output path, relative to workspace_dir

# Optional defaults
defaults:
  context_mode: string     # "claude_code" | "hybrid" | "full_layering"
  token_budget: int
  tools: list[string]      # Base tools available in all stages

# Stage definition
stages:
  - name: string           # Required
    goal: string           # Required. Shown to user in CLI display and prompt injection.
    prompt: string         # Required. System prompt for this stage.
    gate: string           # "human" (default) | "auto" | "none"
    max_turns: int         # Default: 100
    tools: list[string]    # Override default tools. Null = use defaults.
    input_artifacts: list[string]
    output_artifacts: list[string]
```

### 7.2 Validation rules

| Rule | Severity |
|---|---|
| `name` present and non-empty | error |
| `stages` non-empty list | error |
| Each stage has `name`, `goal`, and `prompt` | error |
| Stage names unique within workflow | error |
| `gate` is one of: human, auto, none | error |
| Last stage should be `gate: "none"` | warning |
| `input_artifacts` keys match earlier `output_artifacts` | warning |

Note: `goal` is required (not optional) because it is used in CLI display, prompt injection, and compression summaries. An empty goal degrades user experience significantly.

---

## 8. Workflow Loading (workflow_loader.py)

### 8.1 discover_workflows(directory) -> dict[str, dict]

Scan directory for `*.yaml` files, parse each, return `{name: definition}`. Invalid files produce warnings.

### 8.2 load_workflow_definition(filepath) -> dict

Parse YAML, validate against schema rules. Raise `ValueError` on errors. Log warnings for non-fatal issues.

### 8.3 build_workflow(llm_client, definition, tools, ...) -> Workflow

Convert parsed YAML dict into a Workflow instance:

1. Parse each stage dict → `Stage` dataclass
2. Handle `tools` field: match YAML string names against the provided `tools: list[Tool]` by `tool.name`
3. Read `defaults` for context_mode, token_budget
4. Resolve `output_dir` (CLI override > YAML > default `./workflow_output`)
5. Call `Workflow.create()`

The `tools` parameter is the full list of Tool objects built by CLI (same instances used in normal chat mode). Loader only does name → object lookup.

### 8.4 Workflow discovery path priority

1. `--dir` CLI argument (explicit)
2. `./workflows/` in current working directory
3. `mini_agent/workflows/` package built-in directory

---

## 9. CLI Integration

### 9.1 New commands

```bash
mini-agent workflows [--dir DIR]        # List available workflows
mini-agent workflow <name> [--dir DIR] [--output-dir DIR]  # Run workflow
```

Existing default behavior (no subcommand = interactive chat) is unchanged.

### 9.2 Interactive experience

```
════════════════════════════════════════════
  Workflow: product_design
  从原始需求到最终产品设计和技术设计文档
  Stages: intake → brainstorm → deep_dive → draft → review → finalize
════════════════════════════════════════════

┌─ Stage 1/6: intake
│  Goal: 理解原始需求，形成结构化的问题定义
│  Progress: [●○○○○○]
└──────────────────────────

You> ...
Agent> ...

╔══════════════════════════════╗
║  Stage complete — 等待审核    ║
║                              ║
║  [A] Approve 通过并进入下一阶段 ║
║  [R] Reject 退回修改          ║
║  [S] Status 查看当前状态      ║
║  [Q] Quit 退出               ║
╚══════════════════════════════╝
Decision>
```

### 9.3 Implementation

- **Tool reuse**: Workflow mode reuses `initialize_base_tools()` — same Tool objects as chat mode
- **Rendering reuse**: `WorkflowResponse.events` contains `StreamEvent` list, CLI reuses existing rendering logic
- **Gate interaction**: When `status.gate_pending == True`, enter dedicated approve/reject prompt loop
- **Stage transition display**: Print stage header on each new stage (name, goal, progress bar)
- **Special commands**: `status` prints current state, `quit` exits
- **Entry point**: The `workflow` and `workflows` subcommands are added to the existing `subparsers` in `parse_args()`. No new entry points in `pyproject.toml` are needed — the existing `mini-agent` entry point handles everything

---

## 10. Built-in Workflows

### 10.1 product_design.yaml (6 stages)

| # | Stage | AI Role | Gate | Input Artifacts | Output Artifacts |
|---|---|---|---|---|---|
| 1 | intake | 倾听者 — 复述确认需求 | human | — | structured_understanding |
| 2 | brainstorm | 挑战者 — 发散思维找盲点 | human | structured_understanding | ideas_map |
| 3 | deep_dive | 苏格拉底 — 追问细节逼出精确定义 | human | structured_understanding, ideas_map | refined_requirements |
| 4 | draft | 写作者 — 输出 PRD + 技术方案 | human | refined_requirements | prd_draft, tech_design_draft |
| 5 | review | 对抗评审 — 找问题挑毛病 | human | prd_draft, tech_design_draft | review_report |
| 6 | finalize | 编辑 — 根据评审意见修改定稿 | none | prd_draft, tech_design_draft, review_report | final_prd, final_tech_design |

Prompts written in Chinese. Each stage prompt includes: role definition, workflow steps, output format, boundaries, stage_complete condition.

### 10.2 code_review.yaml (3 stages)

| # | Stage | AI Role | Gate | Input Artifacts | Output Artifacts |
|---|---|---|---|---|---|
| 1 | submit | 阅读者 — 理解代码变更 | auto | — | change_summary |
| 2 | review | 评论者 — 多角度分析问题 | human | change_summary | findings |
| 3 | fix | 实施者 — 应用修复 | none | findings | applied_changes |

`submit` uses `gate: auto` — advances immediately after AI reads the code, no human approval needed.

---

## 11. output_dir Resolution

Workflow supports an `output_dir` for generated documents (PRD, tech design, etc.).

**Resolution priority**: `--output-dir` CLI arg > YAML `output_dir` field > default `./workflow_output`

**Relative paths** resolve against `workspace_dir`:
```
workspace_dir = /data/tasks/task_001/
output_dir = ./docs/output
→ resolved: /data/tasks/task_001/docs/output
```

The resolved absolute path is injected into relevant stage prompts so the AI uses WriteTool to save documents there.

---

## 12. Impact Assessment

### What changes

| Item | Status |
|---|---|
| `mini_agent/workflow.py` | NEW |
| `mini_agent/workflow_loader.py` | NEW |
| `mini_agent/workflows/*.yaml` | NEW |
| `mini_agent/cli.py` | MODIFIED (add subcommands) |
| `mini_agent/agent.py` | UNCHANGED |
| `mini_agent/session.py` | UNCHANGED |
| `mini_agent/context/` | UNCHANGED |
| Existing CLI chat mode | UNCHANGED |

### Dependencies

- No new Python packages required
- YAML parsing uses `pyyaml` (already a dependency via pydantic/config)
- All functionality uses existing framework components

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Workflow modifies Agent system block | Updates `agent.config.system_prompt` so `_rebuild_system_block()` in `run_stream()` produces correct content. Also directly updates system block for immediate effect. |
| StageCompleteTool flag race condition | Single-threaded async — flag checked only after `run_stream()` completes |
| YAML parsing errors | Validation on load with clear error messages |
| Stage prompt quality | Ship tested built-in workflows + custom workflow guide |
| Context bloat across stages | Stage transitions compress via existing `context_compress_to_conclusion` |
