# Agent Context Manager — Implementation Design

> **Version**: 1.0
> **Date**: 2026-03-15
> **Based on**: `2026-03-15-agent-context-manager-final-design.md`
> **Scope**: Full 5-stage pipeline implementation, excluding Multi-Agent Handoff

---

## 1. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Full pipeline minus Multi-Agent Handoff | Handoff deferred to SubAgent phase |
| Integration pattern | Composition — ContextManager as independent module | Agent holds and delegates to ContextManager; testable in isolation |
| Context Editor default | Enabled by default | Model agent capabilities improving; users can disable via config |
| Tool compress strategy | Tool base class extended with `compress_strategy` property | MCP/Skill tools also produce SQL, file, bash outputs needing specialized compression |
| Implementation approach | Incremental per-stage, unified switch at end | Each stage independently testable; problems easy to isolate |

---

## 2. Module Structure

```
mini_agent/context/
├── __init__.py              # Exports: ContextManager, ContextConfig
├── config.py                # ContextConfig dataclass + 3 preset modes
├── models.py                # Layer, BlockType, BlockStatus, ContextBlock
├── block_store.py           # BlockStore storage and queries
├── token_counter.py         # count_tokens() unified entry point
├── classifier.py            # Stage 1: LayerClassifier
├── micro_compressor.py      # Stage 2: MicroCompressor + strategy dispatch
├── compress_strategies.py   # ToolCompressStrategy implementations
├── auto_compressor.py       # Stage 3: AutoCompressor
├── context_editor.py        # Stage 4: ContextEditor + tool definitions
├── budget_assembler.py      # Stage 5: BudgetAssembler
└── context_manager.py       # ContextManager facade, orchestrates 5 stages
```

---

## 3. Core Data Model (`models.py`)

All fields from Section 4.1 of the original design document are included:

- `Layer` enum: L0_CORE, L1_WORKING, L2_REFERENCE, L3_ARCHIVE
- `BlockType` enum: SYSTEM, USER_INTENT, USER_MESSAGE, ASSISTANT_REPLY, TOOL_CALL, SUMMARY, PINNED
- `BlockStatus` enum: ACTIVE, MICRO_COMPRESSED, SUMMARIZED, OBSOLETE, PINNED
- `ContextBlock` dataclass — full field list:
  - `id: str` — e.g. `"turn_015_0_execute_sql"` (turn + index + tool_name to avoid collision)
  - `turn_id: int`
  - `block_type: BlockType`
  - `layer: Layer`
  - `status: BlockStatus` (default ACTIVE)
  - `original_content: str` — preserved, never modified after creation
  - `working_content: str` — compressed version, what gets sent to LLM
  - `token_count: int` — current working_content tokens
  - `original_token_count: int`
  - `tool_name: Optional[str]`
  - `tool_input_summary: str` — always retained for traceability
  - `depends_on: list[str]` — block IDs this block depends on
  - `superseded_by: Optional[str]` — block ID that replaces this one
  - `tags: list[str]` — free-form tags for filtering
  - `compression_history: list[dict]` — audit trail with stage/before/after

### BlockStore Interface (`block_store.py`)

```python
class BlockStore:
    def add(self, block: ContextBlock) -> ContextBlock
    def get(self, block_id: str) -> Optional[ContextBlock]
    def get_blocks_by_turn(self, turn_id: int) -> list[ContextBlock]
    def all(self) -> list[ContextBlock]
    def active_blocks(self) -> list[ContextBlock]   # excludes OBSOLETE, SUMMARIZED
    def total_active_tokens(self) -> int
    def remove(self, block_id: str)
```

---

## 4. Tool Base Class Extension (`tools/base.py`)

```python
class Tool(ABC):
    # ... existing properties ...

    @property
    def compress_strategy(self) -> Optional["ToolCompressStrategy"]:
        """Override to provide a custom compression strategy for this tool's output."""
        return None
```

MicroCompressor strategy lookup priority:
1. `tool.compress_strategy` (tool instance provides its own)
2. `MicroCompressor.strategies[tool_name]` (built-in name mapping)
3. `DefaultTruncateStrategy` (final fallback)

---

## 5. Configuration (`config.py`)

### ContextConfig

Single dataclass controls all behavior. Key fields:

- **Token Budget**: `total_token_budget` (default 80,000)
- **Layer Budget Ratios**: `l0/l1/l2/l3_budget_ratio` (sum ~1.0)
- **Layer Windows**: `l1_window_turns`, `l2_window_turns` (controls demotion timing)
- **Micro Compression**: `micro_compress_after_turns`, `tool_output_max_tokens_l1/l2`
- **Auto Compression**: `auto_compress_trigger_ratio` (0.85), `auto_compress_target_ratio` (0.60), `auto_compress_model`
- **Context Editing**: `enable_context_editing` (default True)
- **Prefix Caching**: `enable_prefix_caching` (default True)

### Preset Modes

| Mode | l1_window | l1_budget | l2_budget | l3_budget | Best for |
|---|---|---|---|---|---|
| `claude_code_mode` | INF | 0.85 | 0.0 | 0.0 | Short conversations (<50 turns) |
| `hybrid_mode` (default) | 8 | 0.40 | 0.30 | 0.15 | Long conversations (50-200 turns) |
| `full_layering_mode` | 5 (l2=15) | 0.30 | 0.35 | 0.20 | Research tasks with frequent lookback |

### Dynamic Mode Upgrade

When in Claude Code mode, auto-upgrade to Hybrid when:
- `token_usage_ratio > 0.70` AND `current_turn > 20`
- All stage configs must be updated: classifier, micro, auto, assembler (original design missed auto)

### config.yaml Integration

```yaml
context:
  mode: "hybrid"
  token_budget: 80000
  enable_context_editing: true
  enable_prefix_caching: true
```

Existing `Config` uses Pydantic `BaseModel`. `ContextConfig` should also be a Pydantic `BaseModel` (not a plain dataclass) for consistency. The `mode` field maps to a preset factory at init time:

```python
class ContextConfig(BaseModel):
    mode: str = "hybrid"  # "claude_code" | "hybrid" | "full_layering"
    # ... fields with defaults from the selected preset ...

    @classmethod
    def from_mode(cls, mode: str, **overrides) -> "ContextConfig":
        presets = {"claude_code": {...}, "hybrid": {...}, "full_layering": {...}}
        return cls(**{**presets[mode], **overrides})
```

`Config` gains `context: ContextConfig` field. Old `Agent.__init__` `token_limit` parameter removed.

---

## 6. Pipeline Stages

### Stage 1: LayerClassifier (`classifier.py`)

Pure rules, zero LLM cost. Assigns each block to L0/L1/L2/L3 based on age.

- Pinned blocks → always L0
- Blocks within `l1_window_turns` → L1
- Blocks within `l2_window_turns` → L2
- Older → L3
- **Dependency chain promotion**: old blocks referenced by recent blocks' `depends_on` get promoted back to L1
- Becomes no-op when `l1_window_turns = INF` (Claude Code mode)

### Stage 2: MicroCompressor (`micro_compressor.py` + `compress_strategies.py`)

Runs every turn, zero LLM cost. Handles 60-80% of token bloat.

```python
class MicroCompressor:
    def __init__(self, config: ContextConfig, tool_registry: dict[str, Tool]):
        self.config = config
        self.tool_registry = tool_registry
        self.strategies: dict[str, ToolCompressStrategy] = {
            "execute_sql": SqlResultStrategy(),
            "execute_code": CodeOutputStrategy(),
            "read_file": FileReadStrategy(),
            "write_file": PassThroughStrategy(),
            "web_search": SearchResultStrategy(),
            "list_files": PassThroughStrategy(),
        }
        self.default_strategy = DefaultTruncateStrategy()
```

**Strategy dispatch** (differs from original design — uses tool registry):

```python
def _get_strategy(self, block: ContextBlock) -> ToolCompressStrategy:
    # 1. Tool instance's own strategy
    tool = self.tool_registry.get(block.tool_name)
    if tool and tool.compress_strategy:
        return tool.compress_strategy
    # 2. Built-in name mapping
    if block.tool_name in self.strategies:
        return self.strategies[block.tool_name]
    # 3. Fallback
    return self.default_strategy
```

**Built-in strategies**:
- `SqlResultStrategy` — light: schema + 10 rows; aggressive: schema + stats only
- `CodeOutputStrategy` — light: head+tail 40 lines; aggressive: errors + tail 5 lines
- `FileReadStrategy` — light: 30 lines + structure; aggressive: structure summary only
- `SearchResultStrategy` — light: title+snippet; aggressive: titles only
- `PassThroughStrategy` — returns content unchanged (for write confirmations, directory listings)
- `DefaultTruncateStrategy` — 60% head + 30% tail

**Compression intensity** based on layer:
- L1: `tool_output_max_tokens_l1` (2000), level="light"
- L2+: `tool_output_max_tokens_l2` (200), level="aggressive"

### Stage 3: AutoCompressor (`auto_compressor.py`)

Triggered when `total_active_tokens > trigger_ratio * budget`. **All methods are async** to match the codebase's async-first architecture.

```python
class AutoCompressor:
    def __init__(self, config: ContextConfig, llm_client: LLMClient):
        self.config = config
        self.llm = llm_client

    def should_trigger(self, store: BlockStore) -> bool: ...

    async def compress(self, store: BlockStore, current_turn: int) -> Optional[ContextBlock]:
        """Async — calls LLM for summary generation."""
        ...

    async def _generate_summary(self, blocks: list[ContextBlock]) -> str:
        """Uses llm_client's async interface (non-streaming complete or collected stream)."""
        ...
```

- Selects oldest `MICRO_COMPRESSED` blocks
- One async LLM call generates YAML structured summary
- Original blocks marked `SUMMARIZED`, new `SUMMARY` block added at L2
- Supports `auto_compress_model` for using a cheaper model
- Target: compress down to `target_ratio * budget`
- **Error handling**: if LLM call fails, skip compression this turn (blocks stay MICRO_COMPRESSED), retry on next trigger

### Stage 4: ContextEditor (`context_editor.py`)

3 tools registered as agent-callable tools:

- `context_mark_obsolete(turn_ids, reason)` — marks blocks obsolete, frees tokens
- `context_compress_to_conclusion(turn_ids, conclusion)` — replaces multi-turn process with conclusion
- `context_pin(turn_ids, reason)` — pins blocks to L0, prevents compression

System prompt section provided via `ContextManager.get_system_prompt_section()`.

Context editing tools routed separately from business tools in Agent loop: `tool_name.startswith("context_")` → ContextEditor, else → normal tool execution.

### Stage 5: BudgetAssembler (`budget_assembler.py`)

Fills layers by priority, produces final `messages[]`:

1. L0 — all core blocks (always loaded)
2. L1 — newest first until L1 budget spent
3. L2 — chronological until L2 budget spent
4. L3 — remaining budget

**Prefix caching optimization**: stable content (system + pinned + summaries) ordered first, volatile (recent turns) last. Maximizes Anthropic/OpenAI prefix cache hits.

**Message format conversion**: ContextBlock → provider-specific message dicts. Tool blocks expanded back to `assistant/tool_use` + `user/tool_result` pairs.

---

## 7. ContextManager Facade (`context_manager.py`)

```python
class ContextManager:
    def __init__(self, config: ContextConfig, llm_client: LLMClient,
                 tool_registry: dict[str, Tool]):
        self.config = config
        self.store = BlockStore()
        self.classifier = LayerClassifier(config)
        self.micro = MicroCompressor(config, tool_registry)
        self.auto = AutoCompressor(config, llm_client)
        self.editor = ContextEditor(self.store)
        self.assembler = BudgetAssembler(config)
        self.current_turn = 0

    def init_system(self, system_prompt: str):
        """Initialize system block (with context editing prompt if enabled)."""

    def add_user_message(self, content: str):
        """Record user message as ContextBlock. First message becomes USER_INTENT.
        Sets both token_count and original_token_count."""

    def add_assistant_reply(self, content: str, thinking: Optional[str] = None):
        """Record assistant reply as ContextBlock.
        Sets both token_count and original_token_count.
        Thinking content stored in ContextBlock.tags as 'has_thinking' marker;
        full thinking text not retained (it's not sent back to the LLM)."""

    def add_tool_call(self, tool_name: str, tool_input: dict, tool_result: str):
        """Record tool_use + tool_result as a single TOOL_CALL ContextBlock.
        tool_input stored as JSON string in tool_input_summary (truncated to 200 chars).
        BudgetAssembler json.loads() it back to dict when expanding to tool_use messages.
        Sets both token_count and original_token_count."""

    async def process_and_assemble(self) -> list[Message]:
        """Run stages 1→2→3 (async for auto compress), assemble final messages.

        Pseudocode:
            self.classifier.classify(self.store.all(), self.current_turn)
            self.micro.compress(self.store.all(), self.current_turn)
            if self.auto.should_trigger(self.store):
                await self.auto.compress(self.store, self.current_turn)
            self._maybe_upgrade_mode()
            messages, usage = self.assembler.assemble(self.store.all())
            return messages
        """

    def _maybe_upgrade_mode(self):
        """Auto-upgrade Claude Code → Hybrid when pressure builds.
        Propagates new config to ALL stages: classifier, micro, auto, assembler."""

    def handle_context_tool(self, tool_name: str, tool_input: dict) -> str:
        """Route context_* tool calls to ContextEditor."""

    def get_system_prompt_section(self) -> str:
        """Return context editing guidance for system prompt."""

    def get_context_tools(self) -> list[Tool]:
        """Return context editing tool definitions for registration."""

    def get_status(self) -> dict:
        """Return monitoring/debugging info (block counts, token usage, etc.)."""
```

---

## 8. Agent Integration

### Async streaming integration

The existing `Agent.run_stream()` is an async generator that yields `StreamEvent` objects. The integration must preserve this architecture:

```python
async def run_stream(self, user_message: str):
    self.context_manager.add_user_message(user_message)

    for step in range(self.max_steps):
        # Async: auto compress may call LLM
        messages = await self.context_manager.process_and_assemble()

        async for event in self.llm_client.generate_stream(messages, tools):
            yield event  # Stream to caller as before

        # After stream completes, record results
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.name.startswith("context_"):
                    result = self.context_manager.handle_context_tool(
                        tool_call.name, tool_call.input)
                    # No yield — context edits are invisible to user
                else:
                    result = await self._execute_tool(tool_call)
                    self.context_manager.add_tool_call(
                        tool_call.name, tool_call.input, result)
        else:
            self.context_manager.add_assistant_reply(response.content)
            break
```

### What changes in Agent:

- **New**: `self.context_manager = ContextManager(config, llm_client, tools)`
- **Modified**: `run_stream()` uses `context_manager` for message assembly (async)
- **Modified**: tool call routing — `context_*` tools go to `context_manager.handle_context_tool()`
- **Modified**: system prompt construction — append `context_manager.get_system_prompt_section()`
- **Modified**: tool registration — include `context_manager.get_context_tools()` alongside business tools

### What gets deleted from Agent:

- `_summarize_messages()`
- `_estimate_tokens()`
- `_create_summary()`
- `self.messages` list (replaced by BlockStore)
- `self._skip_next_token_check`
- `self.token_limit` / `self.api_total_tokens`

### What stays unchanged:

- `run_stream()` async streaming architecture (preserved, not replaced)
- `_execute_tool()` business tool execution
- Cancellation mechanism (`asyncio.Event`)
- CLI entry point and interactive logic

---

## 9. Implementation Order

| Step | Scope | Key Deliverable |
|---|---|---|
| 1 | `models.py`, `block_store.py`, `config.py`, `token_counter.py` | Pure data model, zero risk |
| 2 | `classifier.py`, `budget_assembler.py` | Minimal "messages in → messages out" loop |
| 3 | `compress_strategies.py`, `micro_compressor.py`, Tool base class extension | Highest ROI: deterministic tool output compression |
| 4 | `auto_compressor.py` | LLM-powered summarization |
| 5 | `context_editor.py` | Agent self-editing tools |
| 6 | `context_manager.py` facade + Agent integration | Unified switch, delete old code |
| 7 | Tests + config.yaml integration | End-to-end validation |

---

## 10. Excluded from This Phase

- **Multi-Agent Context Handoff** (`MultiAgentContextBridge`) — deferred to SubAgent implementation phase
- Section 11 of the original design document is not implemented

---

## 11. Token Counter (`token_counter.py`)

Consolidates existing `Agent._estimate_tokens()` logic into a shared utility:

```python
def count_tokens(text: str) -> int:
    """Multi-model token counting. tiktoken primary, char heuristic fallback."""
```

- Primary: `tiktoken.get_encoding("cl100k_base")` (works for GPT-4/Claude, reasonable approximation for others)
- Fallback: ~4 chars/token for ASCII, ~1.5 chars/token for CJK
- Existing tiktoken import in `agent.py` is removed; all token counting goes through this module

---

## 12. Additional Design Notes

### Multiple tool calls per turn

A single LLM response can contain multiple `tool_use` blocks. Each becomes a separate `ContextBlock` with an index suffix to avoid ID collision: `turn_015_0_execute_sql`, `turn_015_1_read_file`.

### Prefix caching and message ordering

The `_optimize_for_caching` reorder only affects the stable/volatile boundary. Within the volatile section, chronological order is preserved. This ensures LLMs that expect temporal ordering still work correctly.

### Logging/observability

Pipeline stages emit log events via the existing `AgentLogger`:
- Block creation: `[CONTEXT] +block turn_015_0_execute_sql (TOOL_CALL, 8200 tokens)`
- Micro compression: `[CONTEXT] micro: turn_012_0_execute_sql 8200→1500 tokens (light)`
- Auto compression: `[CONTEXT] auto: turns 5-12 summarized, freed 12000 tokens`
- Context edit: `[CONTEXT] edit: mark_obsolete turns [8,9], freed 3200 tokens`
- Mode upgrade: `[CONTEXT] mode upgrade: claude_code → hybrid (turn 22, usage 72%)`

---

## 13. Performance Targets

| Metric | Current | Target (Hybrid mode) |
|---|---|---|
| Token reduction at 100 turns | ~40-50% | ~80% |
| Context quality at 50+ turns | Degrades | Stable (clean context) |
| Extra LLM calls per session | 1 per summarization | 3-5 (auto compress only) |
| Micro compression cost | N/A | Zero (deterministic) |
