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

As defined in the original design document:

- `Layer` enum: L0_CORE, L1_WORKING, L2_REFERENCE, L3_ARCHIVE
- `BlockType` enum: SYSTEM, USER_INTENT, USER_MESSAGE, ASSISTANT_REPLY, TOOL_CALL, SUMMARY, PINNED
- `BlockStatus` enum: ACTIVE, MICRO_COMPRESSED, SUMMARIZED, OBSOLETE, PINNED
- `ContextBlock` dataclass with:
  - Dual-track content: `original_content` (preserved) / `working_content` (compressed)
  - `compression_history: list[dict]` audit trail
  - `depends_on: list[str]` for dependency chain promotion
  - `tool_name`, `tool_input_summary` for tool metadata

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
| `full_layering_mode` | 5 | 0.30 | 0.35 | 0.20 | Research tasks with frequent lookback |

### Dynamic Mode Upgrade

When in Claude Code mode, auto-upgrade to Hybrid when:
- `token_usage_ratio > 0.70` AND `current_turn > 20`

### config.yaml Integration

```yaml
context:
  mode: "hybrid"
  token_budget: 80000
  enable_context_editing: true
  enable_prefix_caching: true
```

`Config` dataclass gains `context: ContextConfig` field. Old `agent.token_limit` removed.

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
- `DefaultTruncateStrategy` — 60% head + 30% tail

**Compression intensity** based on layer:
- L1: `tool_output_max_tokens_l1` (2000), level="light"
- L2+: `tool_output_max_tokens_l2` (200), level="aggressive"

### Stage 3: AutoCompressor (`auto_compressor.py`)

Triggered when `total_active_tokens > trigger_ratio * budget`.

- Selects oldest `MICRO_COMPRESSED` blocks
- One LLM call generates YAML structured summary
- Original blocks marked `SUMMARIZED`, new `SUMMARY` block added at L2
- Supports `auto_compress_model` for using a cheaper model
- Target: compress down to `target_ratio * budget`

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
    def __init__(self, config: ContextConfig, llm_client, tool_registry: dict[str, Tool]):
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

    def add_turn(self, user_msg, assistant_reply, tool_calls: list):
        """Convert a conversation turn into ContextBlocks and add to store."""

    def process_and_assemble(self) -> list[Message]:
        """Run stages 1→2→3, assemble final messages. Core method."""

    def handle_context_tool(self, tool_name, tool_input) -> str:
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

### What changes in Agent:

- **New**: `self.context_manager = ContextManager(config, llm_client, tools)`
- **Modified**: `run_stream()` calls `context_manager.process_and_assemble()` instead of `_summarize_messages()`
- **Modified**: tool call routing — `context_*` tools go to `context_manager.handle_context_tool()`
- **Modified**: system prompt construction — append `context_manager.get_system_prompt_section()`

### What gets deleted from Agent:

- `_summarize_messages()`
- `_estimate_tokens()`
- `_create_summary()`
- `self.messages` list (replaced by BlockStore)
- `self._skip_next_token_check`
- `self.token_limit` / `self.api_total_tokens`

### What stays unchanged:

- `run_stream()` async streaming architecture
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

## 11. Performance Targets

| Metric | Current | Target (Hybrid mode) |
|---|---|---|
| Token reduction at 100 turns | ~40-50% | ~80% |
| Context quality at 50+ turns | Degrades | Stable (clean context) |
| Extra LLM calls per session | 1 per summarization | 3-5 (auto compress only) |
| Micro compression cost | N/A | Zero (deterministic) |
