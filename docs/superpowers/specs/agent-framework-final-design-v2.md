# Agent Framework — Final Design Document

> **Version**: 2.0 (consolidated)  
> **Scope**: Context management + sub-agent architecture + preset templates  
> **Target**: 100-turn conversations, multi-model compatible, CLI + SDK  
> **Priority**: LLM output quality > Token savings > Generality > Simplicity

---

## Table of Contents

1. [Terminology](#1-terminology)
2. [Architecture Overview](#2-architecture-overview)
3. [Context Management](#3-context-management)
4. [Sub-Agent Architecture](#4-sub-agent-architecture)
5. [SharedState](#5-sharedstate)
6. [Session API](#6-session-api)
7. [Preset Agent Templates](#7-preset-agent-templates)
8. [CLI Design](#8-cli-design)
9. [Appendices](#9-appendices)

---

## 1. Terminology

Precise terminology prevents confusion. These definitions are used consistently throughout.

### 1.1 Turn and Step

```
Turn (轮):
  One user input → 0~N steps → one final assistant reply.
  Each call to chat() opens a new turn.
  Tracked by: current_turn counter (increments per chat() call).

Step (步):
  One tool call cycle within a turn: agent emits tool_use → framework
  executes tool → returns tool_result → agent continues.
  A single turn may contain many steps.
  Tracked by: step_count counter.

Example:
  Turn 5 (user: "按月汇总销售额，画个趋势图")
    Step 1: execute_sql → 200 rows JSON
    Step 2: execute_code → "chart saved"
    Step 3: context_mark_obsolete → "freed 2000 tokens"
    Step 4: write_file → "saved to report.md"
    Reply: "月度趋势图已生成。Q4 最高..."
```

### 1.2 Config Preset vs Call Depth

These are two **orthogonal** concepts that must never be confused.

```
Config Preset:
  Describes what an agent IS — its context config, system prompt, tools.
  Examples: claude_code_mode, hybrid_mode, full_layering_mode.
  Chosen by the developer at config time.

Call Depth:
  Describes how deep agent nesting goes at runtime.
  Depth 0: main agent executing tools directly.
  Depth 1: main agent → sub-agent (sub-agent runs to completion, returns result).
  Depth 2: main agent → sub-agent → sub-sub-agent (rare, advanced).
  Controlled by: max_delegation_depth setting.

These are independent. An agent with full_layering_mode config can be at depth 0 or 1.
An agent with claude_code_mode config can also be at depth 0 or 1.
```

### 1.3 Layering vs Compression

Two orthogonal dimensions of context management:

```
Spatial selection (Cursor lineage):
  Which blocks to include in context, organized by layer (L0/L1/L2/L3).
  Controlled by: l1_window_turns, l2_window_turns, budget ratios.
  Can be disabled: set l1_window_turns = INF → everything in L1, no layering.

Temporal compression (Claude Code lineage):
  How much detail each block retains over time.
  Three stages: micro compress → auto compress → agent self-edit.
  Always active regardless of layering config.
```

---

## 2. Architecture Overview

### 2.1 Core Principle

There is one `Agent` class. Everything — main agent, sub-agent, orchestrator, worker —
is the same `Agent` with different `AgentConfig`. No subclasses, no special types.

### 2.2 System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Session                              │
│                                                             │
│  SharedState ──────── shared across all agents ───────────  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Agent (main or orchestrator)                         │  │
│  │                                                       │  │
│  │  AgentConfig: identity, tools, context config         │  │
│  │                                                       │  │
│  │  Context Pipeline (runs inside every agent):          │  │
│  │    BlockStore                                         │  │
│  │      → ① Layer Classifier (rules-based)               │  │
│  │      → ② Micro Compressor (per-tool-type, zero LLM)   │  │
│  │      → ③ Auto Compressor (threshold, one LLM call)    │  │
│  │      → ④ Context Editor (agent self-edits via tools)  │  │
│  │      → ⑤ Budget Assembler (fills layers → messages[]) │  │
│  │                                                       │  │
│  │  Registered sub-agents (optional):                    │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                 │  │
│  │  │ Agent   │ │ Agent   │ │ Agent   │                  │  │
│  │  │ (own    │ │ (own    │ │ (own    │  Each has its    │  │
│  │  │ config, │ │ config, │ │ config, │  own full ctx    │  │
│  │  │ own     │ │ own     │ │ own     │  pipeline and    │  │
│  │  │ store)  │ │ store)  │ │ store)  │  ephemeral store │  │
│  │  └─────────┘ └─────────┘ └─────────┘                  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  EventBus ──────── observability across all agents ───────  │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Two SDK Entry Points

```python
# Entry point 1 (covers 95% of use cases):
# Single agent, optionally with delegation capability.
session = Session.create(
    tools=[...],                                # Agent's own tools
    sub_agents={"analyst": data_analyst(), ...}, # Optional: adds delegate_to_agent tool
)

# Entry point 2 (advanced: planning-first coordinator):
# Agent identity changes to "I plan and coordinate."
session = Session.orchestrator(
    workers={"analyst": data_analyst(), "writer": writer(), ...},
)
```

Session.create() with sub_agents is NOT a separate "mode" — it simply registers
delegation targets. The LLM decides per-task whether to delegate or handle directly.
Session.orchestrator() IS a different mode because the agent's identity changes.

---

## 3. Context Management

### 3.1 Data Model: ContextBlock

The fundamental unit. Messages are grouped into logical blocks for atomic management.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Layer(Enum):
    L0_CORE = 0       # Always loaded, never compressed
    L1_WORKING = 1    # Recent active context
    L2_REFERENCE = 2  # Compressed summaries
    L3_ARCHIVE = 3    # Deep-compressed key-value pairs


class BlockType(Enum):
    SYSTEM = "system"
    USER_INTENT = "user_intent"      # First user message (task anchor)
    USER_MESSAGE = "user_message"
    ASSISTANT_REPLY = "assistant"
    TOOL_CALL = "tool_call"          # tool_use + tool_result as one unit
    SUMMARY = "summary"              # Generated by auto compressor
    PINNED = "pinned"                # Pinned by agent


class BlockStatus(Enum):
    ACTIVE = "active"
    MICRO_COMPRESSED = "micro"       # Tool output truncated by rules
    SUMMARIZED = "summarized"        # Replaced by a summary block
    OBSOLETE = "obsolete"            # Marked obsolete by agent
    PINNED = "pinned"                # Protected from compression


@dataclass
class ContextBlock:
    id: str                           # e.g. "turn_015_tool_execute_sql"
    turn_id: int
    block_type: BlockType
    layer: Layer
    status: BlockStatus = BlockStatus.ACTIVE

    # Content (dual-track: original preserved, working gets compressed)
    original_content: str = ""        # Never modified — enables recovery
    working_content: str = ""         # Sent to LLM, may be compressed

    # Metrics
    token_count: int = 0              # Current working_content tokens
    original_token_count: int = 0

    # Tool metadata
    tool_name: Optional[str] = None
    tool_input_summary: str = ""      # Always retained for traceability

    # Relationships
    depends_on: list[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    # Audit trail
    compression_history: list[dict] = field(default_factory=list)
```

### 3.2 BlockStore

```python
class BlockStore:
    """Central storage for all context blocks within one agent."""

    def __init__(self):
        self._blocks: dict[str, ContextBlock] = {}
        self._by_turn: dict[int, list[str]] = {}

    def add(self, block: ContextBlock) -> ContextBlock:
        self._blocks[block.id] = block
        self._by_turn.setdefault(block.turn_id, []).append(block.id)
        return block

    def get(self, block_id: str) -> Optional[ContextBlock]:
        return self._blocks.get(block_id)

    def get_blocks_by_turn(self, turn_id: int) -> list[ContextBlock]:
        ids = self._by_turn.get(turn_id, [])
        return [self._blocks[i] for i in ids if i in self._blocks]

    def all(self) -> list[ContextBlock]:
        return list(self._blocks.values())

    def active_blocks(self) -> list[ContextBlock]:
        return [b for b in self._blocks.values()
                if b.status not in (BlockStatus.OBSOLETE, BlockStatus.SUMMARIZED)]

    def total_active_tokens(self) -> int:
        return sum(b.token_count for b in self.active_blocks())
```

### 3.3 ContextConfig — Configurable Modes

A single config controls all behavior. Layering and compression are orthogonal —
setting layering params to extreme values disables layering without changing code paths.

```python
@dataclass
class ContextConfig:

    # ── Token Budget ──
    total_token_budget: int = 80_000

    # ── Layer Budget Ratios ──
    l0_budget_ratio: float = 0.15
    l1_budget_ratio: float = 0.40
    l2_budget_ratio: float = 0.30
    l3_budget_ratio: float = 0.15

    # ── Layer Windows (turns) ──
    # Setting to large values disables layering (everything stays L1).
    l1_window_turns: int = 8
    l2_window_turns: int = 30

    # ── Micro Compression ──
    micro_compress_after_turns: int = 3
    tool_output_max_tokens_l1: int = 2000
    tool_output_max_tokens_l2: int = 200

    # ── Auto Compression ──
    auto_compress_trigger_ratio: float = 0.85
    auto_compress_target_ratio: float = 0.60
    auto_compress_model: Optional[str] = None  # None = use main model

    # ── Context Editing ──
    enable_context_editing: bool = True

    # ── Prompt Caching ──
    enable_prefix_caching: bool = True

    # ── Derived ──

    @property
    def layering_enabled(self) -> bool:
        return self.l1_window_turns < 1000

    @property
    def layer_budgets(self) -> dict:
        if not self.layering_enabled:
            return {
                "L0": int(self.total_token_budget * self.l0_budget_ratio),
                "L1": int(self.total_token_budget * (1.0 - self.l0_budget_ratio)),
                "L2": 0, "L3": 0,
            }
        return {
            "L0": int(self.total_token_budget * self.l0_budget_ratio),
            "L1": int(self.total_token_budget * self.l1_budget_ratio),
            "L2": int(self.total_token_budget * self.l2_budget_ratio),
            "L3": int(self.total_token_budget * self.l3_budget_ratio),
        }

    # ── Factory Presets ──

    @classmethod
    def claude_code_mode(cls, token_budget: int = 80_000) -> "ContextConfig":
        """
        Layering disabled. Pure progressive compression.
        l1_window = INF → everything in L1 → classifier is no-op.
        BudgetAssembler loads newest-first until budget full = sliding window.
        Best for: short conversations, sub-agents, simple tasks.
        """
        return cls(
            total_token_budget=token_budget,
            l0_budget_ratio=0.15, l1_budget_ratio=0.85,
            l2_budget_ratio=0.0, l3_budget_ratio=0.0,
            l1_window_turns=999_999, l2_window_turns=999_999,
        )

    @classmethod
    def hybrid_mode(cls, token_budget: int = 80_000) -> "ContextConfig":
        """
        Layering + compression. Recommended default.
        Recent 8 turns in L1, turns 9-30 in L2 as summaries, 30+ in L3.
        Best for: long conversations, data analysis, multi-step workflows.
        """
        return cls(
            total_token_budget=token_budget,
            l0_budget_ratio=0.15, l1_budget_ratio=0.40,
            l2_budget_ratio=0.30, l3_budget_ratio=0.15,
            l1_window_turns=8, l2_window_turns=30,
        )

    @classmethod
    def full_layering_mode(cls, token_budget: int = 80_000) -> "ContextConfig":
        """
        Aggressive layering. More budget for reference/archive layers.
        Best for: orchestrator (accumulates worker summaries), research tasks.
        """
        return cls(
            total_token_budget=token_budget,
            l0_budget_ratio=0.15, l1_budget_ratio=0.30,
            l2_budget_ratio=0.35, l3_budget_ratio=0.20,
            l1_window_turns=5, l2_window_turns=15,
        )
```

### 3.4 Configuration Space

```
┌──────────────────────────────────────────────────────────────────┐
│  l1_window = INF           l1_window = 8          l1_window = 5  │
│  l1_budget = 0.85          l1_budget = 0.40       l1_budget = 0.30│
│  l2_budget = 0.00          l2_budget = 0.30       l2_budget = 0.35│
│                                                                   │
│  ◄──────────────────────────────────────────────────────────────► │
│  claude_code_mode          hybrid_mode          full_layering_mode│
│                          (recommended)                            │
│                                                                   │
│  Layering: OFF             Layering: ON          Layering: HEAVY  │
│  Compression: FULL         Compression: FULL     Compression: FULL│
│  Same code. Same pipeline. Only config values differ.             │
└──────────────────────────────────────────────────────────────────┘
```

Dynamic switching is supported at runtime:

```python
def maybe_upgrade_mode(self, current_turn: int, token_usage_ratio: float):
    """Auto-upgrade from claude_code to hybrid when pressure builds."""
    if (self.ctx_config.l1_window_turns > 1000
            and token_usage_ratio > 0.70
            and current_turn > 20):
        self.ctx_config = ContextConfig.hybrid_mode(
            token_budget=self.ctx_config.total_token_budget
        )
        # Reclassify all blocks under new config — no data loss
        self.classifier.classify(self.store.all(), current_turn)
```

### 3.5 Pipeline Stage 1: Layer Classifier

Rules-based, zero LLM cost. Becomes no-op when `l1_window_turns = INF`.

```python
class LayerClassifier:
    def __init__(self, config: ContextConfig):
        self.config = config

    def classify(self, blocks: list[ContextBlock], current_turn: int):
        for block in blocks:
            if block.status == BlockStatus.PINNED:
                block.layer = Layer.L0_CORE
                continue
            if block.status in (BlockStatus.OBSOLETE, BlockStatus.SUMMARIZED):
                continue

            age = current_turn - block.turn_id

            if block.block_type in (BlockType.SYSTEM, BlockType.USER_INTENT):
                block.layer = Layer.L0_CORE
            elif age <= self.config.l1_window_turns:
                block.layer = Layer.L1_WORKING
            elif age <= self.config.l2_window_turns:
                block.layer = Layer.L2_REFERENCE
            else:
                block.layer = Layer.L3_ARCHIVE

            # Promotion: old block still depended on by recent block → stay in L1
            if block.layer >= Layer.L2_REFERENCE:
                if self._is_in_active_chain(block, blocks, current_turn):
                    block.layer = Layer.L1_WORKING

    def _is_in_active_chain(self, block, all_blocks, current_turn) -> bool:
        for other in all_blocks:
            if (current_turn - other.turn_id) <= self.config.l1_window_turns:
                if block.id in other.depends_on:
                    return True
        return False
```

### 3.6 Pipeline Stage 2: Micro Compressor

Highest ROI. Runs every turn, zero LLM cost, handles 60-80% of token bloat.
Different tool types get different compression strategies.

```python
from abc import ABC, abstractmethod
import json, re


class ToolCompressStrategy(ABC):
    @abstractmethod
    def compress(self, content: str, tool_name: str,
                 max_tokens: int, level: str) -> str: ...


class SqlResultStrategy(ToolCompressStrategy):
    def compress(self, content, tool_name, max_tokens, level):
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return DefaultTruncateStrategy().compress(content, tool_name, max_tokens, level)
        rows = data.get("rows", [])
        columns = data.get("columns", list(rows[0].keys()) if rows else [])
        if level == "light":
            return json.dumps({
                "columns": columns, "total_rows": len(rows),
                "sample_rows": rows[:10],
                "note": f"[micro-compressed] Original {len(rows)} rows, showing first 10"
            }, ensure_ascii=False)
        else:
            stats = {}
            for col in columns[:8]:
                values = [r.get(col) for r in rows if r.get(col) is not None]
                if values and isinstance(values[0], (int, float)):
                    stats[col] = {"min": min(values), "max": max(values),
                                  "avg": round(sum(values)/len(values), 2)}
            return (f"[SQL summary] {len(rows)} rows, {len(columns)} cols\n"
                    f"Columns: {', '.join(columns)}\nStats: {json.dumps(stats, ensure_ascii=False)}")


class CodeOutputStrategy(ToolCompressStrategy):
    def compress(self, content, tool_name, max_tokens, level):
        lines = content.split('\n')
        errors = [l for l in lines if any(k in l.lower() for k in ['error', 'exception'])]
        if level == "light":
            head, tail = lines[:20], lines[-20:] if len(lines) > 40 else []
            omitted = max(0, len(lines) - 40)
            r = '\n'.join(head)
            if omitted > 0:
                r += f"\n\n[...{omitted} lines omitted...]\n\n" + '\n'.join(tail)
            return r
        else:
            parts = [f"[exec result] {len(lines)} lines"]
            if errors: parts.append("Errors: " + "; ".join(errors[:3]))
            parts.append("Tail:\n" + '\n'.join(lines[-5:]))
            return '\n'.join(parts)


class FileReadStrategy(ToolCompressStrategy):
    def compress(self, content, tool_name, max_tokens, level):
        lines = content.split('\n')
        classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
        functions = re.findall(r'^(?:def|function|async)\s+(\w+)', content, re.MULTILINE)
        structure = f"classes={classes}, functions={functions}"
        if level == "light":
            return '\n'.join(lines[:30]) + f"\n\n[...{len(lines)} lines total...]\nStructure: {structure}"
        else:
            return (f"[file summary] {len(lines)} lines\nStructure: {structure}\n"
                    f"[Full content analyzed in the assistant reply that followed]")


class SearchResultStrategy(ToolCompressStrategy):
    def compress(self, content, tool_name, max_tokens, level):
        try:
            results = json.loads(content)
        except json.JSONDecodeError:
            return DefaultTruncateStrategy().compress(content, tool_name, max_tokens, level)
        if level == "light":
            return json.dumps([
                {"title": r.get("title",""), "url": r.get("url",""),
                 "snippet": r.get("snippet","")[:200]}
                for r in results
            ], ensure_ascii=False)
        else:
            titles = [r.get("title", "") for r in results]
            return f"[search summary] {len(results)} results: " + "; ".join(titles)


class DelegationResultStrategy(ToolCompressStrategy):
    """Compression for delegate_to_agent results."""
    def compress(self, content, tool_name, max_tokens, level):
        lines = content.split('\n')
        header = lines[0] if lines else ""
        result_start = next((i for i, l in enumerate(lines) if l.startswith("Result:")), 0)
        result_text = '\n'.join(lines[result_start:])
        if level == "light":
            return f"{header}\n{result_text[:2000]}"
        else:
            return f"{header}\n{result_text[:800]}\n[Full delegation result truncated]"


class DefaultTruncateStrategy(ToolCompressStrategy):
    def compress(self, content, tool_name, max_tokens, level):
        max_chars = max_tokens * 4
        if len(content) <= max_chars: return content
        head, tail = int(max_chars * 0.6), int(max_chars * 0.3)
        return content[:head] + f"\n\n[...{len(content)-head-tail} chars omitted...]\n\n" + content[-tail:]


class MicroCompressor:
    def __init__(self, config: ContextConfig):
        self.config = config
        self.strategies: dict[str, ToolCompressStrategy] = {
            "execute_sql": SqlResultStrategy(),
            "execute_code": CodeOutputStrategy(),
            "read_file": FileReadStrategy(),
            "write_file": DefaultTruncateStrategy(),   # Already small
            "web_search": SearchResultStrategy(),
            "list_files": DefaultTruncateStrategy(),    # Already small
            "delegate_to_agent": DelegationResultStrategy(),
        }
        self.default_strategy = DefaultTruncateStrategy()

    def compress(self, blocks: list[ContextBlock], current_turn: int):
        for block in blocks:
            if block.block_type != BlockType.TOOL_CALL: continue
            if block.status in (BlockStatus.OBSOLETE, BlockStatus.PINNED): continue

            age = current_turn - block.turn_id
            if age <= self.config.micro_compress_after_turns: continue

            strategy = self.strategies.get(block.tool_name, self.default_strategy)

            if block.layer == Layer.L1_WORKING:
                max_tokens, level = self.config.tool_output_max_tokens_l1, "light"
            else:
                max_tokens, level = self.config.tool_output_max_tokens_l2, "aggressive"

            if block.token_count > max_tokens:
                before = block.token_count
                block.working_content = strategy.compress(
                    block.original_content, block.tool_name, max_tokens, level)
                block.token_count = count_tokens(block.working_content)
                block.status = BlockStatus.MICRO_COMPRESSED
                block.compression_history.append({
                    "stage": "micro", "level": level,
                    "before_tokens": before, "after_tokens": block.token_count})
```

### 3.7 Pipeline Stage 3: Auto Compressor

Triggered when total tokens exceed threshold. One LLM call generates structured summary.

```python
class AutoCompressor:
    def __init__(self, config: ContextConfig, llm_client):
        self.config = config
        self.llm = llm_client

    def should_trigger(self, store: BlockStore) -> bool:
        threshold = self.config.total_token_budget * self.config.auto_compress_trigger_ratio
        return store.total_active_tokens() > threshold

    def compress(self, store: BlockStore, current_turn: int) -> Optional[ContextBlock]:
        compressible = sorted(
            [b for b in store.active_blocks()
             if b.status == BlockStatus.MICRO_COMPRESSED
             and b.layer in (Layer.L1_WORKING, Layer.L2_REFERENCE)
             and (current_turn - b.turn_id) > self.config.l1_window_turns // 2],
            key=lambda b: b.turn_id)

        if not compressible: return None

        current = store.total_active_tokens()
        target = int(self.config.total_token_budget * self.config.auto_compress_target_ratio)
        tokens_to_free = current - target

        selected, freed = [], 0
        for b in compressible:
            if freed >= tokens_to_free: break
            selected.append(b); freed += b.token_count

        if not selected: return None

        # Generate summary via LLM
        target_tokens = max(200, int(sum(b.token_count for b in selected) * 0.20))
        blocks_text = "\n\n---\n\n".join(
            f"[Turn {b.turn_id}] ({b.block_type.value})\n{b.working_content}"
            for b in selected)

        prompt = f"""Compress this conversation segment into a structured YAML summary.
Preserve: key facts with numbers, decisions with reasoning, current state.
Discard: raw data, intermediate exploration, rejected approaches.
Target: ~{target_tokens} tokens.

Conversation (turns {selected[0].turn_id}-{selected[-1].turn_id}):
{blocks_text}"""

        response = self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=target_tokens)
        summary_content = f"[Summary: turns {selected[0].turn_id}-{selected[-1].turn_id}]\n{response}"

        for b in selected: b.status = BlockStatus.SUMMARIZED

        summary = ContextBlock(
            id=f"summary_{selected[0].turn_id}_to_{selected[-1].turn_id}",
            turn_id=selected[0].turn_id, block_type=BlockType.SUMMARY,
            layer=Layer.L2_REFERENCE, status=BlockStatus.ACTIVE,
            original_content=summary_content, working_content=summary_content,
            token_count=count_tokens(summary_content))
        store.add(summary)
        return summary
```

### 3.8 Pipeline Stage 4: Context Editor

Agent self-edits context via tool calls. Highest impact on quality.

**Tool definitions:**

```python
CONTEXT_EDITING_TOOLS = [
    {
        "name": "context_mark_obsolete",
        "description": (
            "Mark tool results from specified turns as obsolete. "
            "Use when: a newer query replaced an older one; "
            "the user changed direction; an exploration was abandoned."),
        "input_schema": {
            "type": "object",
            "properties": {
                "turn_ids": {"type": "array", "items": {"type": "integer"},
                             "description": "Turns to mark obsolete"},
                "reason": {"type": "string", "description": "Brief reason"}
            },
            "required": ["turn_ids", "reason"]
        }
    },
    {
        "name": "context_compress_to_conclusion",
        "description": (
            "Replace a multi-turn analysis with its conclusion. "
            "Use when: multi-step analysis is complete; "
            "multiple approaches were tried and one selected; "
            "a tuning cycle finished."),
        "input_schema": {
            "type": "object",
            "properties": {
                "turn_ids": {"type": "array", "items": {"type": "integer"},
                             "description": "Turns to compress"},
                "conclusion": {"type": "string",
                               "description": "Final conclusion replacing the process"}
            },
            "required": ["turn_ids", "conclusion"]
        }
    },
    {
        "name": "context_pin",
        "description": (
            "Pin important context permanently. "
            "Use when: user stated a hard constraint; "
            "a key config is needed across many turns; "
            "an important decision must be preserved."),
        "input_schema": {
            "type": "object",
            "properties": {
                "turn_ids": {"type": "array", "items": {"type": "integer"},
                             "description": "Turns to pin"},
                "reason": {"type": "string", "description": "Why pinned"}
            },
            "required": ["turn_ids"]
        }
    },
]
```

**Executor:**

```python
class ContextEditor:
    def __init__(self, store: BlockStore):
        self.store = store
        self.edit_log: list[dict] = []

    def execute(self, tool_name: str, tool_input: dict) -> str:
        if tool_name == "context_mark_obsolete":
            turn_ids, reason = tool_input["turn_ids"], tool_input["reason"]
            freed = 0
            for tid in turn_ids:
                for block in self.store.get_blocks_by_turn(tid):
                    if block.status == BlockStatus.PINNED: continue
                    freed += block.token_count
                    block.status = BlockStatus.OBSOLETE
                    block.working_content = f"[obsolete: {reason}]"
                    block.token_count = count_tokens(block.working_content)
                    freed -= block.token_count
            self.edit_log.append({"action": "mark_obsolete", "turns": turn_ids, "freed": freed})
            return f"Marked turns {turn_ids} obsolete, freed ~{freed} tokens."

        elif tool_name == "context_compress_to_conclusion":
            turn_ids, conclusion = tool_input["turn_ids"], tool_input["conclusion"]
            freed = 0
            first_blocks = self.store.get_blocks_by_turn(turn_ids[0])
            if first_blocks:
                p = first_blocks[0]; freed += p.token_count
                p.working_content = f"[Compressed: turns {turn_ids[0]}-{turn_ids[-1]}]\nConclusion:\n{conclusion}"
                p.token_count = count_tokens(p.working_content)
                p.status = BlockStatus.MICRO_COMPRESSED; freed -= p.token_count
            for tid in turn_ids[1:]:
                for b in self.store.get_blocks_by_turn(tid):
                    freed += b.token_count; b.status = BlockStatus.SUMMARIZED
            self.edit_log.append({"action": "compress", "turns": turn_ids, "freed": freed})
            return f"Compressed turns {turn_ids[0]}-{turn_ids[-1]}, freed ~{freed} tokens."

        elif tool_name == "context_pin":
            turn_ids = tool_input["turn_ids"]
            pinned_tokens = 0
            for tid in turn_ids:
                for b in self.store.get_blocks_by_turn(tid):
                    b.status = BlockStatus.PINNED; b.layer = Layer.L0_CORE
                    pinned_tokens += b.token_count
            self.edit_log.append({"action": "pin", "turns": turn_ids, "tokens": pinned_tokens})
            return f"Pinned turns {turn_ids}, {pinned_tokens} tokens will always be retained."

        return f"Unknown operation: {tool_name}"
```

**System prompt section guiding agent when to self-edit:**

```python
CONTEXT_EDITING_PROMPT = """
## Context management

You have tools to keep conversation history lean and relevant.

### context_mark_obsolete — when to use
- A newer query replaced an older one's results
- User said "never mind", "skip", "let's switch to..."
- You re-read a file that was modified since last read

### context_compress_to_conclusion — when to use
- Multi-step analysis finished (tried groupings → selected best)
- Parameter tuning complete
- Model comparison done, winner chosen
- Signal phrases: "in summary", "the final answer", "after testing"

### context_pin — when to use
- User stated a hard constraint ("must use PostgreSQL", "budget < 100K")
- Configuration referenced across many turns
- Critical decision with reasoning that must persist

### Rules
- After completing related tool calls, consider: can I compress?
- When uncertain, do NOT edit — keeping is safer than deleting
- These tools are internal optimization, invisible to the user
"""
```

### 3.9 Pipeline Stage 5: Budget Assembler

Fills layers by priority, outputs final `messages[]` for LLM API.

```python
class BudgetAssembler:
    def __init__(self, config: ContextConfig):
        self.config = config

    def assemble(self, blocks: list[ContextBlock]) -> tuple[list[dict], dict]:
        budgets = self.config.layer_budgets
        selected, used = [], {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
        active = [b for b in blocks
                  if b.status not in (BlockStatus.OBSOLETE, BlockStatus.SUMMARIZED)]

        # L0: always all
        for b in active:
            if b.layer == Layer.L0_CORE:
                selected.append(b); used["L0"] += b.token_count

        # L1: newest first
        l1 = sorted([b for b in active if b.layer == Layer.L1_WORKING],
                     key=lambda b: b.turn_id, reverse=True)
        for b in l1:
            if used["L1"] + b.token_count <= budgets["L1"]:
                selected.append(b); used["L1"] += b.token_count

        # L2: chronological
        l2 = sorted([b for b in active if b.layer == Layer.L2_REFERENCE],
                     key=lambda b: b.turn_id)
        for b in l2:
            if used["L2"] + b.token_count <= budgets["L2"]:
                selected.append(b); used["L2"] += b.token_count

        # L3: if budget remains
        remaining = self.config.total_token_budget - sum(used.values())
        for b in sorted([b for b in active if b.layer == Layer.L3_ARCHIVE],
                        key=lambda b: b.turn_id):
            if remaining >= b.token_count:
                selected.append(b); remaining -= b.token_count; used["L3"] += b.token_count

        selected.sort(key=lambda b: (b.turn_id, self._type_order(b)))
        messages = self._to_messages(selected)
        if self.config.enable_prefix_caching:
            messages = self._optimize_for_caching(messages)
        return messages, used

    def _type_order(self, b):
        return {BlockType.SYSTEM: 0, BlockType.USER_INTENT: 1, BlockType.SUMMARY: 2,
                BlockType.USER_MESSAGE: 3, BlockType.TOOL_CALL: 4,
                BlockType.ASSISTANT_REPLY: 5, BlockType.PINNED: 6}.get(b.block_type, 99)

    def _to_messages(self, blocks):
        messages = []
        for b in blocks:
            if b.block_type == BlockType.SYSTEM:
                messages.append({"role": "system", "content": b.working_content})
            elif b.block_type in (BlockType.USER_MESSAGE, BlockType.USER_INTENT):
                messages.append({"role": "user", "content": b.working_content})
            elif b.block_type == BlockType.ASSISTANT_REPLY:
                messages.append({"role": "assistant", "content": b.working_content})
            elif b.block_type == BlockType.TOOL_CALL:
                messages.append({"role": "assistant", "content": None,
                    "tool_use": {"id": b.id, "name": b.tool_name, "input": b.tool_input_summary}})
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "content": b.working_content}]})
            elif b.block_type == BlockType.SUMMARY:
                messages.append({"role": "user",
                    "content": f"[System — conversation summary]\n{b.working_content}"})
        return messages

    def _optimize_for_caching(self, messages):
        """Stable content first → maximize prompt cache hit rate."""
        stable, volatile = [], []
        for msg in messages:
            c = msg.get("content", "") or ""
            if msg["role"] == "system": stable.insert(0, msg)
            elif "[Pinned" in c or "[Summary" in c or "[conversation summary]" in c:
                stable.append(msg)
            else: volatile.append(msg)
        return stable + volatile
```

### 3.10 Performance Expectations

| Metric | No optimization | claude_code | hybrid | full_layering |
|---|---|---|---|---|
| Final context (100 turns) | ~200K tokens | ~45K | ~20K | ~25K |
| Cumulative input tokens | ~2.1M | ~800K | ~450K | ~500K |
| Token savings | — | ~55% | ~80% | ~75% |
| Extra LLM calls | 0 | 3-5 | 3-5 | 3-5 |
| Quality at turn 100 | Degraded | Stable | Best | Good |

---

## 4. Sub-Agent Architecture

### 4.1 AgentConfig

```python
@dataclass
class AgentConfig:
    # Identity
    agent_id: str = "main"
    name: str = "Assistant"
    description: str = ""

    # LLM
    model: Optional[str] = None   # None = use session default
    system_prompt: str = ""

    # Tools (list of tool names or tool definition dicts)
    tools: list = field(default_factory=list)

    # Context management
    context_config: Optional[ContextConfig] = None  # None = hybrid_mode

    # Delegation
    can_delegate: bool = False
    max_delegation_depth: int = 1   # Max nesting. 1 = can delegate but subs cannot.

    # Limits
    max_turns: int = 200             # Meaningful for main agent (multi-turn with user)
    max_steps_per_turn: int = 30     # Safety limit on tool calls per turn
    max_steps_total: int = 50        # Meaningful for sub-agents (single-task, multi-step)

    # SharedState
    shared_state: Optional["SharedState"] = None
    state_access: str = "readwrite"  # "read" | "write" | "readwrite" | "none"
```

### 4.2 Agent Class

```python
class Agent:
    """
    Universal agent class. Same code for main, sub-agent, orchestrator, worker.
    Behavior differences come entirely from AgentConfig.
    """

    def __init__(self, llm_client, agent_config: AgentConfig):
        self.config = agent_config
        self.llm = llm_client
        self.agent_id = agent_config.agent_id

        # Context pipeline
        ctx = agent_config.context_config or ContextConfig.hybrid_mode()
        self.ctx_config = ctx
        self.store = BlockStore()
        self.classifier = LayerClassifier(ctx)
        self.micro = MicroCompressor(ctx)
        self.auto = AutoCompressor(ctx, llm_client)
        self.editor = ContextEditor(self.store)
        self.assembler = BudgetAssembler(ctx)

        # SharedState
        self.shared_state = agent_config.shared_state

        # Delegation
        self._sub_agents: dict[str, AgentConfig] = {}
        self._delegation_depth: int = 0

        self.current_turn = 0
        self._init_system_block()

    def register_sub_agent(self, name: str, config: AgentConfig):
        """Register a sub-agent. Adds delegate_to_agent to this agent's tools."""
        self._sub_agents[name] = config
        self._rebuild_system_block()

    def chat(self, user_message: str) -> str:
        """
        Main entry point.
        One turn: user_message → N steps (tool calls) → final reply.
        """
        self.current_turn += 1

        # Record user message
        bt = BlockType.USER_INTENT if self.current_turn == 1 else BlockType.USER_MESSAGE
        self.store.add(ContextBlock(
            id=f"turn_{self.current_turn:03d}_user", turn_id=self.current_turn,
            block_type=bt, layer=Layer.L1_WORKING,
            original_content=user_message, working_content=user_message,
            token_count=count_tokens(user_message)))

        # Pipeline stages 1-3
        self.classifier.classify(self.store.all(), self.current_turn)
        self.micro.compress(self.store.all(), self.current_turn)
        if self.auto.should_trigger(self.store):
            self.auto.compress(self.store, self.current_turn)

        # Dynamic mode upgrade
        usage = self.store.total_active_tokens() / self.ctx_config.total_token_budget
        self._maybe_upgrade_mode(usage)

        # Refresh shared state in system prompt
        if self.shared_state:
            self._rebuild_system_block()

        # LLM loop
        all_tools = self._get_all_tools()
        step_count = 0

        while step_count < self.config.max_steps_per_turn:
            messages, _ = self.assembler.assemble(self.store.all())
            response = self.llm.complete(messages=messages, tools=all_tools)

            if response.stop_reason == "tool_use":
                step_count += 1
                self._dispatch_tool(response.tool_use)
                continue
            else:
                self._record_reply(response.text)
                return response.text

        return "[max steps reached]"

    def _dispatch_tool(self, tool):
        name, input_data = tool.name, tool.input

        if name.startswith("context_"):
            result = self.editor.execute(name, input_data)
            # Context edits don't create a tool_call block — they modify existing blocks
            return result

        if name == "delegate_to_agent":
            result = self._handle_delegation(input_data)
            self._record_tool_block(tool, result)
            return result

        if name.startswith("state_"):
            result = self._handle_state_tool(name, input_data)
            return result

        # Business tool
        result = self._execute_business_tool(tool)
        self._record_tool_block(tool, result)
        return result

    def _handle_delegation(self, input_data: dict) -> str:
        """Spawn sub-agent, run to completion, return result."""
        agent_name = input_data["agent_name"]
        task = input_data["task"]

        if self._delegation_depth >= self.config.max_delegation_depth:
            return f"[Delegation blocked] Max depth ({self.config.max_delegation_depth}) reached."

        sub_config = self._sub_agents.get(agent_name)
        if not sub_config:
            return f"[Delegation failed] Unknown agent: {agent_name}"

        # Prepare sub-agent
        import copy
        cfg = copy.deepcopy(sub_config)
        cfg.context_config = cfg.context_config or ContextConfig.claude_code_mode()
        cfg.shared_state = self.shared_state
        if self._delegation_depth >= self.config.max_delegation_depth - 1:
            cfg.can_delegate = False

        sub = Agent(llm_client=self.llm, agent_config=cfg)
        sub._delegation_depth = self._delegation_depth + 1

        try:
            result = sub.chat(task)
        except Exception as e:
            result = f"[Sub-agent error] {agent_name}: {e}"

        return (f"[Delegation result from {agent_name}]\n"
                f"Task: {task[:200]}{'...' if len(task)>200 else ''}\n"
                f"Steps: {sub.current_turn}, "
                f"Tool calls: {len([b for b in sub.store.all() if b.block_type==BlockType.TOOL_CALL])}\n"
                f"\nResult:\n{result}")

    # ... (state tools, system block building, business tool execution)
```

### 4.3 Context Management × Agent Role

| Role | Context config | Reason |
|---|---|---|
| Main agent (default) | `hybrid_mode` | Good default; degrades to claude_code for short conversations |
| Sub-agent / worker | `claude_code_mode` | Single task, ~20 steps, no layering needed |
| Orchestrator | `full_layering_mode` | L2 budget (35%) stores worker summaries for synthesis |

### 4.4 Sub-Agent Execution Model

```
Sub-agent = single task, multi-step

  1 task (delegation input) → never changes, no user interaction
  N steps (tool calls) → agent loops internally, full context pipeline runs
  1 result (final reply) → returned to parent as tool_result

Key constraint: sub-agent CANNOT ask parent for clarification.
The task description must be self-contained.
Limit: max_steps_total (not max_turns — only 1 turn exists).
```

---

## 5. SharedState

Cross-agent structured data store. Complements context pipeline (which manages
conversational history) with data artifacts (DataFrames, configs, file paths).

```python
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class StateEntry:
    key: str
    value: Any
    written_by: str
    written_at: datetime
    schema_hint: str = ""
    ttl_turns: Optional[int] = None


class SharedState:
    def __init__(self):
        self._store: dict[str, StateEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            e = self._store.get(key)
            return e.value if e else None

    def set(self, key: str, value: Any, agent_id: str,
            schema_hint: str = "", ttl_turns: Optional[int] = None):
        with self._lock:
            self._store[key] = StateEntry(
                key=key, value=value, written_by=agent_id,
                written_at=datetime.now(), schema_hint=schema_hint,
                ttl_turns=ttl_turns)

    def keys(self, prefix: str = "") -> list[str]:
        with self._lock:
            return [k for k in self._store if k.startswith(prefix)]

    def snapshot(self) -> dict[str, str]:
        """Token-efficient summary for LLM context. Keys + schemas only, no values."""
        with self._lock:
            return {k: f"{e.schema_hint} (by {e.written_by})"
                    for k, e in self._store.items()}
```

SharedState tools exposed to LLM: `state_read`, `state_write`, `state_list`.
Snapshot is injected into system prompt so LLM knows what data is available
without bloating context with actual values.

---

## 6. Session API

### 6.1 Two Entry Points

```python
class Session:
    def __init__(self, agent: Agent, shared_state: SharedState):
        self.agent = agent
        self.shared_state = shared_state

    def chat(self, message: str) -> str:
        return self.agent.chat(message)

    def get_context_status(self) -> dict:
        store = self.agent.store
        return {
            "turn": self.agent.current_turn,
            "active_tokens": store.total_active_tokens(),
            "budget_usage": store.total_active_tokens() / self.agent.ctx_config.total_token_budget,
            "layering_enabled": self.agent.ctx_config.layering_enabled,
            "blocks_per_layer": {
                layer.name: len([b for b in store.active_blocks() if b.layer == layer])
                for layer in Layer},
            "shared_state_keys": self.shared_state.keys(),
            "edit_log": self.agent.editor.edit_log[-10:],
        }

    @classmethod
    def create(
        cls,
        llm_client,
        system_prompt: str = "You are a helpful assistant.",
        tools: list = None,
        sub_agents: dict[str, AgentConfig] = None,
        context_mode: str = "hybrid",
        token_budget: int = 80_000,
    ) -> "Session":
        """
        Primary entry point. Covers single-agent and delegation.
        
        If sub_agents is provided, the agent gets delegate_to_agent in its tool list.
        The LLM decides per-task whether to delegate or handle directly.
        """
        state = SharedState()
        cfg_factory = {
            "claude_code": ContextConfig.claude_code_mode,
            "hybrid": ContextConfig.hybrid_mode,
            "full_layering": ContextConfig.full_layering_mode,
        }
        ctx = cfg_factory.get(context_mode, ContextConfig.hybrid_mode)(token_budget)

        agent_config = AgentConfig(
            agent_id="main", system_prompt=system_prompt,
            tools=tools or [], context_config=ctx,
            can_delegate=bool(sub_agents), shared_state=state)
        agent = Agent(llm_client=llm_client, agent_config=agent_config)

        for name, sub_cfg in (sub_agents or {}).items():
            sub_cfg.shared_state = state
            agent.register_sub_agent(name, sub_cfg)

        return cls(agent=agent, shared_state=state)

    @classmethod
    def orchestrator(
        cls,
        llm_client,
        workers: dict[str, AgentConfig],
        orchestrator_prompt: str = None,
        token_budget: int = 100_000,
    ) -> "Session":
        """
        Advanced entry point. Agent identity = planner/coordinator.
        
        Orchestrator primarily plans and delegates. Workers do the actual work.
        """
        state = SharedState()

        default_prompt = """You are an orchestrator agent. Your job is to:
1. Analyze the user's task and break it into subtasks
2. Delegate subtasks to specialized worker agents
3. Synthesize worker results into a coherent final answer

Planning: break into 2-5 subtasks. Each must be self-contained.
Efficiency: don't delegate trivial tasks — do them yourself.
Synthesis: after workers complete, use context_compress_to_conclusion, then produce unified answer."""

        orch_config = AgentConfig(
            agent_id="orchestrator", name="Orchestrator",
            system_prompt=orchestrator_prompt or default_prompt,
            context_config=ContextConfig.full_layering_mode(token_budget),
            can_delegate=True, max_delegation_depth=1,
            shared_state=state, state_access="readwrite")
        agent = Agent(llm_client=llm_client, agent_config=orch_config)

        for name, w_cfg in workers.items():
            w_cfg.shared_state = state
            w_cfg.context_config = w_cfg.context_config or ContextConfig.claude_code_mode()
            agent.register_sub_agent(name, w_cfg)

        return cls(agent=agent, shared_state=state)
```

### 6.2 Usage Examples

```python
# ── Simple: single agent, no delegation ──
session = Session.create(tools=[execute_sql, execute_code, read_file])
response = session.chat("Analyze the sales data in /data/orders.csv")


# ── With delegation: LLM decides when to use sub-agents ──
from presets import researcher, data_analyst, coder

session = Session.create(
    tools=[read_file, write_file],
    sub_agents={
        "research": researcher(),
        "analyst": data_analyst(),
        "coder": coder(),
    },
)
response = session.chat("Research industry benchmarks, then analyze our sales data against them")
# LLM may delegate "research" to researcher and "analyze" to analyst,
# or it may do everything itself if the task is simple enough.


# ── Orchestrator: planning-first coordinator ──
from presets import researcher, data_analyst, writer

session = Session.orchestrator(
    workers={
        "researcher": researcher(),
        "analyst": data_analyst(),
        "writer": writer(),
    },
)
response = session.chat("Create a competitive analysis report with data backing")
```

---

## 7. Preset Agent Templates

Presets are factory functions returning `AgentConfig`. No special classes — just
carefully crafted system prompts + tool lists + context configs.

### 7.1 researcher

```python
def researcher(
    model: Optional[str] = None,
    max_steps_total: int = 20,
    extra_tools: list = None,
) -> AgentConfig:
    """
    Search → read → cross-reference → synthesize.
    
    Use for: "Find out about X", "Research competitors", "Gather benchmarks".
    Tools: web_search, read_file, read_url
    Output: structured summary with sources and confidence levels.
    """
    return AgentConfig(
        agent_id="researcher", name="Researcher",
        description=(
            "Searches web and documents to gather, cross-reference, "
            "and synthesize information. Returns structured summaries with sources."),
        model=model,
        system_prompt="""You are a research specialist. Gather information and produce a well-sourced summary.

## Workflow
1. Break the research question into 2-3 specific search queries (different angles)
2. Search and read the most relevant results for each
3. Cross-reference facts — note where sources agree and disagree
4. Synthesize into structured summary

## Output format
### Key findings
- Finding 1 (source: ...)
- Finding 2 (source: ...)

### Details
[Organized by theme, not by source]

### Confidence & gaps
- High confidence: [well-sourced facts]
- Uncertain: [conflicting sources]
- Not found: [gaps]

## Rules
- Never present single-source claims as established facts
- Prefer recent sources; state date ranges
- If results are insufficient, say so rather than speculating""",
        tools=["web_search", "read_file", "read_url"] + (extra_tools or []),
        context_config=ContextConfig.claude_code_mode(),
        max_steps_total=max_steps_total,
        state_access="readwrite",
    )
```

### 7.2 data_analyst

```python
def data_analyst(
    model: Optional[str] = None,
    max_steps_total: int = 30,
    extra_tools: list = None,
) -> AgentConfig:
    """
    Explore → query → analyze → visualize → conclude.
    
    Use for: "Analyze this dataset", "Find trends", "Build a chart".
    Tools: execute_sql, execute_code, read_file
    Output: conclusions with specific numbers, chart paths.
    """
    return AgentConfig(
        agent_id="data_analyst", name="Data Analyst",
        description=(
            "Expert at SQL, pandas, statistical analysis, and visualization. "
            "Produces conclusions backed by specific numbers."),
        model=model,
        system_prompt="""You are a senior data analyst. Analyze data and produce actionable insights.

## Workflow
1. Understand: schema, row counts, date ranges
2. Explore: descriptive stats, distributions, missing values
3. Analyze: answer the question with appropriate methods
4. Validate: sanity-check, look for anomalies
5. Conclude: specific numbers, not vague statements

## Output rules
Every claim MUST have a specific number.

Good: "Q4 revenue was ¥26.3M (+18% QoQ), driven by electronics (38% share).
       November peaked at ¥9.8M (Double 11). Caveat: 3.2% missing category data."
Bad:  "Revenue showed strong growth in Q4 with electronics leading."

Always state sample size and date range.
Use context_mark_obsolete after exploratory queries — only conclusions matter.
Write reusable DataFrames to SharedState for other agents.
Save charts to file paths and report the paths.""",
        tools=["execute_sql", "execute_code", "read_file"] + (extra_tools or []),
        context_config=ContextConfig.claude_code_mode(),
        max_steps_total=max_steps_total,
        state_access="readwrite",
    )
```

### 7.3 coder

```python
def coder(
    model: Optional[str] = None,
    max_steps_total: int = 40,
    extra_tools: list = None,
) -> AgentConfig:
    """
    Read → plan → implement → test → iterate.
    
    Use for: "Implement feature X", "Fix bug Y", "Refactor Z".
    Tools: read_file, write_file, execute_code, list_files
    Output: changed files list + test results.
    """
    return AgentConfig(
        agent_id="coder", name="Coder",
        description=(
            "Reads, writes, and tests code. Implements features, fixes bugs, "
            "refactors, writes tests. Returns change summary + test results."),
        model=model,
        system_prompt="""You are a senior software engineer. Implement code changes correctly.

## Workflow
1. Read relevant files — understand before changing
2. Plan: list files to modify and why
3. Implement incrementally: one logical change at a time
4. Test after each change
5. If tests fail after 3 attempts, stop and report

## Output format
- **Changes**: file list with one-line descriptions
- **Tests**: pass/fail, coverage
- **Notes**: breaking changes, TODOs

## Rules
- Always read before modifying
- Run existing tests after changes
- Match existing code style
- Use context_compress_to_conclusion after read-plan-implement cycles
- No debug prints in final code""",
        tools=["read_file", "write_file", "execute_code", "list_files"] + (extra_tools or []),
        context_config=ContextConfig.claude_code_mode(),
        max_steps_total=max_steps_total,
        state_access="readwrite",
    )
```

### 7.4 writer

```python
def writer(
    model: Optional[str] = None,
    max_steps_total: int = 15,
    extra_tools: list = None,
) -> AgentConfig:
    """
    Gather inputs → outline → draft → write to file.
    
    Use for: "Write a report", "Draft an email", "Create documentation".
    Tools: read_file, write_file
    Output: document file path + 2-sentence summary.
    """
    return AgentConfig(
        agent_id="writer", name="Writer",
        description=(
            "Produces well-structured documents: reports, emails, documentation. "
            "Reads source material and SharedState data as input."),
        model=model,
        system_prompt="""You are a technical writer. Produce clear, well-structured documents.

## Workflow
1. Read all inputs: task, SharedState data, referenced files
2. Create mental outline
3. Write complete document
4. Save to requested file format

## Principles
- Lead with conclusion / most important information
- Use specific numbers from source data — never invent statistics
- Short paragraphs (3-5 sentences)
- Headings for scannability

## Rules
- Never fabricate data
- Note gaps rather than filling with fluff
- Appropriate length: exec summary = 1 page, report = 3-5 pages, email = 3 paragraphs""",
        tools=["read_file", "write_file"] + (extra_tools or []),
        context_config=ContextConfig.claude_code_mode(),
        max_steps_total=max_steps_total,
        state_access="read",
    )
```

### 7.5 code_reviewer

```python
def code_reviewer(
    model: Optional[str] = None,
    max_steps_total: int = 20,
    extra_tools: list = None,
) -> AgentConfig:
    """
    Read → analyze → produce structured review.
    
    Use for: "Review this PR", "Check for security issues", "Find bugs".
    Tools: read_file, list_files, execute_code (read-only)
    Output: structured review with severity levels + verdict.
    """
    return AgentConfig(
        agent_id="code_reviewer", name="Code Reviewer",
        description=(
            "Reviews code for bugs, security vulnerabilities, performance issues. "
            "Read-only — never modifies files. Returns structured review."),
        model=model,
        system_prompt="""You are a senior code reviewer. Find issues and suggest improvements.

## Workflow
1. Read target files
2. Read related files for context (imports, interfaces, tests)
3. Analyze: correctness → security → performance → style
4. Produce structured review

## Output format per finding
- **Severity**: critical / warning / suggestion / nitpick
- **Location**: file:line or function name
- **Issue**: one sentence
- **Fix**: concrete suggestion

End with:
- **Verdict**: approve / request changes / needs discussion
- **Summary**: 2-3 sentence overall assessment

## Rules
- Be constructive — suggest fixes, don't just complain
- Distinguish real bugs (critical) from preferences (nitpick)
- If code is good, say so
- NEVER modify files — you are read-only""",
        tools=["read_file", "list_files", "execute_code"] + (extra_tools or []),
        context_config=ContextConfig.claude_code_mode(),
        max_steps_total=max_steps_total,
        state_access="read",
    )
```

### 7.6 Preset Registry

```python
PRESET_REGISTRY = {
    "researcher": researcher,
    "data_analyst": data_analyst,
    "coder": coder,
    "writer": writer,
    "code_reviewer": code_reviewer,
}

def get_preset(name: str, **overrides) -> AgentConfig:
    factory = PRESET_REGISTRY.get(name)
    if not factory:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESET_REGISTRY.keys())}")
    return factory(**overrides)

def list_presets() -> dict[str, str]:
    return {name: f.__doc__.strip().split('\n')[0]
            for name, f in PRESET_REGISTRY.items()}
```

---

## 8. CLI Design

### 8.1 Commands

```bash
# Default: single agent, hybrid context mode
$ myagent chat

# With sub-agents from YAML config
$ myagent chat --agents agents.yaml

# Orchestrator mode
$ myagent orchestrate agents.yaml

# Context mode override
$ myagent chat --context-mode claude_code
$ myagent chat --context-mode full_layering

# Token budget override
$ myagent chat --token-budget 120000
```

### 8.2 YAML Config

```yaml
# agents.yaml

sub_agents:
  # Use preset as-is
  research:
    preset: researcher

  # Preset with overrides
  analyst:
    preset: data_analyst
    model: claude-haiku             # cheaper model
    max_steps_total: 20             # tighter limit

  # Preset with extra tools
  coder:
    preset: coder
    extra_tools:
      - name: run_tests
        description: "Run project test suite"

  # Fully custom (no preset)
  domain_expert:
    name: "Domain Expert"
    description: "E-commerce logistics specialist"
    system_prompt: |
      You are an expert in e-commerce logistics.
      Use the knowledge base to look up company-specific information.
    tools:
      - knowledge_base_search
      - read_file
    max_steps_total: 10
```

### 8.3 Defaults

| Setting | Default | Reason |
|---|---|---|
| Agent mode | `Session.create()` | Single agent covers 90% of tasks |
| Context mode | `hybrid_mode` | Degrades to claude_code for short conversations; scales for long ones |
| Token budget | 80,000 | Fits within Claude/GPT-4 context windows with room for output |
| Sub-agents | None | Opt-in via config file |

---

## 9. Appendices

### 9.1 Token Counting Utility

```python
def count_tokens(text: str) -> int:
    if not text: return 0
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    except (ImportError, KeyError):
        pass
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
              or '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')
    return int((len(text) - cjk) / 4 + cjk / 1.5)
```

### 9.2 Block Lifecycle Example

```
Turn 12  │ CREATED          │ L1 active  │ 8,200 tokens │ Full 200-row JSON
Turn 13  │ LLM consumed     │ L1 active  │ 8,200 tokens │ Conclusions in assistant reply
Turn 16  │ MICRO compressed │ L1 micro   │ 1,500 tokens │ Schema + 10 rows
Turn 21  │ Layer demoted    │ L2 micro   │   200 tokens │ Schema + stats only
Turn 25  │ AGENT edited     │ — obsolete │    30 tokens │ "[obsolete: replaced by new query]"
Turn 40+ │ Excluded         │ — removed  │     0 tokens │ Not in messages[], original preserved
```

### 9.3 Context × Sub-Agent Matrix

| Aspect | Main agent | Sub-agent | Orchestrator |
|---|---|---|---|
| Context config | hybrid (default) | claude_code | full_layering |
| BlockStore | Persistent | Ephemeral (discarded after run) | Persistent |
| SharedState | Shared reference | Shared reference | Shared reference |
| Micro compression | Full pipeline | Full pipeline | Full pipeline |
| Auto compression | Active | Usually unnecessary (short-lived) | Active |
| Context editing | Active | Can self-edit | Active (especially after synthesis) |
| Delegation result | N/A | Returned as tool_result block | Returned as tool_result block |

### 9.4 Custom Tool Strategy Registration

```python
class MyApiStrategy(ToolCompressStrategy):
    def compress(self, content, tool_name, max_tokens, level):
        if level == "light":
            return content[:max_tokens * 4]
        return f"[{tool_name} summary] {len(content)} chars"

agent.micro.strategies["my_custom_api"] = MyApiStrategy()
```

### 9.5 Monitoring

```python
status = session.get_context_status()
# {
#   "turn": 45,
#   "active_tokens": 32000,
#   "budget_usage": 0.40,
#   "layering_enabled": True,
#   "blocks_per_layer": {"L0_CORE": 3, "L1_WORKING": 18, "L2_REFERENCE": 5, "L3_ARCHIVE": 0},
#   "shared_state_keys": ["analysis/monthly_revenue", "models/churn_predictor"],
#   "edit_log": [{"action": "mark_obsolete", "turns": [12,13], "freed": 3200}, ...]
# }
```
