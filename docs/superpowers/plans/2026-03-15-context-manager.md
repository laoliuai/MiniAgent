# Context Manager Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic `_summarize_messages()` in Agent with a 5-stage context management pipeline that reduces token usage by ~80% in 100-turn conversations.

**Architecture:** Independent `mini_agent/context/` package with ContextManager facade. Agent delegates all context tracking to ContextManager via composition. Pipeline: LayerClassifier → MicroCompressor → AutoCompressor → ContextEditor → BudgetAssembler.

**Tech Stack:** Python 3.10+, Pydantic BaseModel (config), tiktoken (token counting), pytest + asyncio (testing)

**Spec:** `docs/superpowers/specs/2026-03-15-context-manager-implementation-design.md`
**Original Design:** `docs/superpowers/specs/2026-03-15-agent-context-manager-final-design.md`

---

## Chunk 1: Data Model + BlockStore + Config + TokenCounter

### Task 1: Create `mini_agent/context/` package with data models

**Files:**
- Create: `mini_agent/context/__init__.py`
- Create: `mini_agent/context/models.py`
- Test: `tests/test_context_models.py`

- [ ] **Step 1: Write failing tests for data models**

```python
# tests/test_context_models.py
from mini_agent.context.models import (
    Layer, BlockType, BlockStatus, ContextBlock
)


def test_layer_ordering():
    assert Layer.L0_CORE.value < Layer.L1_WORKING.value
    assert Layer.L1_WORKING.value < Layer.L2_REFERENCE.value
    assert Layer.L2_REFERENCE.value < Layer.L3_ARCHIVE.value


def test_context_block_defaults():
    block = ContextBlock(
        id="turn_001_0_user",
        turn_id=1,
        block_type=BlockType.USER_MESSAGE,
        layer=Layer.L1_WORKING,
        original_content="hello",
        working_content="hello",
        token_count=1,
        original_token_count=1,
    )
    assert block.status == BlockStatus.ACTIVE
    assert block.depends_on == []
    assert block.tags == []
    assert block.compression_history == []
    assert block.superseded_by is None
    assert block.tool_name is None


def test_context_block_dual_track():
    block = ContextBlock(
        id="turn_005_0_execute_sql",
        turn_id=5,
        block_type=BlockType.TOOL_CALL,
        layer=Layer.L1_WORKING,
        original_content="full 8000 token output",
        working_content="compressed 200 token output",
        token_count=50,
        original_token_count=2000,
        tool_name="execute_sql",
        tool_input_summary='{"query": "SELECT ..."}',
    )
    assert block.original_content != block.working_content
    assert block.token_count < block.original_token_count
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_context_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mini_agent.context'`

- [ ] **Step 3: Implement data models**

```python
# mini_agent/context/__init__.py
"""Context management pipeline for MiniAgent."""

# mini_agent/context/models.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Layer(Enum):
    L0_CORE = 0
    L1_WORKING = 1
    L2_REFERENCE = 2
    L3_ARCHIVE = 3


class BlockType(Enum):
    SYSTEM = "system"
    USER_INTENT = "user_intent"
    USER_MESSAGE = "user_message"
    ASSISTANT_REPLY = "assistant"
    TOOL_CALL = "tool_call"
    SUMMARY = "summary"
    PINNED = "pinned"


class BlockStatus(Enum):
    ACTIVE = "active"
    MICRO_COMPRESSED = "micro"
    SUMMARIZED = "summarized"
    OBSOLETE = "obsolete"
    PINNED = "pinned"


@dataclass
class ContextBlock:
    id: str
    turn_id: int
    block_type: BlockType
    layer: Layer
    status: BlockStatus = BlockStatus.ACTIVE

    original_content: str = ""
    working_content: str = ""

    token_count: int = 0
    original_token_count: int = 0

    tool_name: Optional[str] = None
    tool_input_summary: str = ""
    tool_call_id: Optional[str] = None  # Original LLM-assigned tool_use ID

    depends_on: list[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    compression_history: list[dict] = field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_context_models.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add mini_agent/context/__init__.py mini_agent/context/models.py tests/test_context_models.py
git commit -m "feat(context): add data models — Layer, BlockType, BlockStatus, ContextBlock"
```

---

### Task 2: Implement BlockStore

**Files:**
- Create: `mini_agent/context/block_store.py`
- Test: `tests/test_block_store.py`

- [ ] **Step 1: Write failing tests for BlockStore**

```python
# tests/test_block_store.py
from mini_agent.context.block_store import BlockStore
from mini_agent.context.models import (
    ContextBlock, Layer, BlockType, BlockStatus
)


def _make_block(id: str, turn_id: int, block_type: BlockType = BlockType.TOOL_CALL,
                status: BlockStatus = BlockStatus.ACTIVE, token_count: int = 100) -> ContextBlock:
    return ContextBlock(
        id=id, turn_id=turn_id, block_type=block_type,
        layer=Layer.L1_WORKING, status=status,
        original_content="x", working_content="x",
        token_count=token_count, original_token_count=token_count,
    )


def test_add_and_get():
    store = BlockStore()
    block = _make_block("b1", 1)
    store.add(block)
    assert store.get("b1") is block
    assert store.get("nonexistent") is None


def test_get_blocks_by_turn():
    store = BlockStore()
    store.add(_make_block("b1", 1))
    store.add(_make_block("b2", 1))
    store.add(_make_block("b3", 2))
    assert len(store.get_blocks_by_turn(1)) == 2
    assert len(store.get_blocks_by_turn(2)) == 1
    assert len(store.get_blocks_by_turn(99)) == 0


def test_all_and_active_blocks():
    store = BlockStore()
    store.add(_make_block("b1", 1, status=BlockStatus.ACTIVE, token_count=100))
    store.add(_make_block("b2", 2, status=BlockStatus.OBSOLETE, token_count=200))
    store.add(_make_block("b3", 3, status=BlockStatus.SUMMARIZED, token_count=300))
    store.add(_make_block("b4", 4, status=BlockStatus.PINNED, token_count=50))
    assert len(store.all()) == 4
    active = store.active_blocks()
    assert len(active) == 2  # ACTIVE + PINNED
    assert {b.id for b in active} == {"b1", "b4"}


def test_total_active_tokens():
    store = BlockStore()
    store.add(_make_block("b1", 1, token_count=100))
    store.add(_make_block("b2", 2, status=BlockStatus.OBSOLETE, token_count=500))
    store.add(_make_block("b3", 3, token_count=200))
    assert store.total_active_tokens() == 300


def test_remove():
    store = BlockStore()
    store.add(_make_block("b1", 1))
    store.remove("b1")
    assert store.get("b1") is None
    assert len(store.get_blocks_by_turn(1)) == 0
    store.remove("nonexistent")  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_block_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement BlockStore**

```python
# mini_agent/context/block_store.py
from typing import Optional
from .models import ContextBlock, BlockStatus


class BlockStore:
    """Central storage for all context blocks."""

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
        return [self._blocks[bid] for bid in ids if bid in self._blocks]

    def all(self) -> list[ContextBlock]:
        return list(self._blocks.values())

    def active_blocks(self) -> list[ContextBlock]:
        return [b for b in self._blocks.values()
                if b.status not in (BlockStatus.OBSOLETE, BlockStatus.SUMMARIZED)]

    def total_active_tokens(self) -> int:
        return sum(b.token_count for b in self.active_blocks())

    def remove(self, block_id: str):
        if block_id in self._blocks:
            block = self._blocks.pop(block_id)
            turn_ids = self._by_turn.get(block.turn_id, [])
            if block_id in turn_ids:
                turn_ids.remove(block_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_block_store.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add mini_agent/context/block_store.py tests/test_block_store.py
git commit -m "feat(context): add BlockStore with turn-based indexing and active filtering"
```

---

### Task 3: Implement token_counter

**Files:**
- Create: `mini_agent/context/token_counter.py`
- Test: `tests/test_token_counter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_token_counter.py
from mini_agent.context.token_counter import count_tokens


def test_count_tokens_basic():
    tokens = count_tokens("Hello, world!")
    assert tokens > 0
    assert isinstance(tokens, int)


def test_count_tokens_empty():
    assert count_tokens("") == 0
    assert count_tokens(None) == 0


def test_count_tokens_long_text():
    text = "word " * 1000
    tokens = count_tokens(text)
    # ~1000 words should be roughly 1000 tokens
    assert 800 < tokens < 1500


def test_count_tokens_cjk():
    text = "你好世界" * 100
    tokens = count_tokens(text)
    assert tokens > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_token_counter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement token_counter**

```python
# mini_agent/context/token_counter.py
from typing import Optional

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except (ImportError, Exception):
            _encoder = False  # Sentinel: fallback mode
    return _encoder


def count_tokens(text: Optional[str]) -> int:
    """Count tokens. Uses tiktoken if available, char heuristic as fallback."""
    if not text:
        return 0

    encoder = _get_encoder()
    if encoder and encoder is not False:
        return len(encoder.encode(text))

    # Fallback: ~4 chars/token ASCII, ~1.5 chars/token CJK
    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
                    or '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')
    ascii_chars = len(text) - cjk_chars
    return int(ascii_chars / 4 + cjk_chars / 1.5)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_token_counter.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add mini_agent/context/token_counter.py tests/test_token_counter.py
git commit -m "feat(context): add token_counter with tiktoken primary and char heuristic fallback"
```

---

### Task 4: Implement ContextConfig with presets

**Files:**
- Create: `mini_agent/context/config.py`
- Test: `tests/test_context_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_context_config.py
from mini_agent.context.config import ContextConfig


def test_default_is_hybrid():
    config = ContextConfig()
    assert config.mode == "hybrid"
    assert config.l1_window_turns == 8
    assert config.l2_window_turns == 30
    assert config.layering_enabled is True


def test_claude_code_mode():
    config = ContextConfig.from_mode("claude_code")
    assert config.l1_window_turns == 999_999
    assert config.l1_budget_ratio == 0.85
    assert config.l2_budget_ratio == 0.0
    assert config.layering_enabled is False


def test_hybrid_mode():
    config = ContextConfig.from_mode("hybrid")
    assert config.l1_window_turns == 8
    assert config.l1_budget_ratio == 0.40
    assert config.l2_budget_ratio == 0.30
    assert config.l3_budget_ratio == 0.15


def test_full_layering_mode():
    config = ContextConfig.from_mode("full_layering")
    assert config.l1_window_turns == 5
    assert config.l2_window_turns == 15
    assert config.l1_budget_ratio == 0.30


def test_layer_budgets_with_layering():
    config = ContextConfig.from_mode("hybrid", total_token_budget=100_000)
    budgets = config.layer_budgets
    assert budgets["L0"] == 15_000
    assert budgets["L1"] == 40_000
    assert budgets["L2"] == 30_000
    assert budgets["L3"] == 15_000


def test_layer_budgets_without_layering():
    config = ContextConfig.from_mode("claude_code", total_token_budget=100_000)
    budgets = config.layer_budgets
    assert budgets["L0"] == 15_000
    assert budgets["L1"] == 85_000
    assert budgets["L2"] == 0
    assert budgets["L3"] == 0


def test_from_mode_with_overrides():
    config = ContextConfig.from_mode("hybrid", total_token_budget=120_000,
                                     enable_context_editing=False)
    assert config.total_token_budget == 120_000
    assert config.enable_context_editing is False
    assert config.l1_window_turns == 8  # still hybrid defaults


def test_enable_context_editing_default_true():
    config = ContextConfig()
    assert config.enable_context_editing is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_context_config.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ContextConfig**

```python
# mini_agent/context/config.py
from typing import Optional
from pydantic import BaseModel, computed_field

_PRESETS = {
    "claude_code": {
        "l0_budget_ratio": 0.15,
        "l1_budget_ratio": 0.85,
        "l2_budget_ratio": 0.0,
        "l3_budget_ratio": 0.0,
        "l1_window_turns": 999_999,
        "l2_window_turns": 999_999,
    },
    "hybrid": {
        "l0_budget_ratio": 0.15,
        "l1_budget_ratio": 0.40,
        "l2_budget_ratio": 0.30,
        "l3_budget_ratio": 0.15,
        "l1_window_turns": 8,
        "l2_window_turns": 30,
    },
    "full_layering": {
        "l0_budget_ratio": 0.15,
        "l1_budget_ratio": 0.30,
        "l2_budget_ratio": 0.35,
        "l3_budget_ratio": 0.20,
        "l1_window_turns": 5,
        "l2_window_turns": 15,
    },
}


class ContextConfig(BaseModel):
    """Central configuration for the context manager."""

    mode: str = "hybrid"

    # Token Budget
    total_token_budget: int = 80_000

    # Layer Budget Ratios
    l0_budget_ratio: float = 0.15
    l1_budget_ratio: float = 0.40
    l2_budget_ratio: float = 0.30
    l3_budget_ratio: float = 0.15

    # Layer Windows
    l1_window_turns: int = 8
    l2_window_turns: int = 30

    # Micro Compression
    micro_compress_after_turns: int = 3
    tool_output_max_tokens_l1: int = 2000
    tool_output_max_tokens_l2: int = 200

    # Auto Compression
    auto_compress_trigger_ratio: float = 0.85
    auto_compress_target_ratio: float = 0.60
    auto_compress_model: Optional[str] = None

    # Context Editing
    enable_context_editing: bool = True

    # Prefix Caching
    enable_prefix_caching: bool = True

    @computed_field
    @property
    def layering_enabled(self) -> bool:
        return self.l1_window_turns < 1000

    @computed_field
    @property
    def layer_budgets(self) -> dict[str, int]:
        if not self.layering_enabled:
            return {
                "L0": int(self.total_token_budget * self.l0_budget_ratio),
                "L1": int(self.total_token_budget * (1.0 - self.l0_budget_ratio)),
                "L2": 0,
                "L3": 0,
            }
        return {
            "L0": int(self.total_token_budget * self.l0_budget_ratio),
            "L1": int(self.total_token_budget * self.l1_budget_ratio),
            "L2": int(self.total_token_budget * self.l2_budget_ratio),
            "L3": int(self.total_token_budget * self.l3_budget_ratio),
        }

    @classmethod
    def from_mode(cls, mode: str, **overrides) -> "ContextConfig":
        if mode not in _PRESETS:
            raise ValueError(f"Unknown mode: {mode}. Choose from: {list(_PRESETS.keys())}")
        defaults = {"mode": mode, **_PRESETS[mode]}
        defaults.update(overrides)
        return cls(**defaults)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_context_config.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add mini_agent/context/config.py tests/test_context_config.py
git commit -m "feat(context): add ContextConfig with hybrid/claude_code/full_layering presets"
```

---

## Chunk 2: LayerClassifier + BudgetAssembler (Minimal Pipeline)

### Task 5: Implement LayerClassifier

**Files:**
- Create: `mini_agent/context/classifier.py`
- Test: `tests/test_classifier.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_classifier.py
from mini_agent.context.classifier import LayerClassifier
from mini_agent.context.config import ContextConfig
from mini_agent.context.models import (
    ContextBlock, Layer, BlockType, BlockStatus
)


def _block(id: str, turn_id: int, block_type: BlockType = BlockType.TOOL_CALL,
           status: BlockStatus = BlockStatus.ACTIVE, depends_on: list[str] | None = None):
    return ContextBlock(
        id=id, turn_id=turn_id, block_type=block_type, layer=Layer.L1_WORKING,
        status=status, original_content="x", working_content="x",
        token_count=100, original_token_count=100,
        depends_on=depends_on or [],
    )


def test_system_and_user_intent_always_l0():
    config = ContextConfig.from_mode("hybrid")
    classifier = LayerClassifier(config)
    blocks = [
        _block("sys", 0, BlockType.SYSTEM),
        _block("intent", 1, BlockType.USER_INTENT),
    ]
    classifier.classify(blocks, current_turn=50)
    assert blocks[0].layer == Layer.L0_CORE
    assert blocks[1].layer == Layer.L0_CORE


def test_pinned_always_l0():
    config = ContextConfig.from_mode("hybrid")
    classifier = LayerClassifier(config)
    blocks = [_block("b1", 1, status=BlockStatus.PINNED)]
    classifier.classify(blocks, current_turn=50)
    assert blocks[0].layer == Layer.L0_CORE


def test_recent_blocks_stay_l1():
    config = ContextConfig.from_mode("hybrid")  # l1_window=8
    classifier = LayerClassifier(config)
    blocks = [_block("b1", 18)]  # age=2 from turn 20
    classifier.classify(blocks, current_turn=20)
    assert blocks[0].layer == Layer.L1_WORKING


def test_old_blocks_demote_to_l2_l3():
    config = ContextConfig.from_mode("hybrid")  # l1_window=8, l2_window=30
    classifier = LayerClassifier(config)
    blocks = [
        _block("b1", 5),   # age=25, within l2_window
        _block("b2", 1),   # age=29, within l2_window
        _block("b3", 0, block_type=BlockType.TOOL_CALL),  # age=30, exactly at boundary
    ]
    classifier.classify(blocks, current_turn=30)
    assert blocks[0].layer == Layer.L2_REFERENCE
    assert blocks[1].layer == Layer.L2_REFERENCE
    assert blocks[2].layer == Layer.L2_REFERENCE  # age == l2_window → still L2


def test_very_old_blocks_archive():
    config = ContextConfig.from_mode("hybrid")  # l2_window=30
    classifier = LayerClassifier(config)
    blocks = [_block("b1", 1)]  # age=49 > 30
    classifier.classify(blocks, current_turn=50)
    assert blocks[0].layer == Layer.L3_ARCHIVE


def test_claude_code_mode_all_l1():
    """With INF window, everything stays L1 (layering disabled)."""
    config = ContextConfig.from_mode("claude_code")
    classifier = LayerClassifier(config)
    blocks = [_block("b1", 1), _block("b2", 50)]
    classifier.classify(blocks, current_turn=100)
    assert all(b.layer == Layer.L1_WORKING for b in blocks)


def test_dependency_chain_promotion():
    """Old block depended on by recent block gets promoted back to L1."""
    config = ContextConfig.from_mode("hybrid")  # l1_window=8
    classifier = LayerClassifier(config)
    old_block = _block("old", 1)  # age=29, would normally be L2
    recent_block = _block("recent", 25, depends_on=["old"])  # age=5, L1
    blocks = [old_block, recent_block]
    classifier.classify(blocks, current_turn=30)
    assert old_block.layer == Layer.L1_WORKING  # promoted!
    assert recent_block.layer == Layer.L1_WORKING


def test_skip_obsolete_and_summarized():
    config = ContextConfig.from_mode("hybrid")
    classifier = LayerClassifier(config)
    blocks = [
        _block("b1", 1, status=BlockStatus.OBSOLETE),
        _block("b2", 2, status=BlockStatus.SUMMARIZED),
    ]
    original_layers = [b.layer for b in blocks]
    classifier.classify(blocks, current_turn=50)
    # Layers should not change for obsolete/summarized blocks
    assert [b.layer for b in blocks] == original_layers
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: FAIL

- [ ] **Step 3: Implement LayerClassifier**

```python
# mini_agent/context/classifier.py
from .config import ContextConfig
from .models import ContextBlock, Layer, BlockType, BlockStatus


class LayerClassifier:
    """Stage 1: Rules-based layer assignment. No LLM cost."""

    def __init__(self, config: ContextConfig):
        self.config = config

    def classify(self, blocks: list[ContextBlock], current_turn: int):
        for block in blocks:
            if block.status == BlockStatus.PINNED:
                block.layer = Layer.L0_CORE
                continue

            if block.status in (BlockStatus.OBSOLETE, BlockStatus.SUMMARIZED):
                continue

            if block.block_type in (BlockType.SYSTEM, BlockType.USER_INTENT):
                block.layer = Layer.L0_CORE
                continue

            age = current_turn - block.turn_id

            if age <= self.config.l1_window_turns:
                block.layer = Layer.L1_WORKING
            elif age <= self.config.l2_window_turns:
                block.layer = Layer.L2_REFERENCE
            else:
                block.layer = Layer.L3_ARCHIVE

        # Second pass: dependency chain promotion
        for block in blocks:
            if block.layer.value >= Layer.L2_REFERENCE.value:
                if self._is_in_active_chain(block, blocks, current_turn):
                    block.layer = Layer.L1_WORKING

    def _is_in_active_chain(self, block: ContextBlock,
                            all_blocks: list[ContextBlock],
                            current_turn: int) -> bool:
        for other in all_blocks:
            if (current_turn - other.turn_id) <= self.config.l1_window_turns:
                if block.id in other.depends_on:
                    return True
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add mini_agent/context/classifier.py tests/test_classifier.py
git commit -m "feat(context): add LayerClassifier with rules-based assignment and dependency promotion"
```

---

### Task 6: Implement BudgetAssembler

**Files:**
- Create: `mini_agent/context/budget_assembler.py`
- Test: `tests/test_budget_assembler.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_budget_assembler.py
from mini_agent.context.budget_assembler import BudgetAssembler
from mini_agent.context.config import ContextConfig
from mini_agent.context.models import (
    ContextBlock, Layer, BlockType, BlockStatus
)


def _block(id: str, turn_id: int, layer: Layer, block_type: BlockType = BlockType.TOOL_CALL,
           token_count: int = 100, content: str = "test content",
           status: BlockStatus = BlockStatus.ACTIVE,
           tool_name: str | None = None, tool_input_summary: str = ""):
    return ContextBlock(
        id=id, turn_id=turn_id, block_type=block_type, layer=layer,
        status=status, original_content=content, working_content=content,
        token_count=token_count, original_token_count=token_count,
        tool_name=tool_name, tool_input_summary=tool_input_summary,
    )


def test_l0_always_included():
    config = ContextConfig.from_mode("hybrid", total_token_budget=1000)
    assembler = BudgetAssembler(config)
    blocks = [
        _block("sys", 0, Layer.L0_CORE, BlockType.SYSTEM, content="system prompt"),
    ]
    messages, used = assembler.assemble(blocks)
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert used["L0"] > 0


def test_l1_newest_first():
    config = ContextConfig.from_mode("hybrid", total_token_budget=1000)
    assembler = BudgetAssembler(config)
    blocks = [
        _block("b1", 1, Layer.L1_WORKING, BlockType.USER_MESSAGE, token_count=300),
        _block("b2", 5, Layer.L1_WORKING, BlockType.USER_MESSAGE, token_count=300),
    ]
    # Both fit within L1 budget (400 tokens) — only one should fit
    messages, used = assembler.assemble(blocks)
    # Newest first: b2 gets picked, b1 may or may not fit
    assert any("test content" in (m.get("content", "") or "") for m in messages)


def test_obsolete_and_summarized_excluded():
    config = ContextConfig.from_mode("hybrid", total_token_budget=10000)
    assembler = BudgetAssembler(config)
    blocks = [
        _block("b1", 1, Layer.L1_WORKING, status=BlockStatus.OBSOLETE),
        _block("b2", 2, Layer.L1_WORKING, status=BlockStatus.SUMMARIZED),
        _block("b3", 3, Layer.L1_WORKING, BlockType.USER_MESSAGE),
    ]
    messages, used = assembler.assemble(blocks)
    assert len(messages) == 1  # Only b3


def test_tool_block_expansion():
    config = ContextConfig.from_mode("hybrid", total_token_budget=10000)
    assembler = BudgetAssembler(config)
    blocks = [
        _block("b1", 1, Layer.L1_WORKING, BlockType.TOOL_CALL,
               tool_name="read_file", tool_input_summary='{"path": "/tmp/f.py"}',
               content="file contents here"),
    ]
    messages, used = assembler.assemble(blocks)
    # Tool block expands to 2 messages: assistant/tool_use + user/tool_result
    assert len(messages) == 2
    assert messages[0]["role"] == "assistant"
    assert messages[1]["role"] == "user"


def test_summary_block_format():
    config = ContextConfig.from_mode("hybrid", total_token_budget=10000)
    assembler = BudgetAssembler(config)
    blocks = [
        _block("s1", 1, Layer.L2_REFERENCE, BlockType.SUMMARY, content="summary of turns 1-5"),
    ]
    messages, used = assembler.assemble(blocks)
    assert len(messages) == 1
    assert "[System — conversation summary]" in messages[0]["content"]


def test_messages_sorted_by_turn():
    config = ContextConfig.from_mode("hybrid", total_token_budget=10000)
    assembler = BudgetAssembler(config)
    blocks = [
        _block("sys", 0, Layer.L0_CORE, BlockType.SYSTEM, content="sys"),
        _block("u3", 3, Layer.L1_WORKING, BlockType.USER_MESSAGE, content="msg3"),
        _block("u1", 1, Layer.L1_WORKING, BlockType.USER_MESSAGE, content="msg1"),
    ]
    messages, used = assembler.assemble(blocks)
    # System first, then chronological
    assert messages[0]["role"] == "system"
    assert "msg1" in messages[1]["content"]
    assert "msg3" in messages[2]["content"]


def test_claude_code_mode_no_l2_l3():
    config = ContextConfig.from_mode("claude_code", total_token_budget=10000)
    assembler = BudgetAssembler(config)
    blocks = [
        _block("b1", 1, Layer.L2_REFERENCE, BlockType.SUMMARY, token_count=100),
    ]
    messages, used = assembler.assemble(blocks)
    # L2 budget is 0 in claude_code mode
    assert used["L2"] == 0
    assert len(messages) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_budget_assembler.py -v`
Expected: FAIL

- [ ] **Step 3: Implement BudgetAssembler**

```python
# mini_agent/context/budget_assembler.py
import json
from .config import ContextConfig
from .models import ContextBlock, Layer, BlockType, BlockStatus


class BudgetAssembler:
    """Stage 5: Fill layers by priority, produce final messages[]."""

    def __init__(self, config: ContextConfig):
        self.config = config

    def assemble(self, blocks: list[ContextBlock]) -> tuple[list[dict], dict[str, int]]:
        budgets = self.config.layer_budgets
        selected: list[ContextBlock] = []
        used = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}

        active = [b for b in blocks
                  if b.status not in (BlockStatus.OBSOLETE, BlockStatus.SUMMARIZED)]

        # L0: all core blocks
        for b in active:
            if b.layer == Layer.L0_CORE:
                selected.append(b)
                used["L0"] += b.token_count

        # L1: newest first
        l1 = sorted([b for b in active if b.layer == Layer.L1_WORKING],
                     key=lambda b: b.turn_id, reverse=True)
        for b in l1:
            if used["L1"] + b.token_count <= budgets["L1"]:
                selected.append(b)
                used["L1"] += b.token_count

        # L2: chronological
        l2 = sorted([b for b in active if b.layer == Layer.L2_REFERENCE],
                     key=lambda b: b.turn_id)
        for b in l2:
            if used["L2"] + b.token_count <= budgets["L2"]:
                selected.append(b)
                used["L2"] += b.token_count

        # L3: remaining budget
        remaining = self.config.total_token_budget - sum(used.values())
        if remaining > 0:
            l3 = sorted([b for b in active if b.layer == Layer.L3_ARCHIVE],
                        key=lambda b: b.turn_id)
            for b in l3:
                if remaining >= b.token_count:
                    selected.append(b)
                    remaining -= b.token_count
                    used["L3"] += b.token_count

        # Sort by turn_id, then type order
        selected.sort(key=lambda b: (b.turn_id, self._type_order(b)))
        messages = self._to_messages(selected)

        if self.config.enable_prefix_caching:
            messages = self._optimize_for_caching(messages)

        return messages, used

    def _type_order(self, block: ContextBlock) -> int:
        order = {
            BlockType.SYSTEM: 0, BlockType.USER_INTENT: 1, BlockType.SUMMARY: 2,
            BlockType.USER_MESSAGE: 3, BlockType.TOOL_CALL: 4,
            BlockType.ASSISTANT_REPLY: 5, BlockType.PINNED: 6,
        }
        return order.get(block.block_type, 99)

    def _to_messages(self, blocks: list[ContextBlock]) -> list[dict]:
        messages = []
        for block in blocks:
            if block.block_type == BlockType.SYSTEM:
                messages.append({"role": "system", "content": block.working_content})
            elif block.block_type in (BlockType.USER_MESSAGE, BlockType.USER_INTENT):
                messages.append({"role": "user", "content": block.working_content})
            elif block.block_type == BlockType.ASSISTANT_REPLY:
                messages.append({"role": "assistant", "content": block.working_content})
            elif block.block_type == BlockType.TOOL_CALL:
                messages.extend(self._expand_tool_block(block))
            elif block.block_type == BlockType.SUMMARY:
                messages.append({
                    "role": "user",
                    "content": f"[System — conversation summary]\n{block.working_content}",
                })
            elif block.block_type == BlockType.PINNED:
                messages.append({
                    "role": "user",
                    "content": f"[Pinned context]\n{block.working_content}",
                })
        return messages

    def _expand_tool_block(self, block: ContextBlock) -> list[dict]:
        tool_use_id = block.tool_call_id or block.id  # Use original LLM ID if available
        try:
            tool_input = json.loads(block.tool_input_summary)
        except (json.JSONDecodeError, TypeError):
            tool_input = {"_raw": block.tool_input_summary}

        return [
            {
                "role": "assistant",
                "content": None,
                "tool_use": {
                    "id": tool_use_id,
                    "name": block.tool_name,
                    "input": tool_input,
                },
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": block.working_content,
                    }
                ],
            },
        ]

    def _optimize_for_caching(self, messages: list[dict]) -> list[dict]:
        """Stable content first (system, pinned, summaries), volatile last."""
        stable, volatile = [], []
        for msg in messages:
            content = msg.get("content", "") or ""
            if isinstance(content, list):
                content = ""
            if msg["role"] == "system":
                stable.insert(0, msg)
            elif "[Pinned context]" in content or "[System — conversation summary]" in content:
                stable.append(msg)
            else:
                volatile.append(msg)
        return stable + volatile
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_budget_assembler.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add mini_agent/context/budget_assembler.py tests/test_budget_assembler.py
git commit -m "feat(context): add BudgetAssembler with layer-priority filling and prefix caching"
```

---

## Chunk 3: MicroCompressor + Tool Base Class Extension

### Task 7: Extend Tool base class with compress_strategy

**Files:**
- Modify: `mini_agent/tools/base.py:24-26` (add property between `parameters` and `execute`)
- Test: `tests/test_tool_schema.py` (verify existing tests still pass)

- [ ] **Step 1: Run existing tool tests to establish baseline**

Run: `uv run pytest tests/test_tool_schema.py -v`
Expected: PASS

- [ ] **Step 2: Add compress_strategy property to Tool base class**

In `mini_agent/tools/base.py`, add after the `parameters` property (line 24) and before `execute` (line 26):

```python
    @property
    def compress_strategy(self) -> "ToolCompressStrategy | None":
        """Override to provide a custom compression strategy for this tool's output.
        See mini_agent.context.compress_strategies for available strategies."""
        return None
```

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `uv run pytest tests/test_tool_schema.py -v`
Expected: PASS (no change in behavior)

- [ ] **Step 4: Commit**

```bash
git add mini_agent/tools/base.py
git commit -m "feat(tools): add compress_strategy property to Tool base class"
```

---

### Task 8: Implement compress strategies

**Files:**
- Create: `mini_agent/context/compress_strategies.py`
- Test: `tests/test_compress_strategies.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_compress_strategies.py
import json
from mini_agent.context.compress_strategies import (
    SqlResultStrategy,
    CodeOutputStrategy,
    FileReadStrategy,
    SearchResultStrategy,
    PassThroughStrategy,
    DefaultTruncateStrategy,
)


class TestSqlResultStrategy:
    def test_light_keeps_sample_rows(self):
        strategy = SqlResultStrategy()
        data = json.dumps({
            "columns": ["id", "name", "value"],
            "rows": [{"id": i, "name": f"item{i}", "value": i * 10} for i in range(100)]
        })
        result = strategy.compress(data, "execute_sql", 2000, "light")
        parsed = json.loads(result)
        assert parsed["total_rows"] == 100
        assert len(parsed["sample_rows"]) == 10
        assert "columns" in parsed

    def test_aggressive_stats_only(self):
        strategy = SqlResultStrategy()
        data = json.dumps({
            "rows": [{"id": i, "val": i * 10} for i in range(50)]
        })
        result = strategy.compress(data, "execute_sql", 200, "aggressive")
        assert "[SQL result summary]" in result
        assert "50 rows" in result

    def test_invalid_json_fallback(self):
        strategy = SqlResultStrategy()
        result = strategy.compress("not json", "execute_sql", 200, "light")
        # Should fallback to DefaultTruncateStrategy, not crash
        assert isinstance(result, str)


class TestCodeOutputStrategy:
    def test_light_head_tail(self):
        strategy = CodeOutputStrategy()
        lines = [f"line {i}" for i in range(100)]
        content = "\n".join(lines)
        result = strategy.compress(content, "execute_code", 2000, "light")
        assert "line 0" in result
        assert "line 99" in result
        assert "omitted" in result

    def test_aggressive_errors_and_tail(self):
        strategy = CodeOutputStrategy()
        lines = ["ok", "ok", "Error: something failed", "ok", "final"]
        content = "\n".join(lines)
        result = strategy.compress(content, "execute_code", 200, "aggressive")
        assert "Error: something failed" in result
        assert "final" in result


class TestFileReadStrategy:
    def test_light_keeps_head_and_structure(self):
        strategy = FileReadStrategy()
        content = "class Foo:\n    pass\n" + "\n".join(f"line {i}" for i in range(100))
        result = strategy.compress(content, "read_file", 2000, "light")
        assert "class Foo" in result
        assert "Structure:" in result

    def test_aggressive_structure_only(self):
        strategy = FileReadStrategy()
        content = "class Foo:\n    pass\ndef bar():\n    pass\n" + "x\n" * 200
        result = strategy.compress(content, "read_file", 200, "aggressive")
        assert "[file summary]" in result
        assert "Foo" in result


class TestSearchResultStrategy:
    def test_light_keeps_snippets(self):
        strategy = SearchResultStrategy()
        data = json.dumps([
            {"title": "Result 1", "url": "http://a.com", "snippet": "desc 1"},
            {"title": "Result 2", "url": "http://b.com", "snippet": "desc 2"},
        ])
        result = strategy.compress(data, "web_search", 2000, "light")
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["title"] == "Result 1"

    def test_aggressive_titles_only(self):
        strategy = SearchResultStrategy()
        data = json.dumps([{"title": "A"}, {"title": "B"}])
        result = strategy.compress(data, "web_search", 200, "aggressive")
        assert "[search summary]" in result


class TestPassThroughStrategy:
    def test_returns_unchanged(self):
        strategy = PassThroughStrategy()
        assert strategy.compress("hello", "write_file", 100, "aggressive") == "hello"


class TestDefaultTruncateStrategy:
    def test_short_content_unchanged(self):
        strategy = DefaultTruncateStrategy()
        assert strategy.compress("short", "x", 1000, "light") == "short"

    def test_long_content_truncated(self):
        strategy = DefaultTruncateStrategy()
        content = "x" * 10000
        result = strategy.compress(content, "x", 100, "light")  # max 400 chars
        assert len(result) < len(content)
        assert "omitted" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_compress_strategies.py -v`
Expected: FAIL

- [ ] **Step 3: Implement compress strategies**

```python
# mini_agent/context/compress_strategies.py
import json
import re
from abc import ABC, abstractmethod


class ToolCompressStrategy(ABC):
    @abstractmethod
    def compress(self, content: str, tool_name: str, max_tokens: int, level: str) -> str:
        ...


class SqlResultStrategy(ToolCompressStrategy):
    """SQL results: keep schema + stats, drop raw rows."""

    def compress(self, content: str, tool_name: str, max_tokens: int, level: str) -> str:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return DefaultTruncateStrategy().compress(content, tool_name, max_tokens, level)

        rows = data.get("rows", [])
        columns = data.get("columns", list(rows[0].keys()) if rows else [])

        if level == "light":
            return json.dumps({
                "columns": columns,
                "total_rows": len(rows),
                "sample_rows": rows[:10],
                "note": f"[micro-compressed] Original {len(rows)} rows, showing first 10",
            }, ensure_ascii=False)
        else:
            stats = self._column_stats(rows, columns)
            return (
                f"[SQL result summary] {len(rows)} rows, {len(columns)} columns\n"
                f"Columns: {', '.join(str(c) for c in columns)}\n"
                f"Stats: {json.dumps(stats, ensure_ascii=False)}"
            )

    def _column_stats(self, rows: list[dict], columns: list) -> dict:
        stats = {}
        for col in columns[:8]:
            values = [r.get(col) for r in rows if r.get(col) is not None]
            if not values:
                continue
            if isinstance(values[0], (int, float)):
                stats[col] = {
                    "min": min(values), "max": max(values),
                    "avg": round(sum(values) / len(values), 2),
                }
            elif isinstance(values[0], str):
                unique = set(values)
                stats[col] = {"unique": len(unique), "sample": list(unique)[:5]}
        return stats


class CodeOutputStrategy(ToolCompressStrategy):
    """Code execution: keep errors + tail output."""

    def compress(self, content: str, tool_name: str, max_tokens: int, level: str) -> str:
        lines = content.split("\n")
        errors = [l for l in lines
                  if any(k in l.lower() for k in ["error", "exception", "traceback"])]

        if level == "light":
            head, tail = lines[:20], lines[-20:] if len(lines) > 40 else []
            omitted = max(0, len(lines) - 40)
            result = "\n".join(head)
            if omitted > 0:
                result += f"\n\n[...{omitted} lines omitted...]\n\n" + "\n".join(tail)
            return result
        else:
            parts = [f"[exec result] {len(lines)} lines"]
            if errors:
                parts.append("Errors: " + "; ".join(errors[:3]))
            parts.append("Tail:\n" + "\n".join(lines[-5:]))
            return "\n".join(parts)


class FileReadStrategy(ToolCompressStrategy):
    """File content: keep structure info, drop body."""

    def compress(self, content: str, tool_name: str, max_tokens: int, level: str) -> str:
        lines = content.split("\n")
        structure = self._extract_structure(content)

        if level == "light":
            return (
                "\n".join(lines[:30])
                + f"\n\n[...{len(lines)} lines total...]\nStructure: {structure}"
            )
        else:
            return (
                f"[file summary] {len(lines)} lines\n"
                f"Structure: {structure}\n"
                f"[Full content was analyzed in the assistant reply that followed this tool call]"
            )

    def _extract_structure(self, content: str) -> str:
        classes = re.findall(r"^class\s+(\w+)", content, re.MULTILINE)
        functions = re.findall(r"^(?:def|function|const|async)\s+(\w+)", content, re.MULTILINE)
        return f"classes={classes}, functions={functions}"


class SearchResultStrategy(ToolCompressStrategy):
    """Web search: keep titles + snippets, drop full content."""

    def compress(self, content: str, tool_name: str, max_tokens: int, level: str) -> str:
        try:
            results = json.loads(content)
        except json.JSONDecodeError:
            return DefaultTruncateStrategy().compress(content, tool_name, max_tokens, level)

        if level == "light":
            return json.dumps([
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("snippet", "")[:200],
                }
                for r in results
            ], ensure_ascii=False)
        else:
            titles = [r.get("title", "untitled") for r in results]
            return f"[search summary] {len(results)} results: " + "; ".join(titles)


class PassThroughStrategy(ToolCompressStrategy):
    """Returns content unchanged. For tools with already-small output."""

    def compress(self, content: str, tool_name: str, max_tokens: int, level: str) -> str:
        return content


class DefaultTruncateStrategy(ToolCompressStrategy):
    """Fallback: head + tail truncation."""

    def compress(self, content: str, tool_name: str, max_tokens: int, level: str) -> str:
        max_chars = max_tokens * 4
        if len(content) <= max_chars:
            return content
        head = int(max_chars * 0.6)
        tail = int(max_chars * 0.3)
        omitted = len(content) - head - tail
        return content[:head] + f"\n\n[...{omitted} chars omitted...]\n\n" + content[-tail:]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_compress_strategies.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add mini_agent/context/compress_strategies.py tests/test_compress_strategies.py
git commit -m "feat(context): add per-tool compress strategies (SQL, code, file, search, passthrough, default)"
```

---

### Task 9: Implement MicroCompressor

**Files:**
- Create: `mini_agent/context/micro_compressor.py`
- Test: `tests/test_micro_compressor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_micro_compressor.py
from mini_agent.context.micro_compressor import MicroCompressor
from mini_agent.context.config import ContextConfig
from mini_agent.context.models import (
    ContextBlock, Layer, BlockType, BlockStatus
)


def _tool_block(id: str, turn_id: int, tool_name: str = "some_tool",
                token_count: int = 5000, layer: Layer = Layer.L1_WORKING,
                status: BlockStatus = BlockStatus.ACTIVE) -> ContextBlock:
    content = "x" * (token_count * 4)  # ~token_count tokens
    return ContextBlock(
        id=id, turn_id=turn_id, block_type=BlockType.TOOL_CALL,
        layer=layer, status=status,
        original_content=content, working_content=content,
        token_count=token_count, original_token_count=token_count,
        tool_name=tool_name,
    )


def test_skip_recent_blocks():
    """Blocks younger than micro_compress_after_turns are untouched."""
    config = ContextConfig(micro_compress_after_turns=3)
    compressor = MicroCompressor(config, tool_registry={})
    block = _tool_block("b1", 8, token_count=5000)
    compressor.compress([block], current_turn=10)  # age=2 < 3
    assert block.status == BlockStatus.ACTIVE
    assert block.token_count == 5000


def test_compress_old_l1_block():
    """Old tool blocks in L1 get light compression."""
    config = ContextConfig(micro_compress_after_turns=3, tool_output_max_tokens_l1=2000)
    compressor = MicroCompressor(config, tool_registry={})
    block = _tool_block("b1", 1, token_count=5000)
    compressor.compress([block], current_turn=10)  # age=9 > 3
    assert block.status == BlockStatus.MICRO_COMPRESSED
    assert block.token_count < 5000
    assert len(block.compression_history) == 1
    assert block.compression_history[0]["stage"] == "micro"


def test_compress_l2_block_aggressive():
    """L2+ blocks get aggressive compression."""
    config = ContextConfig(tool_output_max_tokens_l2=200)
    compressor = MicroCompressor(config, tool_registry={})
    block = _tool_block("b1", 1, token_count=5000, layer=Layer.L2_REFERENCE)
    compressor.compress([block], current_turn=10)
    assert block.status == BlockStatus.MICRO_COMPRESSED
    assert block.compression_history[0]["level"] == "aggressive"


def test_skip_non_tool_blocks():
    """Only TOOL_CALL blocks get compressed."""
    config = ContextConfig(micro_compress_after_turns=0)
    compressor = MicroCompressor(config, tool_registry={})
    block = ContextBlock(
        id="b1", turn_id=1, block_type=BlockType.USER_MESSAGE,
        layer=Layer.L1_WORKING, original_content="hello",
        working_content="hello", token_count=5000, original_token_count=5000,
    )
    compressor.compress([block], current_turn=10)
    assert block.status == BlockStatus.ACTIVE


def test_skip_pinned_blocks():
    config = ContextConfig(micro_compress_after_turns=0)
    compressor = MicroCompressor(config, tool_registry={})
    block = _tool_block("b1", 1, status=BlockStatus.PINNED)
    compressor.compress([block], current_turn=10)
    assert block.status == BlockStatus.PINNED


def test_skip_already_small_blocks():
    """Blocks already under the limit are not compressed."""
    config = ContextConfig(micro_compress_after_turns=0, tool_output_max_tokens_l1=2000)
    compressor = MicroCompressor(config, tool_registry={})
    block = _tool_block("b1", 1, token_count=500)
    compressor.compress([block], current_turn=10)
    assert block.status == BlockStatus.ACTIVE
    assert len(block.compression_history) == 0


def test_tool_registry_strategy_used():
    """Tool instance's compress_strategy takes priority."""
    from mini_agent.context.compress_strategies import PassThroughStrategy

    class FakeTool:
        compress_strategy = PassThroughStrategy()

    config = ContextConfig(micro_compress_after_turns=0, tool_output_max_tokens_l1=100)
    compressor = MicroCompressor(config, tool_registry={"my_tool": FakeTool()})
    block = _tool_block("b1", 1, tool_name="my_tool", token_count=5000)
    compressor.compress([block], current_turn=10)
    # PassThrough means content unchanged, but status still updated
    assert block.working_content == block.original_content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_micro_compressor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement MicroCompressor**

```python
# mini_agent/context/micro_compressor.py
from .config import ContextConfig
from .models import ContextBlock, Layer, BlockType, BlockStatus
from .token_counter import count_tokens
from .compress_strategies import (
    ToolCompressStrategy,
    SqlResultStrategy,
    CodeOutputStrategy,
    FileReadStrategy,
    SearchResultStrategy,
    PassThroughStrategy,
    DefaultTruncateStrategy,
)


class MicroCompressor:
    """Stage 2: Per-tool deterministic compression. Zero LLM cost."""

    def __init__(self, config: ContextConfig, tool_registry: dict):
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

    def compress(self, blocks: list[ContextBlock], current_turn: int):
        for block in blocks:
            if block.block_type != BlockType.TOOL_CALL:
                continue
            if block.status in (BlockStatus.OBSOLETE, BlockStatus.PINNED):
                continue

            age = current_turn - block.turn_id
            if age <= self.config.micro_compress_after_turns:
                continue

            if block.layer == Layer.L1_WORKING:
                max_tokens = self.config.tool_output_max_tokens_l1
                level = "light"
            else:
                max_tokens = self.config.tool_output_max_tokens_l2
                level = "aggressive"

            if block.token_count <= max_tokens:
                continue

            strategy = self._get_strategy(block)
            before = block.token_count
            block.working_content = strategy.compress(
                block.original_content, block.tool_name or "", max_tokens, level
            )
            block.token_count = count_tokens(block.working_content)
            block.status = BlockStatus.MICRO_COMPRESSED
            block.compression_history.append({
                "stage": "micro",
                "level": level,
                "before_tokens": before,
                "after_tokens": block.token_count,
            })

    def _get_strategy(self, block: ContextBlock) -> ToolCompressStrategy:
        # 1. Tool instance's own strategy
        tool = self.tool_registry.get(block.tool_name)
        if tool and getattr(tool, "compress_strategy", None):
            return tool.compress_strategy
        # 2. Built-in name mapping
        if block.tool_name in self.strategies:
            return self.strategies[block.tool_name]
        # 3. Fallback
        return self.default_strategy
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_micro_compressor.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add mini_agent/context/micro_compressor.py tests/test_micro_compressor.py
git commit -m "feat(context): add MicroCompressor with 3-tier strategy lookup (tool → builtin → default)"
```

---

## Chunk 4: AutoCompressor + ContextEditor

### Task 10: Implement AutoCompressor

**Files:**
- Create: `mini_agent/context/auto_compressor.py`
- Test: `tests/test_auto_compressor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_auto_compressor.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from mini_agent.context.auto_compressor import AutoCompressor
from mini_agent.context.block_store import BlockStore
from mini_agent.context.config import ContextConfig
from mini_agent.context.models import (
    ContextBlock, Layer, BlockType, BlockStatus
)


def _add_block(store: BlockStore, id: str, turn_id: int,
               status: BlockStatus = BlockStatus.MICRO_COMPRESSED,
               layer: Layer = Layer.L1_WORKING, token_count: int = 1000):
    store.add(ContextBlock(
        id=id, turn_id=turn_id, block_type=BlockType.TOOL_CALL,
        layer=layer, status=status,
        original_content="x" * 100, working_content="x" * 100,
        token_count=token_count, original_token_count=token_count * 5,
    ))


def test_should_trigger_above_threshold():
    config = ContextConfig(total_token_budget=10000, auto_compress_trigger_ratio=0.85)
    store = BlockStore()
    _add_block(store, "b1", 1, token_count=9000)  # 90% usage
    compressor = AutoCompressor(config, llm_client=MagicMock())
    assert compressor.should_trigger(store) is True


def test_should_not_trigger_below_threshold():
    config = ContextConfig(total_token_budget=10000, auto_compress_trigger_ratio=0.85)
    store = BlockStore()
    _add_block(store, "b1", 1, token_count=5000)  # 50% usage
    compressor = AutoCompressor(config, llm_client=MagicMock())
    assert compressor.should_trigger(store) is False


async def test_compress_generates_summary():
    config = ContextConfig(
        total_token_budget=5000,  # Low budget so tokens_to_free > 0
        auto_compress_trigger_ratio=0.85,
        auto_compress_target_ratio=0.60,
        l1_window_turns=8,
    )
    store = BlockStore()
    # Add old micro-compressed blocks (total 6000 > budget 5000)
    _add_block(store, "b1", 1, token_count=2000)
    _add_block(store, "b2", 2, token_count=2000)
    _add_block(store, "b3", 3, token_count=2000)

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(content="summary of turns 1-3")

    compressor = AutoCompressor(config, llm_client=mock_llm)
    summary = await compressor.compress(store, current_turn=20)

    assert summary is not None
    assert summary.block_type == BlockType.SUMMARY
    assert summary.layer == Layer.L2_REFERENCE
    # Original blocks should be marked SUMMARIZED
    assert store.get("b1").status == BlockStatus.SUMMARIZED
    mock_llm.generate.assert_called_once()


async def test_compress_skips_when_no_compressible():
    config = ContextConfig(total_token_budget=10000, l1_window_turns=8)
    store = BlockStore()
    _add_block(store, "b1", 1, status=BlockStatus.ACTIVE)  # Not MICRO_COMPRESSED
    mock_llm = AsyncMock()
    compressor = AutoCompressor(config, llm_client=mock_llm)
    result = await compressor.compress(store, current_turn=20)
    assert result is None
    mock_llm.generate.assert_not_called()


async def test_compress_handles_llm_failure():
    config = ContextConfig(
        total_token_budget=10000, l1_window_turns=8,
        auto_compress_trigger_ratio=0.85, auto_compress_target_ratio=0.60,
    )
    store = BlockStore()
    _add_block(store, "b1", 1, token_count=5000)

    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = Exception("LLM timeout")

    compressor = AutoCompressor(config, llm_client=mock_llm)
    result = await compressor.compress(store, current_turn=20)

    assert result is None
    # Block should NOT be marked SUMMARIZED on failure
    assert store.get("b1").status == BlockStatus.MICRO_COMPRESSED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_auto_compressor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement AutoCompressor**

```python
# mini_agent/context/auto_compressor.py
import logging
from typing import Optional
from .config import ContextConfig
from .block_store import BlockStore
from .models import ContextBlock, Layer, BlockType, BlockStatus
from .token_counter import count_tokens
from ..schema.schema import Message

logger = logging.getLogger(__name__)


class AutoCompressor:
    """Stage 3: LLM-powered batch summarization when token pressure builds."""

    def __init__(self, config: ContextConfig, llm_client):
        self.config = config
        self.llm = llm_client

    def should_trigger(self, store: BlockStore) -> bool:
        threshold = self.config.total_token_budget * self.config.auto_compress_trigger_ratio
        return store.total_active_tokens() > threshold

    async def compress(self, store: BlockStore, current_turn: int) -> Optional[ContextBlock]:
        compressible = sorted(
            [b for b in store.active_blocks()
             if b.status == BlockStatus.MICRO_COMPRESSED
             and b.layer in (Layer.L1_WORKING, Layer.L2_REFERENCE)
             and (current_turn - b.turn_id) > self.config.l1_window_turns // 2],
            key=lambda b: b.turn_id,
        )

        if not compressible:
            return None

        current = store.total_active_tokens()
        target = int(self.config.total_token_budget * self.config.auto_compress_target_ratio)
        tokens_to_free = current - target

        selected, freed = [], 0
        for b in compressible:
            if freed >= tokens_to_free:
                break
            selected.append(b)
            freed += b.token_count

        if not selected:
            return None

        try:
            summary_content = await self._generate_summary(selected)
        except Exception:
            logger.warning("AutoCompressor: LLM summary failed, skipping this cycle")
            return None

        # Mark originals as summarized
        for b in selected:
            b.status = BlockStatus.SUMMARIZED

        summary = ContextBlock(
            id=f"summary_{selected[0].turn_id}_to_{selected[-1].turn_id}",
            turn_id=selected[0].turn_id,
            block_type=BlockType.SUMMARY,
            layer=Layer.L2_REFERENCE,
            status=BlockStatus.ACTIVE,
            original_content=summary_content,
            working_content=summary_content,
            token_count=count_tokens(summary_content),
            original_token_count=count_tokens(summary_content),
            tags=list(set(tag for b in selected for tag in b.tags)),
        )
        store.add(summary)
        return summary

    async def _generate_summary(self, blocks: list[ContextBlock]) -> str:
        blocks_text = "\n\n---\n\n".join(
            f"[Turn {b.turn_id}] ({b.block_type.value})\n{b.working_content}"
            for b in blocks
        )
        target_tokens = max(200, int(sum(b.token_count for b in blocks) * 0.20))

        prompt = (
            f"Compress the following conversation segment into a structured summary.\n\n"
            f"Requirements:\n"
            f"1. Preserve all key facts and data conclusions\n"
            f"2. Preserve all important decisions and their reasoning\n"
            f"3. Preserve current state of data/files/variables\n"
            f"4. Discard: raw data, intermediate exploration, rejected approaches\n"
            f"5. Output in YAML format\n"
            f"6. Target length: ~{target_tokens} tokens\n\n"
            f"Conversation segment (turns {blocks[0].turn_id}-{blocks[-1].turn_id}):\n"
            f"{blocks_text}"
        )

        messages = [Message(role="user", content=prompt)]
        response = await self.llm.generate(messages=messages)
        return f"[Summary: turns {blocks[0].turn_id}-{blocks[-1].turn_id}]\n{response.content}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_auto_compressor.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add mini_agent/context/auto_compressor.py tests/test_auto_compressor.py
git commit -m "feat(context): add AutoCompressor with async LLM summarization and failure recovery"
```

---

### Task 11: Implement ContextEditor

**Files:**
- Create: `mini_agent/context/context_editor.py`
- Test: `tests/test_context_editor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_context_editor.py
from mini_agent.context.context_editor import ContextEditor, CONTEXT_EDITING_TOOLS
from mini_agent.context.block_store import BlockStore
from mini_agent.context.models import (
    ContextBlock, Layer, BlockType, BlockStatus
)


def _setup_store() -> BlockStore:
    store = BlockStore()
    for i in range(1, 6):
        store.add(ContextBlock(
            id=f"turn_{i:03d}_0_tool", turn_id=i,
            block_type=BlockType.TOOL_CALL, layer=Layer.L1_WORKING,
            original_content=f"content {i}", working_content=f"content {i}",
            token_count=500, original_token_count=500,
        ))
    return store


def test_mark_obsolete():
    store = _setup_store()
    editor = ContextEditor(store)
    result = editor.execute("context_mark_obsolete", {
        "turn_ids": [1, 2],
        "reason": "superseded by new query",
    })
    assert "obsolete" in result.lower()
    assert store.get("turn_001_0_tool").status == BlockStatus.OBSOLETE
    assert store.get("turn_002_0_tool").status == BlockStatus.OBSOLETE
    assert store.get("turn_003_0_tool").status == BlockStatus.ACTIVE


def test_mark_obsolete_skips_pinned():
    store = _setup_store()
    store.get("turn_001_0_tool").status = BlockStatus.PINNED
    editor = ContextEditor(store)
    editor.execute("context_mark_obsolete", {
        "turn_ids": [1],
        "reason": "test",
    })
    assert store.get("turn_001_0_tool").status == BlockStatus.PINNED


def test_compress_to_conclusion():
    store = _setup_store()
    editor = ContextEditor(store)
    result = editor.execute("context_compress_to_conclusion", {
        "turn_ids": [1, 2, 3],
        "conclusion": "The optimal batch size is 64.",
    })
    assert "compressed" in result.lower() or "Compressed" in result
    primary = store.get("turn_001_0_tool")
    assert "optimal batch size is 64" in primary.working_content
    assert store.get("turn_002_0_tool").status == BlockStatus.SUMMARIZED
    assert store.get("turn_003_0_tool").status == BlockStatus.SUMMARIZED


def test_pin():
    store = _setup_store()
    editor = ContextEditor(store)
    result = editor.execute("context_pin", {
        "turn_ids": [3],
        "reason": "critical constraint",
    })
    assert "pinned" in result.lower() or "Pinned" in result
    block = store.get("turn_003_0_tool")
    assert block.status == BlockStatus.PINNED
    assert block.layer == Layer.L0_CORE


def test_unknown_tool():
    store = BlockStore()
    editor = ContextEditor(store)
    result = editor.execute("context_unknown", {})
    assert "unknown" in result.lower() or "Unknown" in result


def test_edit_log_recorded():
    store = _setup_store()
    editor = ContextEditor(store)
    editor.execute("context_mark_obsolete", {"turn_ids": [1], "reason": "test"})
    editor.execute("context_pin", {"turn_ids": [2]})
    assert len(editor.edit_log) == 2
    assert editor.edit_log[0]["action"] == "mark_obsolete"
    assert editor.edit_log[1]["action"] == "pin"


def test_context_editing_tools_schema():
    """Verify tool definitions are well-formed."""
    assert len(CONTEXT_EDITING_TOOLS) == 3
    names = {t["name"] for t in CONTEXT_EDITING_TOOLS}
    assert names == {"context_mark_obsolete", "context_compress_to_conclusion", "context_pin"}
    for tool in CONTEXT_EDITING_TOOLS:
        assert "description" in tool
        assert "input_schema" in tool
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_context_editor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ContextEditor**

```python
# mini_agent/context/context_editor.py
from .block_store import BlockStore
from .models import ContextBlock, Layer, BlockStatus
from .token_counter import count_tokens

CONTEXT_EDITING_TOOLS = [
    {
        "name": "context_mark_obsolete",
        "description": (
            "Mark tool results from specified turns as obsolete. "
            "Use when: a previous query has been superseded by a new query; "
            "the user explicitly changed direction; "
            "an exploratory attempt proved unhelpful."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "turn_ids": {
                    "type": "array", "items": {"type": "integer"},
                    "description": "Turn numbers to mark as obsolete",
                },
                "reason": {
                    "type": "string",
                    "description": "Brief reason (kept as audit trail)",
                },
            },
            "required": ["turn_ids", "reason"],
        },
    },
    {
        "name": "context_compress_to_conclusion",
        "description": (
            "Replace a multi-turn analysis process with its final conclusion. "
            "Use when: you completed a multi-step analysis and reached a final answer; "
            "you tried multiple approaches and selected the best one; "
            "a tuning/debugging cycle is finished."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "turn_ids": {
                    "type": "array", "items": {"type": "integer"},
                    "description": "Turn range to compress",
                },
                "conclusion": {
                    "type": "string",
                    "description": "The final conclusion that replaces the process",
                },
            },
            "required": ["turn_ids", "conclusion"],
        },
    },
    {
        "name": "context_pin",
        "description": (
            "Pin important context to prevent it from being compressed. "
            "Use when: the user stated an explicit constraint; "
            "a key configuration needs to be referenced across many turns; "
            "an important decision (with reasoning) must be preserved long-term."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "turn_ids": {
                    "type": "array", "items": {"type": "integer"},
                    "description": "Turn numbers to pin",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this is pinned",
                },
            },
            "required": ["turn_ids"],
        },
    },
]

CONTEXT_EDITING_PROMPT = """
## Context management

You have context editing tools to keep the conversation history lean and relevant.

### When to use context_mark_obsolete
- A newer query/analysis has replaced an older one
- The user said "never mind", "skip that", "let's switch to..."
- An exploratory attempt was abandoned
- You re-read a file that has been modified since the last read

### When to use context_compress_to_conclusion
- You finished a multi-step analysis (tried multiple groupings, selected the best)
- A parameter tuning cycle is complete
- You compared multiple models/approaches and chose one
- Signal phrases: "in summary", "the final answer is", "after testing, the best..."

### When to use context_pin
- The user stated a hard constraint ("must use PostgreSQL", "budget under 100K")
- A configuration value or file path that will be referenced repeatedly
- A critical decision with its reasoning

### Rules
- After completing a group of related tool calls, ask yourself: can I compress?
- When uncertain, do NOT edit — keeping is safer than deleting
- Editing tools are invisible to the user; they are internal optimization
- The original data is always preserved and can be recovered if needed
"""


class ContextEditor:
    """Stage 4: Agent self-edits context via tool calls."""

    def __init__(self, store: BlockStore):
        self.store = store
        self.edit_log: list[dict] = []

    def execute(self, tool_name: str, tool_input: dict) -> str:
        handler = {
            "context_mark_obsolete": self._mark_obsolete,
            "context_compress_to_conclusion": self._compress_to_conclusion,
            "context_pin": self._pin,
        }.get(tool_name)

        if not handler:
            return f"Unknown context editing operation: {tool_name}"
        return handler(tool_input)

    def _mark_obsolete(self, input: dict) -> str:
        turn_ids = input["turn_ids"]
        reason = input["reason"]
        freed = 0
        for tid in turn_ids:
            for block in self.store.get_blocks_by_turn(tid):
                if block.status == BlockStatus.PINNED:
                    continue
                freed += block.token_count
                block.status = BlockStatus.OBSOLETE
                block.working_content = f"[obsolete: {reason}]"
                block.token_count = count_tokens(block.working_content)
                freed -= block.token_count
        self.edit_log.append({
            "action": "mark_obsolete", "turns": turn_ids,
            "reason": reason, "freed_tokens": freed,
        })
        return f"Marked turns {turn_ids} as obsolete, freed ~{freed} tokens. Reason: {reason}"

    def _compress_to_conclusion(self, input: dict) -> str:
        turn_ids = input["turn_ids"]
        conclusion = input["conclusion"]
        freed = 0

        first_blocks = self.store.get_blocks_by_turn(turn_ids[0])
        if first_blocks:
            primary = first_blocks[0]
            freed += primary.token_count
            primary.working_content = (
                f"[Compressed: turns {turn_ids[0]}-{turn_ids[-1]}]\n"
                f"Conclusion:\n{conclusion}"
            )
            primary.token_count = count_tokens(primary.working_content)
            primary.status = BlockStatus.MICRO_COMPRESSED
            freed -= primary.token_count

        for tid in turn_ids[1:]:
            for block in self.store.get_blocks_by_turn(tid):
                freed += block.token_count
                block.status = BlockStatus.SUMMARIZED

        self.edit_log.append({
            "action": "compress_to_conclusion", "turns": turn_ids,
            "conclusion_preview": conclusion[:100], "freed_tokens": freed,
        })
        return f"Compressed turns {turn_ids[0]}-{turn_ids[-1]} to conclusion, freed ~{freed} tokens"

    def _pin(self, input: dict) -> str:
        turn_ids = input["turn_ids"]
        reason = input.get("reason", "")
        pinned_tokens = 0
        for tid in turn_ids:
            for block in self.store.get_blocks_by_turn(tid):
                block.status = BlockStatus.PINNED
                block.layer = Layer.L0_CORE
                pinned_tokens += block.token_count
        self.edit_log.append({
            "action": "pin", "turns": turn_ids,
            "reason": reason, "pinned_tokens": pinned_tokens,
        })
        return f"Pinned turns {turn_ids}, {pinned_tokens} tokens will always be retained"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_context_editor.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add mini_agent/context/context_editor.py tests/test_context_editor.py
git commit -m "feat(context): add ContextEditor with mark_obsolete/compress_to_conclusion/pin tools"
```

---

## Chunk 5: ContextManager Facade + Agent Integration

### Task 12: Implement ContextManager facade

**Files:**
- Create: `mini_agent/context/context_manager.py`
- Modify: `mini_agent/context/__init__.py`
- Test: `tests/test_context_manager.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_context_manager.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from mini_agent.context.context_manager import ContextManager
from mini_agent.context.config import ContextConfig
from mini_agent.context.models import BlockType, BlockStatus, Layer


async def test_init_system():
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    cm.init_system("You are a helpful assistant.")
    blocks = cm.store.all()
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.SYSTEM
    assert blocks[0].status == BlockStatus.PINNED
    assert blocks[0].layer == Layer.L0_CORE


async def test_add_user_message_first_is_intent():
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    cm.add_user_message("Analyze sales data")
    blocks = cm.store.all()
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.USER_INTENT
    assert blocks[0].token_count > 0
    assert blocks[0].original_token_count > 0


async def test_add_user_message_subsequent_is_message():
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    cm.add_user_message("First")
    cm.add_user_message("Second")
    blocks = cm.store.all()
    assert blocks[0].block_type == BlockType.USER_INTENT
    assert blocks[1].block_type == BlockType.USER_MESSAGE


async def test_add_tool_call():
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    cm.add_user_message("test")
    cm.add_tool_call("read_file", {"path": "/tmp/f.py"}, "file contents here")
    blocks = cm.store.all()
    tool_block = [b for b in blocks if b.block_type == BlockType.TOOL_CALL][0]
    assert tool_block.tool_name == "read_file"
    assert tool_block.original_content == "file contents here"
    assert tool_block.token_count > 0


async def test_add_assistant_reply():
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    cm.add_user_message("test")
    cm.add_assistant_reply("Here is my answer.", thinking="Let me think...")
    blocks = cm.store.all()
    reply = [b for b in blocks if b.block_type == BlockType.ASSISTANT_REPLY][0]
    assert reply.working_content == "Here is my answer."
    assert "has_thinking" in reply.tags


async def test_process_and_assemble_returns_messages():
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=AsyncMock(), tool_registry={})
    cm.init_system("system prompt")
    cm.add_user_message("hello")
    cm.add_assistant_reply("hi there")
    messages = await cm.process_and_assemble()
    assert len(messages) >= 2
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles


async def test_handle_context_tool():
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    cm.add_user_message("test")
    cm.add_tool_call("read_file", {}, "big content")
    result = cm.handle_context_tool("context_mark_obsolete", {
        "turn_ids": [1], "reason": "test",
    })
    assert "obsolete" in result.lower()


async def test_get_context_tools_when_enabled():
    config = ContextConfig(enable_context_editing=True)
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    tools = cm.get_context_tools()
    assert len(tools) == 3


async def test_get_context_tools_when_disabled():
    config = ContextConfig(enable_context_editing=False)
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    tools = cm.get_context_tools()
    assert len(tools) == 0


async def test_get_status():
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    cm.init_system("sys")
    cm.add_user_message("hello")
    status = cm.get_status()
    assert "total_blocks" in status
    assert "active_blocks" in status
    assert "total_active_tokens" in status
    assert "budget_usage" in status
    assert "blocks_per_layer" in status


async def test_multiple_tool_calls_unique_ids():
    """Two add_tool_call in same turn produce unique block IDs."""
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    cm.add_user_message("test")
    cm.add_tool_call("read_file", {"path": "/a"}, "content a", tool_call_id="tc_1")
    cm.add_tool_call("read_file", {"path": "/b"}, "content b", tool_call_id="tc_2")
    blocks = [b for b in cm.store.all() if b.block_type == BlockType.TOOL_CALL]
    assert len(blocks) == 2
    assert blocks[0].id != blocks[1].id
    assert blocks[0].tool_call_id == "tc_1"
    assert blocks[1].tool_call_id == "tc_2"


async def test_mode_upgrade_claude_code_to_hybrid():
    """When usage > 70% and turn > 20, upgrades from claude_code to hybrid."""
    config = ContextConfig.from_mode("claude_code", total_token_budget=1000)
    cm = ContextManager(config, llm_client=AsyncMock(), tool_registry={})
    cm.init_system("sys")
    assert cm.config.l1_window_turns > 1000  # Claude Code mode

    # Simulate 21 turns with high token usage
    for i in range(1, 22):
        cm.add_user_message(f"msg {i}")
        cm.add_tool_call("bash", {}, "x" * 200)

    await cm.process_and_assemble()
    # Should have upgraded to hybrid
    assert cm.config.l1_window_turns == 8
    assert cm.config.layering_enabled is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_context_manager.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ContextManager**

```python
# mini_agent/context/context_manager.py
import json
import logging
from typing import Optional
from .config import ContextConfig
from .block_store import BlockStore
from .models import ContextBlock, Layer, BlockType, BlockStatus
from .token_counter import count_tokens
from .classifier import LayerClassifier
from .micro_compressor import MicroCompressor
from .auto_compressor import AutoCompressor
from .context_editor import ContextEditor, CONTEXT_EDITING_TOOLS, CONTEXT_EDITING_PROMPT
from .budget_assembler import BudgetAssembler

logger = logging.getLogger(__name__)


class ContextManager:
    """Facade that orchestrates the 5-stage context management pipeline."""

    def __init__(self, config: ContextConfig, llm_client, tool_registry: dict):
        self.config = config
        self.store = BlockStore()
        self.classifier = LayerClassifier(config)
        self.micro = MicroCompressor(config, tool_registry)
        self.auto = AutoCompressor(config, llm_client)
        self.editor = ContextEditor(self.store)
        self.assembler = BudgetAssembler(config)
        self.current_turn = 0
        self._tool_call_index = 0  # For unique IDs within a turn

    def init_system(self, system_prompt: str):
        """Initialize system block with context editing prompt if enabled."""
        content = system_prompt
        if self.config.enable_context_editing:
            content += "\n\n" + CONTEXT_EDITING_PROMPT
        self.store.add(ContextBlock(
            id="system",
            turn_id=0,
            block_type=BlockType.SYSTEM,
            layer=Layer.L0_CORE,
            status=BlockStatus.PINNED,
            original_content=content,
            working_content=content,
            token_count=count_tokens(content),
            original_token_count=count_tokens(content),
        ))

    def add_user_message(self, content: str):
        """Record user message. First message becomes USER_INTENT."""
        self.current_turn += 1
        self._tool_call_index = 0
        block_type = BlockType.USER_INTENT if self.current_turn == 1 else BlockType.USER_MESSAGE
        tokens = count_tokens(content)
        self.store.add(ContextBlock(
            id=f"turn_{self.current_turn:03d}_user",
            turn_id=self.current_turn,
            block_type=block_type,
            layer=Layer.L1_WORKING,
            original_content=content,
            working_content=content,
            token_count=tokens,
            original_token_count=tokens,
        ))

    def add_assistant_reply(self, content: str, thinking: Optional[str] = None):
        """Record assistant reply."""
        tokens = count_tokens(content)
        tags = ["has_thinking"] if thinking else []
        self.store.add(ContextBlock(
            id=f"turn_{self.current_turn:03d}_assistant",
            turn_id=self.current_turn,
            block_type=BlockType.ASSISTANT_REPLY,
            layer=Layer.L1_WORKING,
            original_content=content,
            working_content=content,
            token_count=tokens,
            original_token_count=tokens,
            tags=tags,
        ))

    def add_tool_call(self, tool_name: str, tool_input: dict, tool_result: str,
                      tool_call_id: Optional[str] = None):
        """Record tool_use + tool_result as a single TOOL_CALL block.

        tool_call_id: the original LLM-assigned tool_use ID (for API compatibility).
        tool_input: stored as full JSON (tool inputs are typically small).
        """
        tokens = count_tokens(tool_result)
        input_json = json.dumps(tool_input, ensure_ascii=False)
        self.store.add(ContextBlock(
            id=f"turn_{self.current_turn:03d}_{self._tool_call_index}_{tool_name}",
            turn_id=self.current_turn,
            block_type=BlockType.TOOL_CALL,
            layer=Layer.L1_WORKING,
            original_content=tool_result,
            working_content=tool_result,
            token_count=tokens,
            original_token_count=tokens,
            tool_name=tool_name,
            tool_input_summary=input_json,
            tool_call_id=tool_call_id,
        ))
        self._tool_call_index += 1

    async def process_and_assemble(self) -> list[dict]:
        """Run stages 1→2→3, assemble final messages as dicts.

        Note: Returns list[dict], not list[Message]. The Agent integration layer
        converts these dicts to Message objects before passing to LLM client.
        This keeps ContextManager decoupled from the schema module.
        """
        all_blocks = self.store.all()

        # Stage 1: Classify
        self.classifier.classify(all_blocks, self.current_turn)

        # Stage 2: Micro compress
        self.micro.compress(all_blocks, self.current_turn)

        # Stage 3: Auto compress (async — may call LLM)
        if self.auto.should_trigger(self.store):
            await self.auto.compress(self.store, self.current_turn)

        # Dynamic mode upgrade
        self._maybe_upgrade_mode()

        # Stage 5: Assemble
        messages, usage = self.assembler.assemble(self.store.all())
        logger.debug(
            "Context assembled: L0=%d L1=%d L2=%d L3=%d total=%d/%d",
            usage["L0"], usage["L1"], usage["L2"], usage["L3"],
            sum(usage.values()), self.config.total_token_budget,
        )
        return messages

    def handle_context_tool(self, tool_name: str, tool_input: dict) -> str:
        """Route context_* tool calls to ContextEditor."""
        return self.editor.execute(tool_name, tool_input)

    def get_system_prompt_section(self) -> str:
        """Return context editing guidance for system prompt."""
        if self.config.enable_context_editing:
            return CONTEXT_EDITING_PROMPT
        return ""

    def get_context_tools(self) -> list[dict]:
        """Return context editing tool definitions."""
        if self.config.enable_context_editing:
            return CONTEXT_EDITING_TOOLS
        return []

    def get_status(self) -> dict:
        """Return monitoring/debugging info."""
        return {
            "total_blocks": len(self.store.all()),
            "active_blocks": len(self.store.active_blocks()),
            "total_active_tokens": self.store.total_active_tokens(),
            "budget_usage": (
                self.store.total_active_tokens() / self.config.total_token_budget
                if self.config.total_token_budget > 0 else 0
            ),
            "blocks_per_layer": {
                layer.name: len([b for b in self.store.active_blocks() if b.layer == layer])
                for layer in Layer
            },
            "edit_log": self.editor.edit_log,
            "layering_enabled": self.config.layering_enabled,
            "current_turn": self.current_turn,
        }

    def _maybe_upgrade_mode(self):
        """Auto-upgrade Claude Code → Hybrid when pressure builds."""
        usage_ratio = (
            self.store.total_active_tokens() / self.config.total_token_budget
            if self.config.total_token_budget > 0 else 0
        )
        # Only upgrade when layering is disabled (Claude Code mode) and pressure builds
        if (self.config.l1_window_turns > 1000
                and usage_ratio > 0.70
                and self.current_turn > 20):
            new_config = ContextConfig.from_mode(
                "hybrid", total_token_budget=self.config.total_token_budget
            )
            self.config = new_config
            # Propagate to ALL stages
            self.classifier.config = new_config
            self.micro.config = new_config
            self.auto.config = new_config
            self.assembler.config = new_config
            # Re-classify with new config
            self.classifier.classify(self.store.all(), self.current_turn)
            logger.info("Context mode upgraded: claude_code → hybrid (turn %d)", self.current_turn)
```

- [ ] **Step 4: Update `__init__.py` exports**

```python
# mini_agent/context/__init__.py
"""Context management pipeline for MiniAgent."""

from .config import ContextConfig
from .context_manager import ContextManager

__all__ = ["ContextConfig", "ContextManager"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_context_manager.py -v`
Expected: PASS (10 tests)

- [ ] **Step 6: Run ALL context tests to verify no regressions**

Run: `uv run pytest tests/test_context_models.py tests/test_block_store.py tests/test_token_counter.py tests/test_context_config.py tests/test_classifier.py tests/test_budget_assembler.py tests/test_compress_strategies.py tests/test_micro_compressor.py tests/test_auto_compressor.py tests/test_context_editor.py tests/test_context_manager.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add mini_agent/context/context_manager.py mini_agent/context/__init__.py tests/test_context_manager.py
git commit -m "feat(context): add ContextManager facade orchestrating 5-stage pipeline"
```

---

### Task 13: Integrate ContextConfig into project config.yaml

**Files:**
- Modify: `mini_agent/config.py:39-44` (AgentConfig) and `mini_agent/config.py:83-89` (Config)
- Test: `tests/test_context_config.py` (add integration test)

- [ ] **Step 1: Write failing test for config integration**

Add to `tests/test_context_config.py`:

```python
def test_context_config_in_project_config():
    from mini_agent.config import Config
    # Config requires llm.api_key — check that context field exists on the class
    assert "context" in Config.model_fields
    # Verify default value
    from mini_agent.context.config import ContextConfig
    assert Config.model_fields["context"].default == ContextConfig()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_context_config.py::test_context_config_in_project_config -v`
Expected: FAIL

- [ ] **Step 3: Add ContextConfig to Config class**

In `mini_agent/config.py`, add import and field:

```python
from mini_agent.context.config import ContextConfig
```

Add to `Config` class (around line 83-89):

```python
context: ContextConfig = ContextConfig()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_context_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mini_agent/config.py tests/test_context_config.py
git commit -m "feat(config): integrate ContextConfig into project Config"
```

---

### Task 14: Integrate ContextManager into Agent

**Files:**
- Modify: `mini_agent/agent.py` (major changes to `__init__`, `run_stream`, delete old methods)
- Test: `tests/test_agent.py` (update existing tests)

**Note:** This is the highest-risk task. The key challenge is converting `process_and_assemble()` output (list[dict]) to the `list[Message]` format that `LLMClient.generate_stream()` expects. Add a `_dicts_to_messages()` helper in Agent for this conversion.

- [ ] **Step 1: Run existing agent tests to establish baseline**

Run: `uv run pytest tests/test_agent.py -v`
Note: Record which tests pass. Some may need updating.

- [ ] **Step 2: Modify Agent.__init__ to create ContextManager**

In `mini_agent/agent.py`:

1. Add import at top:
```python
from mini_agent.context import ContextManager, ContextConfig
```

2. Add `context_config: ContextConfig | None = None` parameter to `__init__`.

3. In `__init__` (lines 32-76), add after tool initialization:
```python
# Context management
context_config = context_config or ContextConfig()
tool_registry = {t.name: t for t in self.tools}
self.context_manager = ContextManager(context_config, self.llm_client, tool_registry)
```

4. Keep `self.messages` temporarily as a compatibility shim for `get_history()`:
```python
self.messages = []  # Kept for get_history() compatibility, populated from ContextManager
```

- [ ] **Step 3: Add dict-to-Message conversion helper**

```python
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
                                          arguments=json.dumps(tu["input"])),
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
```

- [ ] **Step 4: Modify run_stream to use ContextManager**

Key changes to `run_stream()` (lines 313-462):

```python
async def run_stream(self, ...):
    # At first call, init system prompt
    if not self.context_manager.store.get("system"):
        self.context_manager.init_system(self.system_prompt)

    # The caller already called add_user_message() — or we delegate:
    # Keep existing add_user_message() on Agent, but delegate to ContextManager
    # self.context_manager.add_user_message() is called by Agent.add_user_message()

    for step in range(self.max_steps):
        await self._check_cancel()

        # Replace _summarize_messages() with ContextManager pipeline
        message_dicts = await self.context_manager.process_and_assemble()
        messages = self._dicts_to_messages(message_dicts)

        # Build tool schemas — include context editing tools
        tool_schemas = [t.to_schema() for t in self.tools]
        for ct in self.context_manager.get_context_tools():
            tool_schemas.append(ct)  # Already in schema dict format

        # Stream LLM response (existing streaming logic unchanged)
        async for chunk in self.llm_client.generate_stream(messages, tool_schemas):
            yield StreamEvent(...)  # existing event yield logic

        # After stream completes, handle tool calls
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name.startswith("context_"):
                    result = self.context_manager.handle_context_tool(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments),
                    )
                    # Context edits don't yield events — invisible to user
                    continue
                else:
                    result = await self._execute_tool(tool_call)
                    self.context_manager.add_tool_call(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments),
                        result,
                        tool_call_id=tool_call.id,
                    )
        else:
            self.context_manager.add_assistant_reply(
                response.content, response.thinking)
            break
```

- [ ] **Step 5: Update add_user_message to delegate to ContextManager**

The existing `Agent.add_user_message()` should delegate:
```python
def add_user_message(self, content: str):
    self.context_manager.add_user_message(content)
```

- [ ] **Step 6: Update get_history() and run() convenience method**

```python
def get_history(self) -> list[dict]:
    """Return current context blocks as summary for external inspection."""
    return [{"id": b.id, "type": b.block_type.value, "status": b.status.value,
             "turn": b.turn_id, "tokens": b.token_count}
            for b in self.context_manager.store.active_blocks()]

async def run(self, user_message: str) -> str:
    """Non-streaming convenience wrapper."""
    self.add_user_message(user_message)
    result = ""
    async for event in self.run_stream():
        if hasattr(event, "text"):
            result += event.text
    return result
```

- [ ] **Step 7: Update _cleanup_incomplete_messages()**

Replace the message-list cleanup with BlockStore-based cleanup:
```python
def _cleanup_incomplete_messages(self):
    """Remove any incomplete blocks from current turn on cancellation."""
    current = self.context_manager.current_turn
    for block in self.context_manager.store.get_blocks_by_turn(current):
        if block.block_type == BlockType.TOOL_CALL and block.status == BlockStatus.ACTIVE:
            self.context_manager.store.remove(block.id)
```

- [ ] **Step 8: Delete old context management code**

Remove from `mini_agent/agent.py`:
- `_summarize_messages()` method (lines 172-252)
- `_estimate_tokens()` method (lines 115-150)
- `_estimate_tokens_fallback()` method (lines 152-170)
- `_create_summary()` method (lines 254-311)
- `self.token_limit` (line 47)
- `self.api_total_tokens` (line 74)
- `self._skip_next_token_check` (line 76)

- [ ] **Step 9: Run agent tests and fix any failures**

Run: `uv run pytest tests/test_agent.py -v`
Fix test failures caused by the interface changes.

- [ ] **Step 10: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 11: Commit**

```bash
git add mini_agent/agent.py tests/test_agent.py
git commit -m "feat(agent): integrate ContextManager, remove old _summarize_messages pipeline"
```

---

### Task 15: End-to-end integration test

**Files:**
- Create: `tests/test_context_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_context_integration.py
"""End-to-end test: ContextManager handles a multi-turn conversation."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from mini_agent.context import ContextManager, ContextConfig
from mini_agent.context.models import BlockType, BlockStatus


async def test_multi_turn_conversation():
    """Simulate a 20-turn conversation and verify compression kicks in."""
    config = ContextConfig.from_mode("hybrid", total_token_budget=5000)
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(content="summary")

    cm = ContextManager(config, llm_client=mock_llm, tool_registry={})
    cm.init_system("You are a helpful assistant.")

    for i in range(1, 21):
        cm.add_user_message(f"Question {i}: " + "detail " * 50)
        cm.add_tool_call("read_file", {"path": f"/file{i}.py"}, "x" * 2000)
        cm.add_assistant_reply(f"Answer {i}: the file contains data.")
        messages = await cm.process_and_assemble()
        assert len(messages) > 0

    status = cm.get_status()
    assert status["total_blocks"] > 20
    # Verify some blocks got compressed
    all_blocks = cm.store.all()
    compressed = [b for b in all_blocks if b.status == BlockStatus.MICRO_COMPRESSED]
    assert len(compressed) > 0


async def test_context_editing_flow():
    """Agent uses context editing tools during conversation."""
    config = ContextConfig.from_mode("hybrid", total_token_budget=10000)
    cm = ContextManager(config, llm_client=AsyncMock(), tool_registry={})
    cm.init_system("sys")
    cm.add_user_message("Analyze data")
    cm.add_tool_call("execute_sql", {"query": "SELECT *"}, "rows..." * 100)
    cm.add_assistant_reply("Found results.")

    # Agent marks old query as obsolete
    result = cm.handle_context_tool("context_mark_obsolete", {
        "turn_ids": [1], "reason": "new query coming",
    })
    assert "obsolete" in result

    # Agent pins important context
    cm.add_user_message("Important: budget must be under 100K")
    result = cm.handle_context_tool("context_pin", {
        "turn_ids": [2], "reason": "budget constraint",
    })
    assert "Pinned" in result

    messages = await cm.process_and_assemble()
    assert len(messages) > 0
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/test_context_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite one final time**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_context_integration.py
git commit -m "test(context): add end-to-end integration tests for multi-turn and editing flows"
```
