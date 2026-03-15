from .block_store import BlockStore
from .models import Layer, BlockStatus
from .token_counter import count_tokens

CONTEXT_EDITING_TOOLS = [
    {"name": "context_mark_obsolete",
     "description": "Mark tool results from specified turns as obsolete. Use when: a previous query has been superseded; the user changed direction; an exploratory attempt proved unhelpful.",
     "input_schema": {"type": "object", "properties": {
         "turn_ids": {"type": "array", "items": {"type": "integer"}, "description": "Turn numbers to mark as obsolete"},
         "reason": {"type": "string", "description": "Brief reason (kept as audit trail)"}},
         "required": ["turn_ids", "reason"]}},
    {"name": "context_compress_to_conclusion",
     "description": "Replace a multi-turn analysis process with its final conclusion. Use when: you completed a multi-step analysis; you tried multiple approaches and selected the best one; a tuning/debugging cycle is finished.",
     "input_schema": {"type": "object", "properties": {
         "turn_ids": {"type": "array", "items": {"type": "integer"}, "description": "Turn range to compress"},
         "conclusion": {"type": "string", "description": "The final conclusion that replaces the process"}},
         "required": ["turn_ids", "conclusion"]}},
    {"name": "context_pin",
     "description": "Pin important context to prevent compression. Use when: the user stated an explicit constraint; a key configuration needs referencing across many turns; an important decision must be preserved.",
     "input_schema": {"type": "object", "properties": {
         "turn_ids": {"type": "array", "items": {"type": "integer"}, "description": "Turn numbers to pin"},
         "reason": {"type": "string", "description": "Why this is pinned"}},
         "required": ["turn_ids"]}},
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

    def execute(self, tool_name, tool_input):
        handler = {"context_mark_obsolete": self._mark_obsolete,
                    "context_compress_to_conclusion": self._compress_to_conclusion,
                    "context_pin": self._pin}.get(tool_name)
        if not handler:
            return f"Unknown context editing operation: {tool_name}"
        return handler(tool_input)

    def _mark_obsolete(self, input):
        turn_ids, reason = input["turn_ids"], input["reason"]
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
        self.edit_log.append({"action": "mark_obsolete", "turns": turn_ids, "reason": reason, "freed_tokens": freed})
        return f"Marked turns {turn_ids} as obsolete, freed ~{freed} tokens. Reason: {reason}"

    def _compress_to_conclusion(self, input):
        turn_ids, conclusion = input["turn_ids"], input["conclusion"]
        freed = 0
        first_blocks = self.store.get_blocks_by_turn(turn_ids[0])
        if first_blocks:
            primary = first_blocks[0]
            freed += primary.token_count
            primary.working_content = f"[Compressed: turns {turn_ids[0]}-{turn_ids[-1]}]\nConclusion:\n{conclusion}"
            primary.token_count = count_tokens(primary.working_content)
            primary.status = BlockStatus.MICRO_COMPRESSED
            freed -= primary.token_count
        for tid in turn_ids[1:]:
            for block in self.store.get_blocks_by_turn(tid):
                freed += block.token_count
                block.status = BlockStatus.SUMMARIZED
        self.edit_log.append({"action": "compress_to_conclusion", "turns": turn_ids,
                              "conclusion_preview": conclusion[:100], "freed_tokens": freed})
        return f"Compressed turns {turn_ids[0]}-{turn_ids[-1]} to conclusion, freed ~{freed} tokens"

    def _pin(self, input):
        turn_ids = input["turn_ids"]
        reason = input.get("reason", "")
        pinned_tokens = 0
        for tid in turn_ids:
            for block in self.store.get_blocks_by_turn(tid):
                block.status = BlockStatus.PINNED
                block.layer = Layer.L0_CORE
                pinned_tokens += block.token_count
        self.edit_log.append({"action": "pin", "turns": turn_ids, "reason": reason, "pinned_tokens": pinned_tokens})
        return f"Pinned turns {turn_ids}, {pinned_tokens} tokens will always be retained"
