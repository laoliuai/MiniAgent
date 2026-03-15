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

        active = [b for b in blocks if b.status not in (BlockStatus.OBSOLETE, BlockStatus.SUMMARIZED)]

        # L0: all core blocks
        for b in active:
            if b.layer == Layer.L0_CORE:
                selected.append(b)
                used["L0"] += b.token_count

        # L1: newest first
        l1 = sorted([b for b in active if b.layer == Layer.L1_WORKING], key=lambda b: b.turn_id, reverse=True)
        for b in l1:
            if used["L1"] + b.token_count <= budgets["L1"]:
                selected.append(b)
                used["L1"] += b.token_count

        # L2: chronological
        l2 = sorted([b for b in active if b.layer == Layer.L2_REFERENCE], key=lambda b: b.turn_id)
        for b in l2:
            if used["L2"] + b.token_count <= budgets["L2"]:
                selected.append(b)
                used["L2"] += b.token_count

        # L3: remaining budget
        remaining = self.config.total_token_budget - sum(used.values())
        if remaining > 0:
            l3 = sorted([b for b in active if b.layer == Layer.L3_ARCHIVE], key=lambda b: b.turn_id)
            for b in l3:
                if remaining >= b.token_count:
                    selected.append(b)
                    remaining -= b.token_count
                    used["L3"] += b.token_count

        selected.sort(key=lambda b: (b.turn_id, self._type_order(b)))
        messages = self._to_messages(selected)

        if self.config.enable_prefix_caching:
            messages = self._optimize_for_caching(messages)

        return messages, used

    def _type_order(self, block):
        order = {BlockType.SYSTEM: 0, BlockType.USER_INTENT: 1, BlockType.SUMMARY: 2,
                 BlockType.USER_MESSAGE: 3, BlockType.TOOL_CALL: 4,
                 BlockType.ASSISTANT_REPLY: 5, BlockType.PINNED: 6}
        return order.get(block.block_type, 99)

    def _to_messages(self, blocks):
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
                messages.append({"role": "user", "content": f"[System — conversation summary]\n{block.working_content}"})
            elif block.block_type == BlockType.PINNED:
                messages.append({"role": "user", "content": f"[Pinned context]\n{block.working_content}"})
        return messages

    def _expand_tool_block(self, block):
        tool_use_id = block.tool_call_id or block.id
        try:
            tool_input = json.loads(block.tool_input_summary)
        except (json.JSONDecodeError, TypeError):
            tool_input = {"_raw": block.tool_input_summary}
        return [
            {"role": "assistant", "content": None,
             "tool_use": {"id": tool_use_id, "name": block.tool_name, "input": tool_input}},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": block.working_content}
            ]},
        ]

    def _optimize_for_caching(self, messages):
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
        result = stable + volatile
        return self._merge_consecutive_roles(result)

    def _merge_consecutive_roles(self, messages):
        """Merge consecutive same-role user messages to satisfy alternating role constraint."""
        if not messages:
            return messages
        merged = [messages[0]]
        for msg in messages[1:]:
            prev = merged[-1]
            # Only merge consecutive user text messages (not tool_result lists)
            if (msg["role"] == "user" and prev["role"] == "user"
                    and isinstance(msg.get("content", ""), str)
                    and isinstance(prev.get("content", ""), str)):
                prev["content"] = prev["content"] + "\n\n" + msg["content"]
            else:
                merged.append(msg)
        return merged
