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
