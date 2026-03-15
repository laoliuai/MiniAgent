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

        for block in blocks:
            if block.layer.value >= Layer.L2_REFERENCE.value:
                if self._is_in_active_chain(block, blocks, current_turn):
                    block.layer = Layer.L1_WORKING

    def _is_in_active_chain(self, block, all_blocks, current_turn):
        for other in all_blocks:
            if (current_turn - other.turn_id) <= self.config.l1_window_turns:
                if block.id in other.depends_on:
                    return True
        return False
