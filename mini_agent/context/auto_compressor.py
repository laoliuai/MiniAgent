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
            key=lambda b: b.turn_id)
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

        for b in selected:
            b.status = BlockStatus.SUMMARIZED

        summary = ContextBlock(
            id=f"summary_{selected[0].turn_id}_to_{selected[-1].turn_id}",
            turn_id=selected[0].turn_id, block_type=BlockType.SUMMARY,
            layer=Layer.L2_REFERENCE, status=BlockStatus.ACTIVE,
            original_content=summary_content, working_content=summary_content,
            token_count=count_tokens(summary_content),
            original_token_count=count_tokens(summary_content),
            tags=list(set(tag for b in selected for tag in b.tags)))
        store.add(summary)
        return summary

    async def _generate_summary(self, blocks):
        blocks_text = "\n\n---\n\n".join(
            f"[Turn {b.turn_id}] ({b.block_type.value})\n{b.working_content}" for b in blocks)
        target_tokens = max(200, int(sum(b.token_count for b in blocks) * 0.20))
        prompt = (f"Compress the following conversation segment into a structured summary.\n\n"
                  f"Requirements:\n1. Preserve all key facts and data conclusions\n"
                  f"2. Preserve all important decisions and their reasoning\n"
                  f"3. Preserve current state of data/files/variables\n"
                  f"4. Discard: raw data, intermediate exploration, rejected approaches\n"
                  f"5. Output in YAML format\n6. Target length: ~{target_tokens} tokens\n\n"
                  f"Conversation segment (turns {blocks[0].turn_id}-{blocks[-1].turn_id}):\n{blocks_text}")
        messages = [Message(role="user", content=prompt)]
        response = await self.llm.generate(messages=messages)
        return f"[Summary: turns {blocks[0].turn_id}-{blocks[-1].turn_id}]\n{response.content}"
