from .config import ContextConfig
from .models import ContextBlock, Layer, BlockType, BlockStatus
from .token_counter import count_tokens
from .compress_strategies import (
    ToolCompressStrategy, SqlResultStrategy, CodeOutputStrategy,
    FileReadStrategy, SearchResultStrategy, PassThroughStrategy,
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
                block.original_content, block.tool_name or "", max_tokens, level)
            block.token_count = count_tokens(block.working_content)
            block.status = BlockStatus.MICRO_COMPRESSED
            block.compression_history.append({
                "stage": "micro", "level": level,
                "before_tokens": before, "after_tokens": block.token_count,
            })

    def _get_strategy(self, block):
        tool = self.tool_registry.get(block.tool_name)
        if tool and getattr(tool, "compress_strategy", None):
            return tool.compress_strategy
        if block.tool_name in self.strategies:
            return self.strategies[block.tool_name]
        return self.default_strategy
