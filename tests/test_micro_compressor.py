from mini_agent.context.micro_compressor import MicroCompressor
from mini_agent.context.config import ContextConfig
from mini_agent.context.models import ContextBlock, Layer, BlockType, BlockStatus


def _tool_block(id, turn_id, tool_name="some_tool", token_count=5000,
                layer=Layer.L1_WORKING, status=BlockStatus.ACTIVE):
    content = "x" * (token_count * 4)
    return ContextBlock(id=id, turn_id=turn_id, block_type=BlockType.TOOL_CALL,
                        layer=layer, status=status, original_content=content,
                        working_content=content, token_count=token_count,
                        original_token_count=token_count, tool_name=tool_name)


def test_skip_recent_blocks():
    config = ContextConfig(micro_compress_after_turns=3)
    compressor = MicroCompressor(config, tool_registry={})
    block = _tool_block("b1", 8, token_count=5000)
    compressor.compress([block], current_turn=10)
    assert block.status == BlockStatus.ACTIVE
    assert block.token_count == 5000


def test_compress_old_l1_block():
    config = ContextConfig(micro_compress_after_turns=3, tool_output_max_tokens_l1=2000)
    compressor = MicroCompressor(config, tool_registry={})
    block = _tool_block("b1", 1, token_count=5000)
    compressor.compress([block], current_turn=10)
    assert block.status == BlockStatus.MICRO_COMPRESSED
    assert block.token_count < 5000
    assert len(block.compression_history) == 1
    assert block.compression_history[0]["stage"] == "micro"


def test_compress_l2_block_aggressive():
    config = ContextConfig(tool_output_max_tokens_l2=200)
    compressor = MicroCompressor(config, tool_registry={})
    block = _tool_block("b1", 1, token_count=5000, layer=Layer.L2_REFERENCE)
    compressor.compress([block], current_turn=10)
    assert block.status == BlockStatus.MICRO_COMPRESSED
    assert block.compression_history[0]["level"] == "aggressive"


def test_skip_non_tool_blocks():
    config = ContextConfig(micro_compress_after_turns=0)
    compressor = MicroCompressor(config, tool_registry={})
    block = ContextBlock(id="b1", turn_id=1, block_type=BlockType.USER_MESSAGE,
                         layer=Layer.L1_WORKING, original_content="hello",
                         working_content="hello", token_count=5000, original_token_count=5000)
    compressor.compress([block], current_turn=10)
    assert block.status == BlockStatus.ACTIVE


def test_skip_pinned_blocks():
    config = ContextConfig(micro_compress_after_turns=0)
    compressor = MicroCompressor(config, tool_registry={})
    block = _tool_block("b1", 1, status=BlockStatus.PINNED)
    compressor.compress([block], current_turn=10)
    assert block.status == BlockStatus.PINNED


def test_skip_already_small_blocks():
    config = ContextConfig(micro_compress_after_turns=0, tool_output_max_tokens_l1=2000)
    compressor = MicroCompressor(config, tool_registry={})
    block = _tool_block("b1", 1, token_count=500)
    compressor.compress([block], current_turn=10)
    assert block.status == BlockStatus.ACTIVE
    assert len(block.compression_history) == 0


def test_tool_registry_strategy_used():
    from mini_agent.context.compress_strategies import PassThroughStrategy
    class FakeTool:
        compress_strategy = PassThroughStrategy()
    config = ContextConfig(micro_compress_after_turns=0, tool_output_max_tokens_l1=100)
    compressor = MicroCompressor(config, tool_registry={"my_tool": FakeTool()})
    block = _tool_block("b1", 1, tool_name="my_tool", token_count=5000)
    compressor.compress([block], current_turn=10)
    assert block.working_content == block.original_content
