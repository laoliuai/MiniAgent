from unittest.mock import AsyncMock, MagicMock
from mini_agent.context.auto_compressor import AutoCompressor
from mini_agent.context.block_store import BlockStore
from mini_agent.context.config import ContextConfig
from mini_agent.context.models import ContextBlock, Layer, BlockType, BlockStatus


def _add_block(store, id, turn_id, status=BlockStatus.MICRO_COMPRESSED,
               layer=Layer.L1_WORKING, token_count=1000):
    store.add(ContextBlock(id=id, turn_id=turn_id, block_type=BlockType.TOOL_CALL,
                           layer=layer, status=status, original_content="x" * 100,
                           working_content="x" * 100, token_count=token_count,
                           original_token_count=token_count * 5))


def test_should_trigger_above_threshold():
    config = ContextConfig(total_token_budget=10000, auto_compress_trigger_ratio=0.85)
    store = BlockStore()
    _add_block(store, "b1", 1, token_count=9000)
    compressor = AutoCompressor(config, llm_client=MagicMock())
    assert compressor.should_trigger(store) is True


def test_should_not_trigger_below_threshold():
    config = ContextConfig(total_token_budget=10000, auto_compress_trigger_ratio=0.85)
    store = BlockStore()
    _add_block(store, "b1", 1, token_count=5000)
    compressor = AutoCompressor(config, llm_client=MagicMock())
    assert compressor.should_trigger(store) is False


async def test_compress_generates_summary():
    config = ContextConfig(total_token_budget=5000, auto_compress_trigger_ratio=0.85,
                           auto_compress_target_ratio=0.60, l1_window_turns=8)
    store = BlockStore()
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
    assert store.get("b1").status == BlockStatus.SUMMARIZED
    mock_llm.generate.assert_called_once()


async def test_compress_skips_when_no_compressible():
    config = ContextConfig(total_token_budget=10000, l1_window_turns=8)
    store = BlockStore()
    _add_block(store, "b1", 1, status=BlockStatus.ACTIVE)
    mock_llm = AsyncMock()
    compressor = AutoCompressor(config, llm_client=mock_llm)
    result = await compressor.compress(store, current_turn=20)
    assert result is None
    mock_llm.generate.assert_not_called()


async def test_compress_handles_llm_failure():
    config = ContextConfig(total_token_budget=5000, l1_window_turns=8,
                           auto_compress_trigger_ratio=0.85, auto_compress_target_ratio=0.60)
    store = BlockStore()
    _add_block(store, "b1", 1, token_count=5000)
    mock_llm = AsyncMock()
    mock_llm.generate.side_effect = Exception("LLM timeout")
    compressor = AutoCompressor(config, llm_client=mock_llm)
    result = await compressor.compress(store, current_turn=20)
    assert result is None
    assert store.get("b1").status == BlockStatus.MICRO_COMPRESSED
