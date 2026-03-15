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
    assert len(active) == 2
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
    store.remove("nonexistent")
