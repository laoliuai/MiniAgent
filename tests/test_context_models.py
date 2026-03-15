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
