from mini_agent.context.classifier import LayerClassifier
from mini_agent.context.config import ContextConfig
from mini_agent.context.models import ContextBlock, Layer, BlockType, BlockStatus


def _block(id, turn_id, block_type=BlockType.TOOL_CALL, status=BlockStatus.ACTIVE, depends_on=None):
    return ContextBlock(id=id, turn_id=turn_id, block_type=block_type, layer=Layer.L1_WORKING,
                        status=status, original_content="x", working_content="x",
                        token_count=100, original_token_count=100, depends_on=depends_on or [])


def test_system_and_user_intent_always_l0():
    classifier = LayerClassifier(ContextConfig.from_mode("hybrid"))
    blocks = [_block("sys", 0, BlockType.SYSTEM), _block("intent", 1, BlockType.USER_INTENT)]
    classifier.classify(blocks, current_turn=50)
    assert blocks[0].layer == Layer.L0_CORE
    assert blocks[1].layer == Layer.L0_CORE


def test_pinned_always_l0():
    classifier = LayerClassifier(ContextConfig.from_mode("hybrid"))
    blocks = [_block("b1", 1, status=BlockStatus.PINNED)]
    classifier.classify(blocks, current_turn=50)
    assert blocks[0].layer == Layer.L0_CORE


def test_recent_blocks_stay_l1():
    classifier = LayerClassifier(ContextConfig.from_mode("hybrid"))
    blocks = [_block("b1", 18)]
    classifier.classify(blocks, current_turn=20)
    assert blocks[0].layer == Layer.L1_WORKING


def test_old_blocks_demote_to_l2_l3():
    classifier = LayerClassifier(ContextConfig.from_mode("hybrid"))
    blocks = [_block("b1", 5), _block("b2", 1), _block("b3", 0)]
    classifier.classify(blocks, current_turn=30)
    assert blocks[0].layer == Layer.L2_REFERENCE
    assert blocks[1].layer == Layer.L2_REFERENCE
    assert blocks[2].layer == Layer.L2_REFERENCE


def test_very_old_blocks_archive():
    classifier = LayerClassifier(ContextConfig.from_mode("hybrid"))
    blocks = [_block("b1", 1)]
    classifier.classify(blocks, current_turn=50)
    assert blocks[0].layer == Layer.L3_ARCHIVE


def test_claude_code_mode_all_l1():
    classifier = LayerClassifier(ContextConfig.from_mode("claude_code"))
    blocks = [_block("b1", 1), _block("b2", 50)]
    classifier.classify(blocks, current_turn=100)
    assert all(b.layer == Layer.L1_WORKING for b in blocks)


def test_dependency_chain_promotion():
    classifier = LayerClassifier(ContextConfig.from_mode("hybrid"))
    old_block = _block("old", 1)
    recent_block = _block("recent", 25, depends_on=["old"])
    blocks = [old_block, recent_block]
    classifier.classify(blocks, current_turn=30)
    assert old_block.layer == Layer.L1_WORKING
    assert recent_block.layer == Layer.L1_WORKING


def test_skip_obsolete_and_summarized():
    classifier = LayerClassifier(ContextConfig.from_mode("hybrid"))
    blocks = [_block("b1", 1, status=BlockStatus.OBSOLETE), _block("b2", 2, status=BlockStatus.SUMMARIZED)]
    original_layers = [b.layer for b in blocks]
    classifier.classify(blocks, current_turn=50)
    assert [b.layer for b in blocks] == original_layers
