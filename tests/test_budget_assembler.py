from mini_agent.context.budget_assembler import BudgetAssembler
from mini_agent.context.config import ContextConfig
from mini_agent.context.models import ContextBlock, Layer, BlockType, BlockStatus


def _block(id, turn_id, layer, block_type=BlockType.TOOL_CALL, token_count=100,
           content="test content", status=BlockStatus.ACTIVE,
           tool_name=None, tool_input_summary=""):
    return ContextBlock(id=id, turn_id=turn_id, block_type=block_type, layer=layer,
                        status=status, original_content=content, working_content=content,
                        token_count=token_count, original_token_count=token_count,
                        tool_name=tool_name, tool_input_summary=tool_input_summary)


def test_l0_always_included():
    assembler = BudgetAssembler(ContextConfig.from_mode("hybrid", total_token_budget=1000))
    blocks = [_block("sys", 0, Layer.L0_CORE, BlockType.SYSTEM, content="system prompt")]
    messages, used = assembler.assemble(blocks)
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert used["L0"] > 0


def test_l1_newest_first():
    assembler = BudgetAssembler(ContextConfig.from_mode("hybrid", total_token_budget=1000))
    blocks = [
        _block("b1", 1, Layer.L1_WORKING, BlockType.USER_MESSAGE, token_count=300),
        _block("b2", 5, Layer.L1_WORKING, BlockType.USER_MESSAGE, token_count=300),
    ]
    messages, used = assembler.assemble(blocks)
    assert any("test content" in (m.get("content", "") or "") for m in messages)


def test_obsolete_and_summarized_excluded():
    assembler = BudgetAssembler(ContextConfig.from_mode("hybrid", total_token_budget=10000))
    blocks = [
        _block("b1", 1, Layer.L1_WORKING, status=BlockStatus.OBSOLETE),
        _block("b2", 2, Layer.L1_WORKING, status=BlockStatus.SUMMARIZED),
        _block("b3", 3, Layer.L1_WORKING, BlockType.USER_MESSAGE),
    ]
    messages, used = assembler.assemble(blocks)
    assert len(messages) == 1


def test_tool_block_expansion():
    assembler = BudgetAssembler(ContextConfig.from_mode("hybrid", total_token_budget=10000))
    blocks = [_block("b1", 1, Layer.L1_WORKING, BlockType.TOOL_CALL,
                     tool_name="read_file", tool_input_summary='{"path": "/tmp/f.py"}',
                     content="file contents here")]
    messages, used = assembler.assemble(blocks)
    assert len(messages) == 2
    assert messages[0]["role"] == "assistant"
    assert messages[1]["role"] == "user"


def test_summary_block_format():
    assembler = BudgetAssembler(ContextConfig.from_mode("hybrid", total_token_budget=10000))
    blocks = [_block("s1", 1, Layer.L2_REFERENCE, BlockType.SUMMARY, content="summary of turns 1-5")]
    messages, used = assembler.assemble(blocks)
    assert len(messages) == 1
    assert "[System — conversation summary]" in messages[0]["content"]


def test_messages_sorted_by_turn():
    assembler = BudgetAssembler(ContextConfig.from_mode("hybrid", total_token_budget=10000))
    blocks = [
        _block("sys", 0, Layer.L0_CORE, BlockType.SYSTEM, content="sys"),
        _block("u3", 3, Layer.L1_WORKING, BlockType.USER_MESSAGE, content="msg3"),
        _block("u1", 1, Layer.L1_WORKING, BlockType.USER_MESSAGE, content="msg1"),
    ]
    messages, used = assembler.assemble(blocks)
    assert messages[0]["role"] == "system"
    # Consecutive user messages get merged to satisfy alternating role constraint
    assert "msg1" in messages[1]["content"]
    assert "msg3" in messages[1]["content"]


def test_claude_code_mode_no_l2_l3():
    assembler = BudgetAssembler(ContextConfig.from_mode("claude_code", total_token_budget=10000))
    blocks = [_block("b1", 1, Layer.L2_REFERENCE, BlockType.SUMMARY, token_count=100)]
    messages, used = assembler.assemble(blocks)
    assert used["L2"] == 0
    assert len(messages) == 0
