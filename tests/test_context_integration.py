"""End-to-end test: ContextManager handles a multi-turn conversation."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from mini_agent.context import ContextManager, ContextConfig
from mini_agent.context.models import BlockType, BlockStatus


async def test_multi_turn_conversation():
    """Simulate a 20-turn conversation and verify compression kicks in."""
    config = ContextConfig.from_mode("hybrid", total_token_budget=5000)
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(content="summary")

    cm = ContextManager(config, llm_client=mock_llm, tool_registry={})
    cm.init_system("You are a helpful assistant.")

    for i in range(1, 21):
        cm.add_user_message(f"Question {i}: " + "detail " * 50)
        cm.add_tool_call("read_file", {"path": f"/file{i}.py"}, "x" * 2000)
        cm.add_assistant_reply(f"Answer {i}: the file contains data.")
        messages = await cm.process_and_assemble()
        assert len(messages) > 0

    status = cm.get_status()
    assert status["total_blocks"] > 20
    # Verify some blocks got compressed
    all_blocks = cm.store.all()
    compressed = [b for b in all_blocks if b.status == BlockStatus.MICRO_COMPRESSED]
    assert len(compressed) > 0


async def test_context_editing_flow():
    """Agent uses context editing tools during conversation."""
    config = ContextConfig.from_mode("hybrid", total_token_budget=10000)
    cm = ContextManager(config, llm_client=AsyncMock(), tool_registry={})
    cm.init_system("sys")
    cm.add_user_message("Analyze data")
    cm.add_tool_call("execute_sql", {"query": "SELECT *"}, "rows..." * 100)
    cm.add_assistant_reply("Found results.")

    # Agent marks old query as obsolete
    result = cm.handle_context_tool("context_mark_obsolete", {
        "turn_ids": [1], "reason": "new query coming",
    })
    assert "obsolete" in result

    # Agent pins important context
    cm.add_user_message("Important: budget must be under 100K")
    result = cm.handle_context_tool("context_pin", {
        "turn_ids": [2], "reason": "budget constraint",
    })
    assert "Pinned" in result

    messages = await cm.process_and_assemble()
    assert len(messages) > 0
