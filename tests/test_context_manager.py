# tests/test_context_manager.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from mini_agent.context.context_manager import ContextManager
from mini_agent.context.config import ContextConfig
from mini_agent.context.models import BlockType, BlockStatus, Layer


async def test_init_system():
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    cm.init_system("You are a helpful assistant.")
    blocks = cm.store.all()
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.SYSTEM
    assert blocks[0].status == BlockStatus.PINNED
    assert blocks[0].layer == Layer.L0_CORE


async def test_add_user_message_first_is_intent():
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    cm.add_user_message("Analyze sales data")
    blocks = cm.store.all()
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockType.USER_INTENT
    assert blocks[0].token_count > 0
    assert blocks[0].original_token_count > 0


async def test_add_user_message_subsequent_is_message():
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    cm.add_user_message("First")
    cm.add_user_message("Second")
    blocks = cm.store.all()
    assert blocks[0].block_type == BlockType.USER_INTENT
    assert blocks[1].block_type == BlockType.USER_MESSAGE


async def test_add_tool_call():
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    cm.add_user_message("test")
    cm.add_tool_call("read_file", {"path": "/tmp/f.py"}, "file contents here")
    blocks = cm.store.all()
    tool_block = [b for b in blocks if b.block_type == BlockType.TOOL_CALL][0]
    assert tool_block.tool_name == "read_file"
    assert tool_block.original_content == "file contents here"
    assert tool_block.token_count > 0


async def test_add_assistant_reply():
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    cm.add_user_message("test")
    cm.add_assistant_reply("Here is my answer.", thinking="Let me think...")
    blocks = cm.store.all()
    reply = [b for b in blocks if b.block_type == BlockType.ASSISTANT_REPLY][0]
    assert reply.working_content == "Here is my answer."
    assert "has_thinking" in reply.tags


async def test_process_and_assemble_returns_messages():
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=AsyncMock(), tool_registry={})
    cm.init_system("system prompt")
    cm.add_user_message("hello")
    cm.add_assistant_reply("hi there")
    messages, events = await cm.process_and_assemble()
    assert len(messages) >= 2
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles
    assert any("assembled" in e for e in events)


async def test_handle_context_tool():
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    cm.add_user_message("test")
    cm.add_tool_call("read_file", {}, "big content")
    result = cm.handle_context_tool("context_mark_obsolete", {
        "turn_ids": [1], "reason": "test",
    })
    assert "obsolete" in result.lower()


async def test_get_context_tools_when_enabled():
    config = ContextConfig(enable_context_editing=True)
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    tools = cm.get_context_tools()
    assert len(tools) == 3


async def test_get_context_tools_when_disabled():
    config = ContextConfig(enable_context_editing=False)
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    tools = cm.get_context_tools()
    assert len(tools) == 0


async def test_get_status():
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    cm.init_system("sys")
    cm.add_user_message("hello")
    status = cm.get_status()
    assert "total_blocks" in status
    assert "active_blocks" in status
    assert "total_active_tokens" in status
    assert "budget_usage" in status
    assert "blocks_per_layer" in status


async def test_multiple_tool_calls_unique_ids():
    """Two add_tool_call in same turn produce unique block IDs."""
    config = ContextConfig.from_mode("hybrid")
    cm = ContextManager(config, llm_client=MagicMock(), tool_registry={})
    cm.add_user_message("test")
    cm.add_tool_call("read_file", {"path": "/a"}, "content a", tool_call_id="tc_1")
    cm.add_tool_call("read_file", {"path": "/b"}, "content b", tool_call_id="tc_2")
    blocks = [b for b in cm.store.all() if b.block_type == BlockType.TOOL_CALL]
    assert len(blocks) == 2
    assert blocks[0].id != blocks[1].id
    assert blocks[0].tool_call_id == "tc_1"
    assert blocks[1].tool_call_id == "tc_2"


async def test_mode_upgrade_claude_code_to_hybrid():
    """When usage > 70% and turn > 20, upgrades from claude_code to hybrid."""
    config = ContextConfig.from_mode("claude_code", total_token_budget=1000)
    cm = ContextManager(config, llm_client=AsyncMock(), tool_registry={})
    cm.init_system("sys")
    assert cm.config.l1_window_turns > 1000  # Claude Code mode

    # Simulate 21 turns with high token usage
    for i in range(1, 22):
        cm.add_user_message(f"msg {i}")
        cm.add_tool_call("bash", {}, "x" * 200)

    messages, events = await cm.process_and_assemble()
    # Should have upgraded to hybrid
    assert any("mode_upgrade" in e for e in events)
    assert cm.config.l1_window_turns == 8
    assert cm.config.layering_enabled is True
