"""
Session integration tests - Testing multi-turn conversations and session management
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent import LLMClient
from mini_agent.agent import Agent
from mini_agent.agent_config import AgentConfig
from mini_agent.schema import LLMResponse, Message
from mini_agent.tools.bash_tool import BashTool
from mini_agent.tools.file_tools import ReadTool, WriteTool
from mini_agent.tools.note_tool import RecallNoteTool, SessionNoteTool


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client"""
    client = MagicMock(spec=LLMClient)
    return client


@pytest.fixture
def temp_workspace():
    """Create temporary workspace directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_multi_turn_conversation(mock_llm_client, temp_workspace):
    """Test multi-turn conversation and context sharing"""
    # Prepare test data
    system_prompt = "You are an intelligent assistant"
    tools = [
        ReadTool(workspace_dir=temp_workspace),
        WriteTool(workspace_dir=temp_workspace),
        SessionNoteTool(),
    ]

    # Create agent
    config = AgentConfig(system_prompt=system_prompt, tools=tools)
    agent = Agent(
        llm_client=mock_llm_client,
        config=config,
        workspace_dir=temp_workspace,
    )

    # Verify initial state: messages is empty list (populated during run_stream)
    assert len(agent.messages) == 0
    # System prompt should contain workspace info
    assert system_prompt in agent.system_prompt
    assert "Current Workspace" in agent.system_prompt

    # Add first user message
    agent.add_user_message("Hello")
    # Messages stay empty until run_stream populates them
    # But context_manager has the messages internally

    # Add second user message
    agent.add_user_message("Help me create a file")

    # Verify context_manager has the blocks (system + 2 user)
    history = agent.get_history()
    # get_history now returns block info, not Message objects
    assert len(history) >= 2  # At least 2 user blocks


def test_session_history_management(mock_llm_client, temp_workspace):
    """Test session history management"""
    config = AgentConfig(system_prompt="System prompt")
    agent = Agent(
        llm_client=mock_llm_client,
        config=config,
        workspace_dir=temp_workspace,
    )

    # Add multiple messages
    for i in range(5):
        agent.add_user_message(f"Message {i}")

    # Verify blocks were added via get_history
    history = agent.get_history()
    assert len(history) >= 5  # At least 5 user blocks


def test_get_history(mock_llm_client, temp_workspace):
    """Test getting session history"""
    config = AgentConfig(system_prompt="System")
    agent = Agent(
        llm_client=mock_llm_client,
        config=config,
        workspace_dir=temp_workspace,
    )

    # Add message
    agent.add_user_message("Test message")

    # Get history returns block info dicts
    history = agent.get_history()

    # Verify history is not empty
    assert len(history) >= 1

    # Verify it returns a list of dicts (block summaries)
    assert isinstance(history[0], dict)
    assert "type" in history[0]


@pytest.mark.asyncio
async def test_session_note_persistence(temp_workspace):
    """Test SessionNoteTool persistence functionality"""
    memory_file = Path(temp_workspace) / "memory.json"

    # Create first tool instance and record note
    record_tool = SessionNoteTool(memory_file=str(memory_file))
    result1 = await record_tool.execute(content="Test note", category="test")
    assert result1.success

    # Create second tool instance (simulating new session)
    recall_tool = RecallNoteTool(memory_file=str(memory_file))

    # Verify ability to read previous notes
    result2 = await recall_tool.execute()
    assert result2.success
    assert "Test note" in result2.content


def test_message_statistics(mock_llm_client, temp_workspace):
    """Test message statistics functionality via get_history block counts"""
    config = AgentConfig(system_prompt="System")
    agent = Agent(
        llm_client=mock_llm_client,
        config=config,
        workspace_dir=temp_workspace,
    )

    # Add user messages via context manager
    agent.add_user_message("User message 1")
    agent.add_user_message("User message 2")

    # Get history (block summaries from context manager)
    history = agent.get_history()

    # Should have system block + 2 user blocks
    assert len(history) >= 2
    # Verify block types are present
    block_types = [h["type"] for h in history]
    assert any("user" in bt for bt in block_types)
