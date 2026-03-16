"""Tests for Agent sub-agent features (AgentConfig, delegation, state tools)."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from mini_agent.agent import Agent
from mini_agent.agent_config import AgentConfig
from mini_agent.shared_state import SharedState


def make_mock_llm():
    """Create a mock LLM client."""
    llm = MagicMock()
    llm.generate_stream = AsyncMock(return_value=AsyncMock())
    return llm


class TestAgentConfigConstructor:
    def test_basic_construction(self, tmp_path):
        config = AgentConfig(system_prompt="You are helpful.")
        agent = Agent(
            llm_client=make_mock_llm(),
            config=config,
            workspace_dir=tmp_path,
        )
        assert agent.config.agent_id == "main"
        assert agent.config.system_prompt == "You are helpful."
        assert "Current Workspace" in agent.system_prompt

    def test_tools_registered(self, tmp_path):
        from mini_agent.tools.base import Tool, ToolResult

        class FakeTool(Tool):
            @property
            def name(self): return "fake"
            @property
            def description(self): return "A fake tool"
            @property
            def parameters(self): return {"type": "object", "properties": {}}
            async def execute(self) -> ToolResult:
                return ToolResult(success=True, content="ok")

        config = AgentConfig(tools=[FakeTool()])
        agent = Agent(llm_client=make_mock_llm(), config=config, workspace_dir=tmp_path)
        assert "fake" in agent.tools

    def test_path_policy_in_system_prompt(self, tmp_path):
        config = AgentConfig(system_prompt="Hello")
        agent = Agent(llm_client=make_mock_llm(), config=config, workspace_dir=tmp_path)
        assert "Path Access Policy" in agent.system_prompt


class TestSubAgentRegistration:
    def test_register_sub_agent(self, tmp_path):
        config = AgentConfig()
        agent = Agent(llm_client=make_mock_llm(), config=config, workspace_dir=tmp_path)
        sub_config = AgentConfig(agent_id="coder", name="Coder", description="Writes code")
        agent.register_sub_agent("coder", sub_config)
        assert "coder" in agent.sub_agent_names
        assert "delegate_to_agent" in agent.tools

    def test_register_enables_delegation(self, tmp_path):
        config = AgentConfig(can_delegate=False)
        agent = Agent(llm_client=make_mock_llm(), config=config, workspace_dir=tmp_path)
        assert "delegate_to_agent" not in agent.tools
        agent.register_sub_agent("helper", AgentConfig(description="Helps"))
        assert "delegate_to_agent" in agent.tools
        assert agent.config.can_delegate is True

    def test_system_prompt_updated_after_registration(self, tmp_path):
        config = AgentConfig(system_prompt="Base prompt")
        agent = Agent(llm_client=make_mock_llm(), config=config, workspace_dir=tmp_path)
        assert "Available Sub-Agents" not in agent.system_prompt
        agent.register_sub_agent("coder", AgentConfig(description="Writes code"))
        assert "Available Sub-Agents" in agent.system_prompt
        assert "coder" in agent.system_prompt


class TestStateToolsRegistration:
    def test_state_tools_registered_with_readwrite(self, tmp_path):
        config = AgentConfig(state_access="readwrite")
        state = SharedState()
        agent = Agent(
            llm_client=make_mock_llm(), config=config,
            workspace_dir=tmp_path, shared_state=state,
        )
        assert "state_read" in agent.tools
        assert "state_write" in agent.tools
        assert "state_list" in agent.tools

    def test_state_tools_read_only(self, tmp_path):
        config = AgentConfig(state_access="read")
        state = SharedState()
        agent = Agent(
            llm_client=make_mock_llm(), config=config,
            workspace_dir=tmp_path, shared_state=state,
        )
        assert "state_read" in agent.tools
        assert "state_list" in agent.tools
        assert "state_write" not in agent.tools

    def test_no_state_tools_without_shared_state(self, tmp_path):
        config = AgentConfig(state_access="readwrite")
        agent = Agent(llm_client=make_mock_llm(), config=config, workspace_dir=tmp_path)
        assert "state_read" not in agent.tools

    def test_state_tools_write_only(self, tmp_path):
        config = AgentConfig(state_access="write")
        state = SharedState()
        agent = Agent(
            llm_client=make_mock_llm(), config=config,
            workspace_dir=tmp_path, shared_state=state,
        )
        assert "state_write" in agent.tools
        assert "state_read" not in agent.tools
        assert "state_list" not in agent.tools

    def test_no_state_tools_with_none_access(self, tmp_path):
        config = AgentConfig(state_access="none")
        state = SharedState()
        agent = Agent(
            llm_client=make_mock_llm(), config=config,
            workspace_dir=tmp_path, shared_state=state,
        )
        assert "state_read" not in agent.tools


class TestCanDelegateWithoutSubAgents:
    def test_can_delegate_true_but_no_sub_agents(self, tmp_path):
        """can_delegate=True without registered sub-agents should NOT add delegation tool."""
        config = AgentConfig(can_delegate=True)
        agent = Agent(llm_client=make_mock_llm(), config=config, workspace_dir=tmp_path)
        assert "delegate_to_agent" not in agent.tools


class TestSubAgentNamesAccessor:
    def test_empty_by_default(self, tmp_path):
        agent = Agent(llm_client=make_mock_llm(), config=AgentConfig(), workspace_dir=tmp_path)
        assert agent.sub_agent_names == []

    def test_returns_registered_names(self, tmp_path):
        agent = Agent(llm_client=make_mock_llm(), config=AgentConfig(), workspace_dir=tmp_path)
        agent.register_sub_agent("a", AgentConfig())
        agent.register_sub_agent("b", AgentConfig())
        assert sorted(agent.sub_agent_names) == ["a", "b"]
