"""Tests for AgentConfig dataclass."""
from mini_agent.agent_config import AgentConfig


class TestAgentConfigDefaults:
    def test_default_values(self):
        config = AgentConfig()
        assert config.agent_id == "main"
        assert config.name == "Assistant"
        assert config.description == ""
        assert config.model is None
        assert config.system_prompt == ""
        assert config.tools == []
        assert config.context_config is None
        assert config.can_delegate is False
        assert config.max_delegation_depth == 1
        assert config.max_steps_per_turn == 30
        assert config.max_steps_total == 50
        assert config.state_access == "readwrite"

    def test_custom_values(self):
        config = AgentConfig(
            agent_id="coder",
            name="Coder",
            description="Writes code",
            model="gpt-4",
            system_prompt="You are a coder.",
            can_delegate=True,
            max_delegation_depth=2,
            max_steps_per_turn=10,
            max_steps_total=20,
            state_access="read",
        )
        assert config.agent_id == "coder"
        assert config.name == "Coder"
        assert config.model == "gpt-4"
        assert config.can_delegate is True
        assert config.max_delegation_depth == 2
        assert config.state_access == "read"

    def test_tools_list_independence(self):
        """Each AgentConfig gets its own tools list (no shared mutable default)."""
        c1 = AgentConfig()
        c2 = AgentConfig()
        c1.tools.append("fake_tool")
        assert c2.tools == []
