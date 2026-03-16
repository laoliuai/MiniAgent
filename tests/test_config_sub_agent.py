# tests/test_config_sub_agent.py
"""Tests for sub-agent config parsing."""
import pytest
import yaml
import tempfile
from pathlib import Path
from mini_agent.config import Config, AgentSettings, SubAgentEntry


class TestAgentSettingsRename:
    def test_agent_settings_exists(self):
        """AgentSettings should exist (renamed from AgentConfig)."""
        settings = AgentSettings()
        assert settings.max_steps_per_turn == 30
        assert settings.max_steps_total == 50
        assert settings.workspace_dir == "./workspace"

    def test_backward_compat_max_steps(self):
        """If only max_steps is in YAML (flat top level), it maps to max_steps_per_turn."""
        yaml_content = """
api_key: test-key
api_base: https://api.test.com
model: test-model
max_steps: 25
workspace_dir: ./workspace
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = Config.from_yaml(f.name)
        assert config.agent.max_steps_per_turn == 25
        assert config.agent.max_steps_total == 50  # default

    def test_both_step_limits(self):
        """Both max_steps_per_turn and max_steps_total can be set."""
        yaml_content = """
api_key: test-key
api_base: https://api.test.com
model: test-model
max_steps_per_turn: 20
max_steps_total: 100
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = Config.from_yaml(f.name)
        assert config.agent.max_steps_per_turn == 20
        assert config.agent.max_steps_total == 100


class TestSubAgentEntry:
    def test_defaults(self):
        entry = SubAgentEntry()
        assert entry.description == ""
        assert entry.system_prompt == ""
        assert entry.system_prompt_path == ""
        assert entry.model is None
        assert entry.tools == []

    def test_custom_values(self):
        entry = SubAgentEntry(
            description="Writes code",
            system_prompt="You are a coder.",
            tools=["bash", "read", "write"],
        )
        assert entry.description == "Writes code"
        assert entry.tools == ["bash", "read", "write"]


class TestSubAgentConfigParsing:
    def test_parse_sub_agents_from_yaml(self):
        yaml_content = """
api_key: test-key
api_base: https://api.test.com
model: test-model
sub_agents:
  coder:
    description: "Writes code"
    system_prompt: "You are a coder."
    tools: ["bash", "read", "write"]
  researcher:
    description: "Researches"
    system_prompt: "You research."
    tools: ["bash", "read"]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = Config.from_yaml(f.name)
        assert "coder" in config.sub_agents
        assert "researcher" in config.sub_agents
        assert config.sub_agents["coder"].description == "Writes code"
        assert config.sub_agents["coder"].tools == ["bash", "read", "write"]

    def test_empty_sub_agents(self):
        yaml_content = """
api_key: test-key
api_base: https://api.test.com
model: test-model
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = Config.from_yaml(f.name)
        assert config.sub_agents == {}
