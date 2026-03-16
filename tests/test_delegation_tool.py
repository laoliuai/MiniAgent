"""Tests for DelegationTool."""
import pytest
from mini_agent.agent_config import AgentConfig
from mini_agent.tools.delegation_tool import DelegationTool


def make_runner(result: str = "done", raise_error: Exception | None = None):
    """Create a mock SubAgentRunner."""
    async def runner(config: AgentConfig, task: str) -> str:
        if raise_error:
            raise raise_error
        return result
    return runner


class TestDelegationToolSchema:
    def test_name(self):
        tool = DelegationTool(
            sub_agents={"coder": AgentConfig(name="Coder", description="Writes code")},
            runner=make_runner(),
        )
        assert tool.name == "delegate_to_agent"

    def test_description_lists_agents(self):
        tool = DelegationTool(
            sub_agents={
                "coder": AgentConfig(name="Coder", description="Writes code"),
                "analyst": AgentConfig(name="Analyst", description="Analyzes data"),
            },
            runner=make_runner(),
        )
        desc = tool.description
        assert "coder" in desc
        assert "Writes code" in desc
        assert "analyst" in desc

    def test_parameters_enum(self):
        tool = DelegationTool(
            sub_agents={
                "a": AgentConfig(),
                "b": AgentConfig(),
            },
            runner=make_runner(),
        )
        params = tool.parameters
        enum = params["properties"]["agent_name"]["enum"]
        assert sorted(enum) == ["a", "b"]

    def test_schema_format(self):
        tool = DelegationTool(
            sub_agents={"x": AgentConfig()},
            runner=make_runner(),
        )
        schema = tool.to_schema()
        assert schema["name"] == "delegate_to_agent"
        assert "input_schema" in schema


class TestDelegationToolExecute:
    async def test_successful_delegation(self):
        tool = DelegationTool(
            sub_agents={"coder": AgentConfig(name="Coder", description="Writes code")},
            runner=make_runner(result="Task completed successfully."),
        )
        result = await tool.execute(agent_name="coder", task="Write hello.py")
        assert result.success is True
        assert "Task completed successfully." in result.content

    async def test_unknown_agent(self):
        tool = DelegationTool(
            sub_agents={"coder": AgentConfig()},
            runner=make_runner(),
        )
        result = await tool.execute(agent_name="unknown", task="Do something")
        assert result.success is False
        assert "Unknown agent" in result.error

    async def test_runner_exception(self):
        tool = DelegationTool(
            sub_agents={"coder": AgentConfig(name="Coder")},
            runner=make_runner(raise_error=RuntimeError("LLM failed")),
        )
        result = await tool.execute(agent_name="coder", task="Do something")
        assert result.success is False
        assert "failed" in result.error.lower()
