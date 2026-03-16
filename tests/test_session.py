# tests/test_session.py
"""Tests for Session API."""
import pytest
from unittest.mock import MagicMock, AsyncMock
from mini_agent.agent_config import AgentConfig
from mini_agent.session import Session
from mini_agent.shared_state import SharedState


def make_mock_llm():
    llm = MagicMock()
    llm.generate_stream = AsyncMock()
    return llm


class TestSessionCreate:
    def test_creates_session_with_agent(self, tmp_path):
        session = Session.create(
            llm_client=make_mock_llm(),
            system_prompt="Hello",
            workspace_dir=tmp_path,
        )
        assert session.agent is not None
        assert session.shared_state is not None
        assert isinstance(session.shared_state, SharedState)
        assert session.agent.config.agent_id == "main"

    def test_creates_session_with_sub_agents(self, tmp_path):
        session = Session.create(
            llm_client=make_mock_llm(),
            sub_agents={
                "coder": AgentConfig(name="Coder", description="Writes code"),
            },
            workspace_dir=tmp_path,
        )
        assert session.agent.config.can_delegate is True
        assert "coder" in session.agent.sub_agent_names
        assert "delegate_to_agent" in session.agent.tools

    def test_no_delegation_without_sub_agents(self, tmp_path):
        session = Session.create(llm_client=make_mock_llm(), workspace_dir=tmp_path)
        assert session.agent.config.can_delegate is False
        assert "delegate_to_agent" not in session.agent.tools

    def test_state_tools_always_registered(self, tmp_path):
        """Session.create() always creates SharedState, so state tools should be registered."""
        session = Session.create(llm_client=make_mock_llm(), workspace_dir=tmp_path)
        assert session.shared_state is not None
        assert "state_read" in session.agent.tools
        assert "state_write" in session.agent.tools

    def test_custom_context_config(self, tmp_path):
        from mini_agent.context.config import ContextConfig
        ctx = ContextConfig.from_mode("claude_code")
        session = Session.create(llm_client=make_mock_llm(), context_config=ctx, workspace_dir=tmp_path)
        assert session.agent.context_manager is not None

    def test_workspace_dir_passed(self, tmp_path):
        session = Session.create(
            llm_client=make_mock_llm(),
            workspace_dir=tmp_path,
        )
        assert session.agent.workspace_dir == tmp_path


class TestSessionCreateOrchestrator:
    def test_creates_orchestrator(self, tmp_path):
        session = Session.create_orchestrator(
            llm_client=make_mock_llm(),
            workers={
                "coder": AgentConfig(name="Coder", description="Writes code"),
            },
            workspace_dir=tmp_path,
        )
        assert session.agent.config.agent_id == "orchestrator"
        assert session.agent.config.can_delegate is True
        assert "coder" in session.agent.sub_agent_names

    def test_orchestrator_default_prompt(self, tmp_path):
        session = Session.create_orchestrator(
            llm_client=make_mock_llm(),
            workers={"w": AgentConfig()},
            workspace_dir=tmp_path,
        )
        assert "orchestrator" in session.agent.config.system_prompt.lower()

    def test_orchestrator_custom_prompt(self, tmp_path):
        session = Session.create_orchestrator(
            llm_client=make_mock_llm(),
            workers={"w": AgentConfig()},
            orchestrator_prompt="Custom orchestrator.",
            workspace_dir=tmp_path,
        )
        assert session.agent.config.system_prompt == "Custom orchestrator."


class TestSessionDelegation:
    def test_add_user_message(self, tmp_path):
        session = Session.create(llm_client=make_mock_llm(), workspace_dir=tmp_path)
        session.add_user_message("Hello")
        status = session.get_status()
        assert status["turn"] == 1

    def test_get_status(self, tmp_path):
        session = Session.create(
            llm_client=make_mock_llm(),
            sub_agents={"coder": AgentConfig(description="Codes")},
            workspace_dir=tmp_path,
        )
        status = session.get_status()
        assert status["agent_id"] == "main"
        assert status["turn"] == 0
        assert status["shared_state_keys"] == []
        assert "coder" in status["sub_agents"]
        assert "context" in status
