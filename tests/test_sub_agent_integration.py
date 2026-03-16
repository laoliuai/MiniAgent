"""Integration tests for sub-agent delegation and SharedState."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from mini_agent.agent import Agent
from mini_agent.agent_config import AgentConfig
from mini_agent.session import Session
from mini_agent.shared_state import SharedState
from mini_agent.schema import LLMStreamChunk, LLMStreamChunkType


def make_mock_llm_with_responses(responses: list[list[LLMStreamChunk]]):
    """Create a mock LLM that returns predefined response sequences.

    Each call to generate_stream returns the next response in the list.
    """
    llm = MagicMock()
    call_count = [0]

    async def mock_generate_stream(messages, tools=None, model=None):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        for chunk in responses[idx]:
            yield chunk

    llm.generate_stream = mock_generate_stream
    llm.generate = AsyncMock(return_value=MagicMock(content="summary"))
    return llm


def text_response(text: str) -> list[LLMStreamChunk]:
    """Create a simple text-only response (no tool calls)."""
    return [
        LLMStreamChunk(type=LLMStreamChunkType.TEXT_DELTA, content=text),
        LLMStreamChunk(type=LLMStreamChunkType.DONE, finish_reason="stop"),
    ]


def tool_call_response(tool_name: str, tool_id: str, arguments: dict) -> list[LLMStreamChunk]:
    """Create a response with a single tool call."""
    import json
    return [
        LLMStreamChunk(
            type=LLMStreamChunkType.TOOL_CALL_START,
            tool_call_id=tool_id, tool_name=tool_name,
        ),
        LLMStreamChunk(
            type=LLMStreamChunkType.TOOL_CALL_DELTA,
            tool_call_id=tool_id, tool_arguments=json.dumps(arguments),
        ),
        LLMStreamChunk(
            type=LLMStreamChunkType.TOOL_CALL_END,
            tool_call_id=tool_id,
        ),
        LLMStreamChunk(type=LLMStreamChunkType.DONE, finish_reason="tool_use"),
    ]


class TestDelegationFlow:
    async def test_parent_delegates_to_sub_agent(self, tmp_path):
        """Parent calls delegate_to_agent, sub-agent runs and returns result."""
        # Sub-agent LLM: just returns text immediately
        sub_response = text_response("Sub-agent completed the task.")

        # Parent LLM call 1: calls delegate_to_agent tool
        # Parent LLM call 2: synthesizes final answer after delegation
        parent_call_1 = tool_call_response(
            "delegate_to_agent", "tc_1",
            {"agent_name": "coder", "task": "Write hello.py"},
        )
        parent_call_2 = text_response("Done! The coder wrote hello.py.")

        # The mock LLM serves responses in order: parent call 1, sub-agent call, parent call 2.
        # Both parent and sub-agent share the same LLM instance (as in real usage).
        llm = make_mock_llm_with_responses([parent_call_1, sub_response, parent_call_2])

        session = Session.create(
            llm_client=llm,
            system_prompt="You are helpful.",
            sub_agents={
                "coder": AgentConfig(
                    name="Coder",
                    description="Writes code",
                    system_prompt="You write code.",
                ),
            },
            workspace_dir=tmp_path,
        )

        session.add_user_message("Write hello.py")
        result = await session.run()
        assert "The coder wrote hello.py" in result

    async def test_depth_limit_enforced(self, tmp_path):
        """Sub-agent at max depth cannot re-delegate."""
        config = AgentConfig(
            system_prompt="Test",
            can_delegate=True,
            max_delegation_depth=1,
        )
        agent = Agent(llm_client=MagicMock(), config=config, workspace_dir=tmp_path)
        agent._delegation_depth = 1  # Already at max depth

        # The runner should block delegation
        runner = agent._make_sub_agent_runner()
        result = await runner(AgentConfig(name="sub"), "do something")
        assert "blocked" in result.lower() or "depth" in result.lower()


class TestSharedStateIntegration:
    async def test_state_tools_registered_via_session(self, tmp_path):
        session = Session.create(llm_client=MagicMock(), workspace_dir=tmp_path)
        # Session creates SharedState, so state tools should be registered
        assert "state_read" in session.agent.tools
        assert "state_write" in session.agent.tools
        assert "state_list" in session.agent.tools

    async def test_state_write_read_across_session(self, tmp_path):
        session = Session.create(llm_client=MagicMock(), workspace_dir=tmp_path)
        # Write via tool
        write_tool = session.agent.tools["state_write"]
        await write_tool.execute(key="test_key", value="test_value", schema_hint="string")

        # Read via tool
        read_tool = session.agent.tools["state_read"]
        result = await read_tool.execute(key="test_key")
        assert result.success is True
        assert "test_value" in result.content

    async def test_shared_state_visible_in_status(self, tmp_path):
        session = Session.create(llm_client=MagicMock(), workspace_dir=tmp_path)
        write_tool = session.agent.tools["state_write"]
        await write_tool.execute(key="data", value="123")
        status = session.get_status()
        assert "data" in status["shared_state_keys"]


class TestModelOverrideIntegration:
    async def test_sub_agent_uses_own_model(self, tmp_path):
        """Sub-agent with model override passes it to LLM generate_stream."""
        captured_models = []

        async def mock_generate_stream(messages, tools=None, model=None):
            captured_models.append(model)
            for chunk in text_response("done"):
                yield chunk

        llm = MagicMock()
        llm.generate_stream = mock_generate_stream
        llm.generate = AsyncMock(return_value=MagicMock(content="summary"))

        session = Session.create(
            llm_client=llm,
            system_prompt="Parent",
            sub_agents={
                "coder": AgentConfig(
                    name="Coder",
                    description="Writes code",
                    system_prompt="Code.",
                    model="special-model-v2",
                ),
            },
            workspace_dir=tmp_path,
        )

        # Directly run the sub-agent via the runner to verify model is forwarded
        runner = session.agent._make_sub_agent_runner()
        sub_config = session.agent._sub_agents["coder"]
        await runner(sub_config, "Write hello.py")

        # The sub-agent's generate_stream call should have model="special-model-v2"
        assert "special-model-v2" in captured_models
