"""Tests for SharedState tool wrappers."""
import pytest
from mini_agent.shared_state import SharedState
from mini_agent.tools.state_tools import StateReadTool, StateWriteTool, StateListTool


class TestStateReadTool:
    async def test_read_existing_key(self):
        state = SharedState()
        await state.set("key1", "value1", agent_id="test")
        tool = StateReadTool(state)
        result = await tool.execute(key="key1")
        assert result.success is True
        assert "value1" in result.content

    async def test_read_nonexistent_key(self):
        state = SharedState()
        tool = StateReadTool(state)
        result = await tool.execute(key="missing")
        assert result.success is True
        assert "not found" in result.content.lower() or "null" in result.content.lower()

    async def test_schema(self):
        tool = StateReadTool(SharedState())
        schema = tool.to_schema()
        assert schema["name"] == "state_read"
        assert "key" in schema["input_schema"]["properties"]


class TestStateWriteTool:
    async def test_write_value(self):
        state = SharedState()
        tool = StateWriteTool(state, agent_id="main")
        result = await tool.execute(key="k1", value="v1", schema_hint="string")
        assert result.success is True
        stored = await state.get("k1")
        assert stored == "v1"

    async def test_write_records_agent_id(self):
        state = SharedState()
        tool = StateWriteTool(state, agent_id="coder")
        await tool.execute(key="k1", value="v1")
        entry = state._store["k1"]
        assert entry.written_by == "coder"

    async def test_schema(self):
        tool = StateWriteTool(SharedState(), agent_id="main")
        schema = tool.to_schema()
        assert schema["name"] == "state_write"
        assert "key" in schema["input_schema"]["properties"]
        assert "value" in schema["input_schema"]["properties"]


class TestStateListTool:
    async def test_list_all(self):
        state = SharedState()
        await state.set("a", 1, agent_id="main")
        await state.set("b", 2, agent_id="main")
        tool = StateListTool(state)
        result = await tool.execute()
        assert result.success is True
        assert "a" in result.content
        assert "b" in result.content

    async def test_list_with_prefix(self):
        state = SharedState()
        await state.set("data.x", 1, agent_id="main")
        await state.set("config.y", 2, agent_id="main")
        tool = StateListTool(state)
        result = await tool.execute(prefix="data.")
        assert result.success is True
        assert "data.x" in result.content
        assert "config.y" not in result.content

    async def test_schema(self):
        tool = StateListTool(SharedState())
        schema = tool.to_schema()
        assert schema["name"] == "state_list"
