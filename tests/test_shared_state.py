"""Tests for SharedState cross-agent data store."""
import asyncio
from mini_agent.shared_state import SharedState, StateEntry


class TestSharedStateBasicOps:
    async def test_set_and_get(self):
        state = SharedState()
        await state.set("key1", "value1", agent_id="main")
        result = await state.get("key1")
        assert result == "value1"

    async def test_get_nonexistent_returns_none(self):
        state = SharedState()
        result = await state.get("missing")
        assert result is None

    async def test_overwrite_value(self):
        state = SharedState()
        await state.set("key1", "old", agent_id="main")
        await state.set("key1", "new", agent_id="sub1")
        result = await state.get("key1")
        assert result == "new"

    async def test_keys_all(self):
        state = SharedState()
        await state.set("a", 1, agent_id="main")
        await state.set("b", 2, agent_id="main")
        keys = await state.keys()
        assert sorted(keys) == ["a", "b"]

    async def test_keys_with_prefix(self):
        state = SharedState()
        await state.set("data.sales", 1, agent_id="main")
        await state.set("data.costs", 2, agent_id="main")
        await state.set("config.model", "x", agent_id="main")
        keys = await state.keys(prefix="data.")
        assert sorted(keys) == ["data.costs", "data.sales"]

    async def test_delete_existing(self):
        state = SharedState()
        await state.set("key1", "val", agent_id="main")
        result = await state.delete("key1")
        assert result is True
        assert await state.get("key1") is None

    async def test_delete_nonexistent(self):
        state = SharedState()
        result = await state.delete("missing")
        assert result is False


class TestSharedStateSnapshot:
    async def test_snapshot_empty(self):
        state = SharedState()
        assert state.snapshot() == {}

    async def test_snapshot_includes_schema_hint(self):
        state = SharedState()
        await state.set("df", "data", agent_id="analyst", schema_hint="DataFrame(200 rows)")
        snap = state.snapshot()
        assert "df" in snap
        assert "DataFrame(200 rows)" in snap["df"]
        assert "analyst" in snap["df"]

    async def test_snapshot_is_sync(self):
        """snapshot() should be callable without await."""
        state = SharedState()
        await state.set("k", "v", agent_id="main")
        result = state.snapshot()
        assert isinstance(result, dict)


class TestStateEntry:
    async def test_entry_metadata(self):
        state = SharedState()
        await state.set("key1", "val", agent_id="sub1", schema_hint="str", ttl_turns=5)
        entry = state._store["key1"]
        assert entry.written_by == "sub1"
        assert entry.schema_hint == "str"
        assert entry.ttl_turns == 5
        assert entry.written_at is not None
