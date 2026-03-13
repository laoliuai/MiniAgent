"""Test cases for TodoTool."""

import pytest

from mini_agent.tools.todo_tool import TodoStore, TodoTool


class TestTodoStore:
    """Unit tests for TodoStore."""

    def test_add(self):
        store = TodoStore()
        item = store.add("Task 1")
        assert item.id == 1
        assert item.content == "Task 1"
        assert item.status == "pending"

    def test_add_multiple(self):
        store = TodoStore()
        item1 = store.add("Task 1")
        item2 = store.add("Task 2")
        assert item1.id == 1
        assert item2.id == 2

    def test_update_status(self):
        store = TodoStore()
        store.add("Task 1")
        updated = store.update(1, status="in_progress")
        assert updated.status == "in_progress"
        assert updated.content == "Task 1"

    def test_update_content(self):
        store = TodoStore()
        store.add("Task 1")
        updated = store.update(1, content="Updated Task 1")
        assert updated.content == "Updated Task 1"

    def test_update_nonexistent_raises(self):
        store = TodoStore()
        with pytest.raises(KeyError):
            store.update(999)

    def test_remove(self):
        store = TodoStore()
        store.add("Task 1")
        store.remove(1)
        assert store.list_all() == []

    def test_remove_nonexistent_raises(self):
        store = TodoStore()
        with pytest.raises(KeyError):
            store.remove(999)

    def test_list_all(self):
        store = TodoStore()
        store.add("Task 1")
        store.add("Task 2")
        items = store.list_all()
        assert len(items) == 2

    def test_summary(self):
        store = TodoStore()
        store.add("Task 1")
        store.add("Task 2", status="completed")
        assert store.summary() == "Progress: 1/2 completed"


class TestTodoTool:
    """Integration tests for TodoTool."""

    @pytest.mark.asyncio
    async def test_add_todos(self):
        tool = TodoTool()
        result = await tool.execute(
            operation="add",
            items=[{"content": "Task A"}, {"content": "Task B", "status": "in_progress"}],
        )
        assert result.success
        assert "#1" in result.content
        assert "#2" in result.content

    @pytest.mark.asyncio
    async def test_add_requires_items(self):
        tool = TodoTool()
        result = await tool.execute(operation="add")
        assert not result.success
        assert "items" in result.error

    @pytest.mark.asyncio
    async def test_update_todos(self):
        tool = TodoTool()
        await tool.execute(operation="add", items=[{"content": "Task A"}])
        result = await tool.execute(
            operation="update",
            items=[{"id": 1, "status": "completed"}],
        )
        assert result.success
        assert "completed" in result.content

    @pytest.mark.asyncio
    async def test_update_nonexistent(self):
        tool = TodoTool()
        result = await tool.execute(
            operation="update",
            items=[{"id": 999, "status": "completed"}],
        )
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_list_empty(self):
        tool = TodoTool()
        result = await tool.execute(operation="list")
        assert result.success
        assert "No todos" in result.content

    @pytest.mark.asyncio
    async def test_list_with_items(self):
        tool = TodoTool()
        await tool.execute(operation="add", items=[
            {"content": "Task A"},
            {"content": "Task B", "status": "in_progress"},
        ])
        result = await tool.execute(operation="list")
        assert result.success
        assert "Task A" in result.content
        assert "Task B" in result.content
        assert "[~]" in result.content  # in_progress icon
        assert "Progress:" in result.content

    @pytest.mark.asyncio
    async def test_remove_todo(self):
        tool = TodoTool()
        await tool.execute(operation="add", items=[{"content": "Task A"}])
        result = await tool.execute(operation="remove", id=1)
        assert result.success
        assert "#1" in result.content

    @pytest.mark.asyncio
    async def test_remove_requires_id(self):
        tool = TodoTool()
        result = await tool.execute(operation="remove")
        assert not result.success
        assert "id" in result.error

    @pytest.mark.asyncio
    async def test_unknown_operation(self):
        tool = TodoTool()
        result = await tool.execute(operation="invalid")
        assert not result.success
        assert "Unknown operation" in result.error

    @pytest.mark.asyncio
    async def test_schema(self):
        tool = TodoTool()
        schema = tool.to_schema()
        assert schema["name"] == "todo"
        assert "input_schema" in schema
        assert "operation" in schema["input_schema"]["properties"]
        assert schema["input_schema"]["required"] == ["operation"]

    @pytest.mark.asyncio
    async def test_store_isolation(self):
        """Each TodoTool instance has its own store."""
        tool1 = TodoTool()
        tool2 = TodoTool()
        await tool1.execute(operation="add", items=[{"content": "Tool1 task"}])
        result = await tool2.execute(operation="list")
        assert "No todos" in result.content
