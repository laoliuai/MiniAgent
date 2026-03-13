"""Session-scoped task tracking tool."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .base import Tool, ToolResult


@dataclass
class TodoItem:
    """A single todo item."""

    id: int
    content: str
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class TodoStore:
    """In-memory todo storage (session-scoped, no persistence)."""

    def __init__(self):
        self._items: dict[int, TodoItem] = {}
        self._next_id: int = 1

    def add(self, content: str, status: str = "pending") -> TodoItem:
        """Add a new todo item."""
        item = TodoItem(id=self._next_id, content=content, status=status)
        self._items[self._next_id] = item
        self._next_id += 1
        return item

    def update(self, item_id: int, content: str | None = None, status: str | None = None) -> TodoItem:
        """Update an existing todo item. Raises KeyError if not found."""
        item = self._items[item_id]
        if content is not None:
            item.content = content
        if status is not None:
            item.status = status
        return item

    def remove(self, item_id: int) -> None:
        """Remove a todo item. Raises KeyError if not found."""
        del self._items[item_id]

    def list_all(self) -> list[TodoItem]:
        """List all todo items."""
        return list(self._items.values())

    def summary(self) -> str:
        """Return progress summary string."""
        total = len(self._items)
        completed = sum(1 for item in self._items.values() if item.status == "completed")
        return f"Progress: {completed}/{total} completed"


class TodoTool(Tool):
    """Session-level task tracking for complex multi-step work."""

    def __init__(self):
        self._store = TodoStore()

    @property
    def name(self) -> str:
        return "todo"

    @property
    def description(self) -> str:
        return (
            "Manage a task list to track progress on complex multi-step work. "
            "Operations: add (create tasks), update (change status/content), "
            "list (show all tasks), remove (delete a task). "
            "For complex tasks (3+ steps), create a todo list first, then work through it."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "update", "list", "remove"],
                    "description": "Operation to perform",
                },
                "items": {
                    "type": "array",
                    "description": "Items for add/update operations",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "Item ID (required for update)"},
                            "content": {"type": "string", "description": "Task description"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                    },
                },
                "id": {
                    "type": "integer",
                    "description": "Item ID for remove operation",
                },
            },
            "required": ["operation"],
        }

    async def execute(  # pylint: disable=arguments-differ
        self,
        operation: str,
        items: list[dict] | None = None,
        id: int | None = None,  # noqa: A002
    ) -> ToolResult:
        """Execute todo operation."""
        try:
            match operation:
                case "add":
                    if not items:
                        return ToolResult(success=False, content="", error="'items' required for add")
                    added = []
                    for item in items:
                        todo = self._store.add(item["content"], item.get("status", "pending"))
                        added.append(f"#{todo.id}")
                    return ToolResult(success=True, content=f"Added {len(added)} todo(s): {', '.join(added)}")

                case "update":
                    if not items:
                        return ToolResult(success=False, content="", error="'items' required for update")
                    updates = []
                    for item in items:
                        todo = self._store.update(item["id"], item.get("content"), item.get("status"))
                        updates.append(f"#{todo.id} -> {todo.status}")
                    return ToolResult(success=True, content=f"Updated: {', '.join(updates)}")

                case "list":
                    todos = self._store.list_all()
                    if not todos:
                        return ToolResult(success=True, content="No todos. Use add to create tasks.")
                    lines = []
                    status_icon = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
                    for t in todos:
                        lines.append(f"  #{t.id} {status_icon.get(t.status, '[ ]')} {t.content}")
                    lines.append(f"\n{self._store.summary()}")
                    return ToolResult(success=True, content="Todo List:\n" + "\n".join(lines))

                case "remove":
                    if id is None:
                        return ToolResult(success=False, content="", error="'id' required for remove")
                    self._store.remove(id)
                    return ToolResult(success=True, content=f"Removed todo #{id}")

                case _:
                    return ToolResult(success=False, content="", error=f"Unknown operation: {operation}")

        except KeyError as e:
            return ToolResult(success=False, content="", error=f"Todo not found: {e}")
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))
