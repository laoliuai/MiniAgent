# Phase 1: Tool Layer Refactoring + GrepTool + TodoTool + Logging Enhancement

> Date: 2026-03-13
> Status: Design approved
> Part of: [Feature Roadmap](../../2026-03-13-feature-roadmap.md) Phase 1

---

## Overview

Phase 1 focuses on strengthening the foundation of MiniAgent: improving existing tools, adding missing tools (Grep, Todo), and making logging configurable. These changes are independent of each other and can be developed in parallel.

**Scope:** 6 modifications, 4 new files, ~10 changed files.

---

## 1. ReadTool Improvements

**File:** `mini_agent/tools/file_tools.py` (class `ReadTool`)

### Current Issues
- Reads entire file into memory via `f.readlines()` regardless of size
- No long-line truncation protection
- Token truncation threshold hardcoded to 32000
- No file metadata in output

### Changes

#### 1.1 Large File Protection
When a file exceeds `MAX_LINES` (2000) and the caller did **not** specify `offset`/`limit`, return a truncated preview instead of the full file:

```python
MAX_LINES = 2000
PREVIEW_LINES = 100  # enough for LLM to get structure context

lines = f.readlines()
total_lines = len(lines)

if total_lines > MAX_LINES and offset is None and limit is None:
    # Return first PREVIEW_LINES + file info + hint
    selected_lines = lines[:PREVIEW_LINES]
    # ... format with line numbers ...
    header = f"[File: {path}, {total_lines} lines, {file_size} bytes]\n"
    footer = f"\n[Showing first {PREVIEW_LINES} of {total_lines} lines. Use offset/limit to read specific ranges.]"
    return ToolResult(success=True, content=header + content + footer)
```

> **Note:** Preview set to 100 lines (not 50) to give the LLM enough structural context for orientation. The token truncation (32K) remains as the ultimate safety net.

#### 1.2 Long Line Truncation
Truncate individual lines exceeding `MAX_LINE_LENGTH` (2000 characters):

```python
MAX_LINE_LENGTH = 2000

line_content = line.rstrip("\n")
if len(line_content) > MAX_LINE_LENGTH:
    original_len = len(line_content)
    line_content = line_content[:MAX_LINE_LENGTH] + f"... [truncated, {original_len} chars total]"
```

#### 1.3 File Metadata Header
Always prepend a metadata line to help the LLM understand file context:

```
[File: path/to/file.py, 342 lines, 12.5 KB]
```

When showing a partial range:

```
[File: path/to/file.py, 342 lines, 12.5 KB]
[Showing lines 100-200 of 342. Use offset/limit to read more.]
```

#### 1.4 What Stays the Same
- Line-based addressing (industry consensus — Claude Code, Aider, SWE-agent all use it)
- `LINENUM|CONTENT` output format
- Token-level truncation as last resort safety net
- Workspace-relative path resolution

---

## 2. WriteTool Improvements

**File:** `mini_agent/tools/file_tools.py` (class `WriteTool`)

### Changes
Minimal — only enhance the return message with write statistics:

```python
line_count = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
byte_count = len(content.encode('utf-8'))
return ToolResult(
    success=True,
    content=f"Successfully wrote to {file_path} ({line_count} lines, {byte_count} bytes)"
)
```

Everything else stays the same.

---

## 3. EditTool Improvements

**File:** `mini_agent/tools/file_tools.py` (class `EditTool`)

### Current Issues
- `content.replace(old_str, new_str)` replaces ALL occurrences when description says "must be unique" — behavior/docs mismatch
- No `replace_all` option for intentional batch replacement
- Return message lacks line number information

### Changes

#### 3.1 Add `replace_all` Parameter

```python
# Schema addition
"replace_all": {
    "type": "boolean",
    "description": "Replace all occurrences (default: false, requires unique match)",
    "default": False,
}
```

#### 3.2 Complete Execute Flow (replaces current `execute` method)

The full flow in a single linear block — line number calculation happens **before** replacement:

```python
async def execute(self, path: str, old_str: str, new_str: str, replace_all: bool = False) -> ToolResult:
    try:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.workspace_dir / file_path

        if not file_path.exists():
            return ToolResult(success=False, content="", error=f"File not found: {path}")

        content = file_path.read_text(encoding="utf-8")

        # 1. Count matches
        count = content.count(old_str)

        if count == 0:
            return ToolResult(success=False, content="",
                              error=f"Text not found in file: {old_str}")

        if count > 1 and not replace_all:
            return ToolResult(success=False, content="",
                              error=f"Found {count} matches. Provide more context for a unique match, or set replace_all=true.")

        # 2. Calculate line number of first match (BEFORE replacement)
        match_offset = content.index(old_str)
        match_line = content[:match_offset].count('\n') + 1

        # 3. Perform replacement
        if replace_all:
            new_content = content.replace(old_str, new_str)
        else:
            new_content = content.replace(old_str, new_str, 1)

        file_path.write_text(new_content, encoding="utf-8")

        # 4. Return with line info
        if replace_all:
            msg = f"Edited {file_path}: replaced {count} occurrence(s) (first at line {match_line})"
        else:
            msg = f"Edited {file_path}: replaced at line {match_line}"

        return ToolResult(success=True, content=msg)

    except Exception as e:
        return ToolResult(success=False, content="", error=str(e))
```

---

## 4. BashTool Output Truncation

**File:** `mini_agent/tools/bash_tool.py`

### Current Issue
Long command output is returned verbatim — can overwhelm the context window.

### Changes
Apply token-based truncation to stdout before returning, reusing the existing `truncate_text_by_tokens` function:

```python
from .file_tools import truncate_text_by_tokens

BASH_MAX_OUTPUT_TOKENS = 16000

# In foreground execution, before creating BashOutputResult:
stdout_text = truncate_text_by_tokens(stdout_text, BASH_MAX_OUTPUT_TOKENS)
```

Only truncate stdout. stderr is typically short and should be preserved in full for debugging.

> **Note:** Truncation happens on the raw `stdout_text` string **before** constructing `BashOutputResult`. The Pydantic `model_validator` in `BashOutputResult` auto-generates `content` from `stdout`+`stderr`, so `content` will reflect the already-truncated stdout.

---

## 5. New GrepTool

**New file:** `mini_agent/tools/grep_tool.py`

### Design
Calls system search tools (ripgrep preferred, grep fallback) via subprocess. Not a pure Python implementation — performance and behavior consistency matter more.

### Interface

All tools in this codebase use `@property` for `name`, `description`, `parameters` — GrepTool follows the same pattern:

```python
class GrepTool(Tool):
    """Search file contents using regex patterns."""

    _search_tool_cache: str | None = None  # Class-level cache

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).absolute()

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents using regex patterns. "
            "Use this instead of bash for searching files. "
            "Returns matching lines with file paths and line numbers."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search (default: workspace root)",
                },
                "glob": {
                    "type": "string",
                    "description": "File filter pattern, e.g. '*.py', '*.json'",
                },
                "context": {
                    "type": "integer",
                    "description": "Lines of context around each match (default: 0)",
                    "default": 0,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of result lines to return (default: 50)",
                    "default": 50,
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_only", "count"],
                    "description": "Output format (default: content)",
                    "default": "content",
                },
            },
            "required": ["pattern"],
        }
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| pattern | str | Yes | - | Regex pattern to search for |
| path | str | No | workspace_dir | Directory or file to search |
| glob | str | No | None | File filter, e.g. `"*.py"` |
| context | int | No | 0 | Lines of context around matches |
| max_results | int | No | 50 | Maximum number of result lines to return |
| output_mode | str | No | "content" | `"content"`, `"files_only"`, or `"count"` |

### Path Resolution

Consistent with ReadTool/WriteTool/EditTool pattern:

```python
def _resolve_path(self, path: str | None) -> Path:
    if path is None:
        return self.workspace_dir
    p = Path(path)
    if not p.is_absolute():
        p = self.workspace_dir / p
    return p
```

### Execution Logic

```python
async def execute(self, pattern: str, path: str | None = None, glob: str | None = None,
                  context: int = 0, max_results: int = 50,
                  output_mode: str = "content") -> ToolResult:
    search_path = self._resolve_path(path)
    cmd = self._build_command(pattern, search_path, glob, context, output_mode)

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(self.workspace_dir),
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    except asyncio.TimeoutError:
        process.kill()
        return ToolResult(success=False, content="", error="Search timed out after 30s")

    output = stdout.decode("utf-8", errors="replace")

    # Total result limiting: take first max_results lines
    lines = output.splitlines()
    if len(lines) > max_results:
        output = "\n".join(lines[:max_results])
        output += f"\n\n[Truncated: showing {max_results} of {len(lines)} result lines]"

    # Token truncation as safety net
    output = truncate_text_by_tokens(output, 16000)

    if not output.strip():
        return ToolResult(success=True, content="No matches found.")

    return ToolResult(success=True, content=output)
```

### Search Tool Detection

```python
@classmethod
def _detect_search_tool(cls) -> str:
    if cls._search_tool_cache is not None:
        return cls._search_tool_cache

    import shutil
    if shutil.which("rg"):
        cls._search_tool_cache = "rg"
    elif shutil.which("grep"):
        cls._search_tool_cache = "grep"
    else:
        cls._search_tool_cache = "grep"  # fallback, will fail gracefully

    return cls._search_tool_cache
```

### Command Building

**ripgrep path:**
```
rg --no-heading -n [--glob GLOB] [-C CONTEXT] [-l | -c] PATTERN PATH
```

**grep fallback:**
```
grep -rn [--include=GLOB] [-C CONTEXT] [-l | -c] PATTERN PATH
```

> **Note:** `rg --max-count` limits per-file, not total. Total result limiting is handled in Python after output collection (see `max_results` logic in execute). Token truncation serves as an additional backstop.

### Registration
- `GrepTool(workspace_dir=str(workspace_dir))` in `cli.py:add_workspace_tools()`
- Add `enable_grep: bool = True` to `ToolsConfig` (or bundle with `enable_file_tools`)

---

## 6. New TodoTool

**New file:** `mini_agent/tools/todo_tool.py`

### Design Decisions
- **Single tool with operation parameter** — reduces tool count, easier for LLM to discover
- **In-memory only, no persistence** — todo is session-scoped; cross-session memory belongs to Phase 4
- **Instance-level storage** — each Agent instance has its own TodoStore; future SubAgents won't interfere

### TodoStore

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class TodoItem:
    id: int
    content: str
    status: str = "pending"  # pending | in_progress | completed
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

class TodoStore:
    def __init__(self):
        self._items: dict[int, TodoItem] = {}
        self._next_id: int = 1

    def add(self, content: str, status: str = "pending") -> TodoItem:
        item = TodoItem(id=self._next_id, content=content, status=status)
        self._items[self._next_id] = item
        self._next_id += 1
        return item

    def update(self, item_id: int, content: str | None = None, status: str | None = None) -> TodoItem:
        item = self._items[item_id]  # KeyError if not found
        if content is not None:
            item.content = content
        if status is not None:
            item.status = status
        return item

    def remove(self, item_id: int) -> None:
        del self._items[item_id]  # KeyError if not found

    def list_all(self) -> list[TodoItem]:
        return list(self._items.values())

    def summary(self) -> str:
        total = len(self._items)
        completed = sum(1 for item in self._items.values() if item.status == "completed")
        return f"Progress: {completed}/{total} completed"
```

### TodoTool Interface

Follows `@property` pattern consistent with all existing tools:

```python
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
                            "id": {"type": "integer", "description": "Item ID (required for update/remove)"},
                            "content": {"type": "string", "description": "Task description"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
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
```

> **Note on `id` parameter:** The JSON schema uses `"id"` which maps directly to the Python kwarg. This shadows Python's built-in `id()` but is acceptable here since `id()` is never called inside `execute`. The LLM generates `{"id": 3}` in its tool call JSON, and `**kwargs` unpacking requires the parameter name to match the schema key.

### Execution

Explicit `KeyError` handling for invalid IDs — returns a clean error instead of a traceback:

```python
async def execute(self, operation: str, items: list[dict] | None = None,
                  id: int | None = None) -> ToolResult:  # noqa: A002
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
                for t in todos:
                    status_icon = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
                    lines.append(f"  #{t.id} {status_icon.get(t.status, '[ ]')} {t.content}")
                lines.append(f"\n{self._store.summary()}")
                return ToolResult(success=True, content="Todo List:\n" + "\n".join(lines))

            case "remove":
                if id is None:
                    return ToolResult(success=False, content="", error="'id' required for remove")
                self._store.remove(id)
                return ToolResult(success=True, content=f"Removed todo #{id}")

            case _:
                return ToolResult(success=False, content="",
                                  error=f"Unknown operation: {operation}")

    except KeyError as e:
        return ToolResult(success=False, content="", error=f"Todo not found: {e}")
    except Exception as e:
        return ToolResult(success=False, content="", error=str(e))
```

### Registration
- `TodoTool()` in `cli.py:add_workspace_tools()` (no workspace dependency, but grouped with session tools)
- Add `enable_todo: bool = True` to `ToolsConfig`

---

## 7. Logging Enhancement

**Files:** `mini_agent/logger.py`, `mini_agent/config.py`, `mini_agent/config/config-example.yaml`, `mini_agent/agent.py`, `mini_agent/cli.py`

### Log Level Definition

```python
from enum import Enum

class LogLevel(str, Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    VERBOSE = "verbose"
```

### Level Coverage Matrix

| Content | minimal | standard | verbose |
|---------|---------|----------|---------|
| Tool name + success/fail | Y | Y | Y |
| finish_reason | Y | Y | Y |
| Full messages list | - | Y | Y |
| Full response content | - | Y | Y |
| Tool arguments + result content | - | Y | Y |
| Thinking content | - | Y | Y |
| Tool schema definitions | - | - | Y |
| Token usage breakdown | - | - | Y |
| Full system prompt text | - | - | Y |

### Configuration

**`config.py`** — new model with Pydantic enum validation:

```python
class LoggingConfig(BaseModel):
    file_level: LogLevel = LogLevel.STANDARD
    console_level: LogLevel = LogLevel.MINIMAL
    max_files: int = 50
```

> **Note:** Using `LogLevel` enum directly (not `str`) so Pydantic validates at config load time. Invalid values like `"debug"` will produce a clear validation error instead of a cryptic `ValueError` at Agent construction.

**`Config`** — add field:

```python
class Config(BaseModel):
    llm: LLMConfig
    agent: AgentConfig
    tools: ToolsConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
```

**`Config.from_yaml()`** — add parsing (required because `from_yaml` does manual field extraction):

```python
# Parse logging configuration
logging_data = data.get("logging", {})
logging_config = LoggingConfig(
    file_level=logging_data.get("file_level", "standard"),
    console_level=logging_data.get("console_level", "minimal"),
    max_files=logging_data.get("max_files", 50),
)

return cls(
    llm=llm_config,
    agent=agent_config,
    tools=tools_config,
    logging=logging_config,  # new
)
```

**`config-example.yaml`** — add section:

```yaml
# ===== Logging Configuration =====
logging:
  file_level: "standard"     # minimal / standard / verbose
  console_level: "minimal"   # minimal / standard / verbose
  max_files: 50              # Maximum log files to retain
```

### AgentLogger Changes

```python
class AgentLogger:
    def __init__(self, file_level: LogLevel = LogLevel.STANDARD,
                 console_level: LogLevel = LogLevel.MINIMAL,
                 max_files: int = 50):
        self.file_level = file_level
        self.console_level = console_level
        self.max_files = max_files
        self._console_logger = logging.getLogger("mini_agent.trace")
```

**Key methods updated:**

- `start_new_run()` — add `self._rotate_logs()` call
- `log_request()` — respect level: minimal skips, standard writes messages (current behavior), verbose adds tool schemas
- `log_response()` — respect level: minimal writes finish_reason only, standard writes full response, verbose adds usage breakdown
- `log_tool_result()` — respect level: minimal writes name+status, standard writes full result, verbose same as standard

**New `_rotate_logs()`:**

```python
def _rotate_logs(self):
    """Keep only the most recent max_files log files, delete older ones."""
    log_files = sorted(self.log_dir.glob("agent_run_*.log"), key=lambda f: f.stat().st_mtime)
    if len(log_files) > self.max_files:
        for f in log_files[:-self.max_files]:
            f.unlink(missing_ok=True)
```

**New `_write_console()`:**

```python
def _write_console(self, log_type: str, content: str):
    """Write log entry to console via Python logging module."""
    self._console_logger.info("[%s] %s", log_type, content)
```

The `_write_console` method is called alongside `_write_log` in each `log_*` method when the content passes the `console_level` threshold check.

### Agent / CLI Integration

**`agent.py`** — accept logger config:

```python
class Agent:
    def __init__(self, ..., log_file_level: str = "standard",
                 log_console_level: str = "minimal",
                 log_max_files: int = 50):
        self.logger = AgentLogger(
            file_level=LogLevel(log_file_level),
            console_level=LogLevel(log_console_level),
            max_files=log_max_files,
        )
```

**`cli.py`** — pass config through:

```python
agent = Agent(
    ...,
    log_file_level=config.logging.file_level,
    log_console_level=config.logging.console_level,
    log_max_files=config.logging.max_files,
)
```

---

## 8. System Prompt Updates

**File:** `mini_agent/config/system_prompt.md`

Add tool guidance:

```markdown
### 1. **Basic Tools**
- **File Operations**: Read, write, edit files with full path support
- **Search**: Use `grep` tool to search file contents (do NOT use bash for searching)
- **Bash Execution**: Run commands, manage git, packages, and system operations
- **MCP Tools**: Access additional tools from configured MCP servers

### 3. **Task Tracking**
- For complex tasks (3+ steps), use `todo` tool to create a task list first
- Mark tasks as `in_progress` before starting, `completed` when done
- Use `todo(operation="list")` to review progress and stay on track
```

---

## 9. File Change Summary

| Action | File | Changes |
|--------|------|---------|
| Modify | `mini_agent/tools/file_tools.py` | ReadTool: large file protection, long line truncation, metadata header. WriteTool: return stats. EditTool: add replace_all, fix multi-match, return line numbers. |
| Modify | `mini_agent/tools/bash_tool.py` | Output truncation via truncate_text_by_tokens |
| **Create** | `mini_agent/tools/grep_tool.py` | GrepTool: regex search via rg/grep subprocess |
| **Create** | `mini_agent/tools/todo_tool.py` | TodoTool + TodoStore: session-scoped task tracking |
| Modify | `mini_agent/logger.py` | LogLevel enum, configurable file/console levels, log rotation |
| Modify | `mini_agent/config.py` | Add LoggingConfig, update Config class and from_yaml parsing |
| Modify | `mini_agent/config/config-example.yaml` | Add logging configuration section |
| Modify | `mini_agent/config/system_prompt.md` | Add grep/todo tool guidance |
| Modify | `mini_agent/cli.py` | Register GrepTool/TodoTool, pass logging config to Agent |
| Modify | `mini_agent/agent.py` | Accept logging config parameters |
| **Create** | `tests/test_grep_tool.py` | GrepTool unit tests |
| **Create** | `tests/test_todo_tool.py` | TodoTool + TodoStore unit tests |
| Modify | `tests/test_agent.py` | Adapt to new Agent constructor parameters (if needed) |

---

## 10. Testing Strategy

### New Tests
- **test_grep_tool.py**: search with rg/grep, glob filtering, output modes, timeout handling, empty results, truncation
- **test_todo_tool.py**: add/update/list/remove operations, batch operations, error cases (not found, missing params), store isolation

### Existing Test Impact
- Agent constructor signature change (new optional params with defaults) — existing tests should still pass without modification
- EditTool tests may need updates for new `replace_all` parameter behavior

### Manual Verification
- Run agent interactively, test grep on real codebase
- Test todo workflow: create list → work through → complete
- Verify logging at each level (minimal, standard, verbose)
- Check log rotation works with max_files limit

---

## 11. Design Principles

1. **Backward compatible** — all new parameters have defaults, existing code works without changes
2. **Consistent with existing patterns** — all tools extend `Tool` base, return `ToolResult`, async execute
3. **Reuse existing utilities** — `truncate_text_by_tokens` for all truncation needs
4. **YAGNI** — no persistence for Todo, no custom regex engine for Grep, no structured logging format
5. **Independent deliverables** — each sub-task (ReadTool fix, GrepTool, TodoTool, Logging) can be implemented and tested independently
