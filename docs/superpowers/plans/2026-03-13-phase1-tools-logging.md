# Phase 1: Tools + Logging Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve existing tools (Read/Write/Edit/Bash), add GrepTool and TodoTool, and make logging configurable with three levels.

**Architecture:** Each tool change is independent — modify in place for existing tools, create new files for Grep/Todo. Config and logging changes form the foundation. CLI registration ties everything together at the end.

**Tech Stack:** Python 3.10+, asyncio, Pydantic, tiktoken, ripgrep (optional, grep fallback), pytest

**Spec:** `docs/superpowers/specs/2026-03-13-phase1-tools-logging-design.md`

---

## File Structure

### Files to Create
| File | Responsibility |
|------|---------------|
| `mini_agent/tools/grep_tool.py` | GrepTool: regex search via rg/grep subprocess |
| `mini_agent/tools/todo_tool.py` | TodoTool + TodoStore: session-scoped task tracking |
| `tests/test_grep_tool.py` | GrepTool unit tests |
| `tests/test_todo_tool.py` | TodoTool + TodoStore unit tests |
| `tests/test_logging.py` | AgentLogger level/rotation tests |

### Files to Modify
| File | Changes |
|------|---------|
| `mini_agent/config.py` | Add `LoggingConfig`, add `enable_grep`/`enable_todo` to `ToolsConfig`, update `from_yaml` |
| `mini_agent/tools/file_tools.py` | ReadTool (large file, line truncation, metadata), WriteTool (stats), EditTool (replace_all, line nums) |
| `mini_agent/tools/bash_tool.py` | stdout truncation before BashOutputResult construction |
| `mini_agent/logger.py` | LogLevel enum, configurable levels, rotation, console output |
| `mini_agent/agent.py` | Accept logging config params in constructor |
| `mini_agent/cli.py` | Register GrepTool/TodoTool, pass logging config |
| `mini_agent/config/config-example.yaml` | Add logging section |
| `mini_agent/config/system_prompt.md` | Add grep/todo guidance |

---

## Dependency Graph

```
Task 1 (Config)  ─────────────────────────────────> Task 8 (Logging) ──┐
                                                                        │
Task 2 (ReadTool) -> Task 3 (WriteTool) -> Task 4 (EditTool) ────────> │
                                                                        ├──> Task 9 (CLI + Prompt)
Task 5 (BashTool)  ─── independent ───────────────────────────────────> │
Task 6 (GrepTool)  ─── independent ───────────────────────────────────> │
Task 7 (TodoTool)  ─── independent ───────────────────────────────────> │
```

**Tasks 2→3→4 must be sequential** (they share `file_tools.py` and `test_tools.py`). Tasks 5, 6, 7, 8 can run in parallel with each other and with the 2→3→4 chain. Task 9 depends on all.

---

## Chunk 1: Foundation (Config)

### Task 1: Config Changes

**Files:**
- Modify: `mini_agent/config.py`
- Modify: `mini_agent/config/config-example.yaml`

- [ ] **Step 1: Add LogLevel enum and LoggingConfig to config.py**

Add at the top of the file, after existing imports:

```python
from enum import Enum

class LogLevel(str, Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    VERBOSE = "verbose"
```

Add after `MCPConfig`:

```python
class LoggingConfig(BaseModel):
    """Logging configuration"""
    file_level: LogLevel = LogLevel.STANDARD
    console_level: LogLevel = LogLevel.MINIMAL
    max_files: int = 50
```

- [ ] **Step 2: Update ToolsConfig with new tool flags**

Add to `ToolsConfig` class:

```python
enable_grep: bool = True
enable_todo: bool = True
```

- [ ] **Step 3: Add logging field to Config class**

```python
class Config(BaseModel):
    """Main configuration class"""
    llm: LLMConfig
    agent: AgentConfig
    tools: ToolsConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
```

- [ ] **Step 4: Update from_yaml to parse logging config**

In `from_yaml()`, before the `return cls(...)` statement, add:

```python
# Parse logging configuration
logging_data = data.get("logging", {})
logging_config = LoggingConfig(
    file_level=logging_data.get("file_level", "standard"),
    console_level=logging_data.get("console_level", "minimal"),
    max_files=logging_data.get("max_files", 50),
)
```

Update `from_yaml()` tools parsing to include new fields:

```python
tools_config = ToolsConfig(
    enable_file_tools=tools_data.get("enable_file_tools", True),
    enable_bash=tools_data.get("enable_bash", True),
    enable_note=tools_data.get("enable_note", True),
    enable_grep=tools_data.get("enable_grep", True),      # new
    enable_todo=tools_data.get("enable_todo", True),       # new
    enable_skills=tools_data.get("enable_skills", True),
    skills_dir=tools_data.get("skills_dir", "./skills"),
    enable_mcp=tools_data.get("enable_mcp", True),
    mcp_config_path=tools_data.get("mcp_config_path", "mcp.json"),
    mcp=mcp_config,
)
```

Update the return statement:

```python
return cls(
    llm=llm_config,
    agent=agent_config,
    tools=tools_config,
    logging=logging_config,
)
```

- [ ] **Step 5: Update config-example.yaml**

Add `enable_grep` and `enable_todo` to the tools section, after `enable_note`:

```yaml
  enable_grep: true        # Grep search tool (GrepTool)
  enable_todo: true        # Todo task tracking tool (TodoTool)
```

Add at the end of the file:

```yaml

# ===== Logging Configuration =====
logging:
  file_level: "standard"     # minimal / standard / verbose
  console_level: "minimal"   # minimal / standard / verbose
  max_files: 50              # Maximum log files to retain
```

- [ ] **Step 6: Verify config loads correctly**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && python -c "from mini_agent.config import Config, LogLevel, LoggingConfig; print('OK:', LogLevel.STANDARD, LoggingConfig())"`

Expected: `OK: standard file_level=<LogLevel.STANDARD: 'standard'> console_level=<LogLevel.MINIMAL: 'minimal'> max_files=50`

- [ ] **Step 7: Run existing tests to verify no regressions**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/ -v --ignore=tests/test_agent.py --ignore=tests/test_integration.py --ignore=tests/test_session_integration.py --ignore=tests/test_acp.py -x`

Expected: All tests PASS (these tests don't use real LLM API)

- [ ] **Step 8: Commit**

```bash
git add mini_agent/config.py mini_agent/config/config-example.yaml
git commit -m "feat(config): add LoggingConfig, LogLevel enum, enable_grep/enable_todo flags"
```

---

## Chunk 2: File Tools (Read/Write/Edit)

### Task 2: ReadTool Improvements

**Files:**
- Modify: `mini_agent/tools/file_tools.py` (class `ReadTool`, lines 63-152)
- Test: `tests/test_tools.py` (add new test cases)

- [ ] **Step 1: Write failing tests for ReadTool improvements**

Append to `tests/test_tools.py`:

```python
@pytest.mark.asyncio
async def test_read_tool_large_file_protection():
    """Test that large files get preview + hint instead of full content."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        # Write 2500 lines (exceeds MAX_LINES=2000)
        for i in range(2500):
            f.write(f"Line {i+1}: some content here\n")
        temp_path = f.name

    try:
        tool = ReadTool()
        result = await tool.execute(path=temp_path)

        assert result.success
        # Should show preview, not all 2500 lines
        assert "2500 lines" in result.content
        assert "Use offset/limit" in result.content
        # Should contain first ~100 lines but NOT line 2500
        assert "|Line 1:" in result.content
        assert "|Line 2500:" not in result.content
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_read_tool_large_file_with_offset():
    """Test that large files can be read with offset/limit."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        for i in range(2500):
            f.write(f"Line {i+1}: content\n")
        temp_path = f.name

    try:
        tool = ReadTool()
        result = await tool.execute(path=temp_path, offset=2490, limit=10)

        assert result.success
        assert "|Line 2490:" in result.content
        assert "|Line 2499:" in result.content
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_read_tool_long_line_truncation():
    """Test that long lines get truncated."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("short line\n")
        f.write("x" * 5000 + "\n")  # Very long line
        f.write("another short line\n")
        temp_path = f.name

    try:
        tool = ReadTool()
        result = await tool.execute(path=temp_path)

        assert result.success
        assert "truncated" in result.content
        assert "5000 chars" in result.content
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_read_tool_metadata_header():
    """Test that output includes file metadata header."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Hello\nWorld\n")
        temp_path = f.name

    try:
        tool = ReadTool()
        result = await tool.execute(path=temp_path)

        assert result.success
        assert "[File:" in result.content
        assert "2 lines" in result.content
    finally:
        Path(temp_path).unlink()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/test_tools.py::test_read_tool_large_file_protection tests/test_tools.py::test_read_tool_long_line_truncation tests/test_tools.py::test_read_tool_metadata_header -v`

Expected: FAIL (features not implemented yet)

- [ ] **Step 3: Implement ReadTool improvements**

Replace the `execute` method of `ReadTool` in `mini_agent/tools/file_tools.py`:

```python
MAX_LINES = 2000
MAX_LINE_LENGTH = 2000
PREVIEW_LINES = 100

async def execute(self, path: str, offset: int | None = None, limit: int | None = None) -> ToolResult:
    """Execute read file."""
    try:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.workspace_dir / file_path

        if not file_path.exists():
            return ToolResult(success=False, content="", error=f"File not found: {path}")

        file_size = file_path.stat().st_size

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)

        # File metadata header
        size_str = f"{file_size / 1024:.1f} KB" if file_size >= 1024 else f"{file_size} bytes"
        header = f"[File: {path}, {total_lines} lines, {size_str}]"

        # Large file protection: preview only when no offset/limit specified
        if total_lines > MAX_LINES and offset is None and limit is None:
            selected_lines = lines[:PREVIEW_LINES]
            start = 0
            end = PREVIEW_LINES
            header += f"\n[Showing first {PREVIEW_LINES} of {total_lines} lines. Use offset/limit to read specific ranges.]"
        else:
            # Apply offset and limit
            start = (offset - 1) if offset else 0
            end = (start + limit) if limit else len(lines)
            if start < 0:
                start = 0
            if end > len(lines):
                end = len(lines)
            selected_lines = lines[start:end]

            if offset is not None or limit is not None:
                header += f"\n[Showing lines {start + 1}-{end} of {total_lines}.]"

        # Format with line numbers, truncate long lines
        numbered_lines = []
        for i, line in enumerate(selected_lines, start=start + 1):
            line_content = line.rstrip("\n")
            if len(line_content) > MAX_LINE_LENGTH:
                original_len = len(line_content)
                line_content = line_content[:MAX_LINE_LENGTH] + f"... [truncated, {original_len} chars total]"
            numbered_lines.append(f"{i:6d}|{line_content}")

        content = header + "\n" + "\n".join(numbered_lines)

        # Token truncation as safety net
        max_tokens = 32000
        content = truncate_text_by_tokens(content, max_tokens)

        return ToolResult(success=True, content=content)
    except Exception as e:
        return ToolResult(success=False, content="", error=str(e))
```

Also add the constants at the module level (after the `truncate_text_by_tokens` function):

```python
# ReadTool constants
MAX_LINES = 2000
MAX_LINE_LENGTH = 2000
PREVIEW_LINES = 100
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/test_tools.py -v`

Expected: All tests PASS (including existing `test_read_tool`)

- [ ] **Step 5: Commit**

```bash
git add mini_agent/tools/file_tools.py tests/test_tools.py
git commit -m "feat(read): add large file protection, long line truncation, metadata header"
```

---

### Task 3: WriteTool Improvements

**Files:**
- Modify: `mini_agent/tools/file_tools.py` (class `WriteTool`, lines 155-209)
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_tools.py`:

```python
@pytest.mark.asyncio
async def test_write_tool_returns_stats():
    """Test that WriteTool returns line count and byte count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.txt"

        tool = WriteTool()
        result = await tool.execute(path=str(file_path), content="line1\nline2\nline3")

        assert result.success
        assert "3 lines" in result.content
        assert "bytes" in result.content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/test_tools.py::test_write_tool_returns_stats -v`

Expected: FAIL (current message is just "Successfully wrote to ...")

- [ ] **Step 3: Implement WriteTool improvement**

In `WriteTool.execute`, replace the success return:

```python
file_path.write_text(content, encoding="utf-8")

line_count = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
byte_count = len(content.encode('utf-8'))
return ToolResult(success=True, content=f"Successfully wrote to {file_path} ({line_count} lines, {byte_count} bytes)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/test_tools.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mini_agent/tools/file_tools.py tests/test_tools.py
git commit -m "feat(write): return line count and byte count in success message"
```

---

### Task 4: EditTool Improvements

**Files:**
- Modify: `mini_agent/tools/file_tools.py` (class `EditTool`, lines 212-286)
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_tools.py`:

```python
@pytest.mark.asyncio
async def test_edit_tool_replace_all():
    """Test EditTool with replace_all=True."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("foo bar foo baz foo")
        temp_path = f.name

    try:
        tool = EditTool()
        result = await tool.execute(path=temp_path, old_str="foo", new_str="qux", replace_all=True)

        assert result.success
        content = Path(temp_path).read_text()
        assert content == "qux bar qux baz qux"
        assert "3 occurrence" in result.content
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_edit_tool_multi_match_error():
    """Test EditTool rejects multiple matches without replace_all."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("foo bar foo baz foo")
        temp_path = f.name

    try:
        tool = EditTool()
        result = await tool.execute(path=temp_path, old_str="foo", new_str="qux")

        assert not result.success
        assert "3 matches" in result.error
        # File should be unchanged
        content = Path(temp_path).read_text()
        assert content == "foo bar foo baz foo"
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_edit_tool_returns_line_number():
    """Test EditTool returns line number in success message."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("line1\nline2\ntarget_text\nline4\n")
        temp_path = f.name

    try:
        tool = EditTool()
        result = await tool.execute(path=temp_path, old_str="target_text", new_str="replaced")

        assert result.success
        assert "line 3" in result.content
    finally:
        Path(temp_path).unlink()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/test_tools.py::test_edit_tool_replace_all tests/test_tools.py::test_edit_tool_multi_match_error tests/test_tools.py::test_edit_tool_returns_line_number -v`

Expected: FAIL

- [ ] **Step 3: Implement EditTool improvements**

Replace the entire `execute` method and add `replace_all` to `parameters`:

In the `parameters` property, add to `"properties"`:

```python
"replace_all": {
    "type": "boolean",
    "description": "Replace all occurrences (default: false, requires unique match)",
    "default": False,
},
```

Replace the `execute` method:

```python
async def execute(self, path: str, old_str: str, new_str: str, replace_all: bool = False) -> ToolResult:
    """Execute edit file."""
    try:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.workspace_dir / file_path

        if not file_path.exists():
            return ToolResult(success=False, content="", error=f"File not found: {path}")

        content = file_path.read_text(encoding="utf-8")

        # Count matches
        count = content.count(old_str)

        if count == 0:
            return ToolResult(success=False, content="", error=f"Text not found in file: {old_str}")

        if count > 1 and not replace_all:
            return ToolResult(
                success=False, content="",
                error=f"Found {count} matches. Provide more context for a unique match, or set replace_all=true.",
            )

        # Calculate line number of first match (before replacement)
        match_offset = content.index(old_str)
        match_line = content[:match_offset].count('\n') + 1

        # Perform replacement
        if replace_all:
            new_content = content.replace(old_str, new_str)
        else:
            new_content = content.replace(old_str, new_str, 1)

        file_path.write_text(new_content, encoding="utf-8")

        # Return with line info
        if replace_all:
            msg = f"Edited {file_path}: replaced {count} occurrence(s) (first at line {match_line})"
        else:
            msg = f"Edited {file_path}: replaced at line {match_line}"

        return ToolResult(success=True, content=msg)
    except Exception as e:
        return ToolResult(success=False, content="", error=str(e))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/test_tools.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mini_agent/tools/file_tools.py tests/test_tools.py
git commit -m "feat(edit): add replace_all param, fix multi-match behavior, return line numbers"
```

---

## Chunk 3: BashTool + GrepTool + TodoTool

### Task 5: BashTool Output Truncation

**Files:**
- Modify: `mini_agent/tools/bash_tool.py`
- Test: `tests/test_bash_tool.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_bash_tool.py`:

```python
@pytest.mark.asyncio
async def test_foreground_output_truncation():
    """Test that very long stdout gets truncated."""
    bash_tool = BashTool()
    # Generate ~80000 characters of output (well above 16000 tokens)
    result = await bash_tool.execute(
        command="python3 -c \"print('a' * 80000)\"",
        timeout=10,
    )

    assert result.success
    # 80000 chars >> 16000 tokens, so truncation MUST trigger
    assert "truncated" in result.content.lower()
```

- [ ] **Step 2: Run test to verify behavior before change**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/test_bash_tool.py::test_foreground_output_truncation -v`

- [ ] **Step 3: Implement BashTool truncation**

In `mini_agent/tools/bash_tool.py`, add import at top:

```python
from .file_tools import truncate_text_by_tokens
```

Add constant:

```python
BASH_MAX_OUTPUT_TOKENS = 16000
```

In the `BashTool.execute` method, in the foreground execution path, after decoding stdout and before creating `BashOutputResult`, add:

```python
# Truncate long stdout to prevent context overflow
stdout_text = truncate_text_by_tokens(stdout_text, BASH_MAX_OUTPUT_TOKENS)
```

- [ ] **Step 4: Run all bash tests to verify no regressions**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/test_bash_tool.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mini_agent/tools/bash_tool.py tests/test_bash_tool.py
git commit -m "feat(bash): add stdout truncation to prevent context overflow"
```

---

### Task 6: GrepTool

**Files:**
- Create: `mini_agent/tools/grep_tool.py`
- Create: `tests/test_grep_tool.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_grep_tool.py`:

```python
"""Test cases for GrepTool."""

import tempfile
from pathlib import Path

import pytest

from mini_agent.tools.grep_tool import GrepTool


@pytest.fixture
def search_dir():
    """Create a temp directory with searchable files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "hello.py").write_text("def hello():\n    print('hello world')\n    return True\n")
        (Path(tmpdir) / "goodbye.py").write_text("def goodbye():\n    print('goodbye world')\n    return False\n")
        (Path(tmpdir) / "data.json").write_text('{"key": "value", "hello": "world"}\n')
        (Path(tmpdir) / "empty.txt").write_text("")
        yield tmpdir


@pytest.mark.asyncio
async def test_grep_basic_search(search_dir):
    """Test basic regex search."""
    tool = GrepTool(workspace_dir=search_dir)
    result = await tool.execute(pattern="hello")

    assert result.success
    assert "hello" in result.content


@pytest.mark.asyncio
async def test_grep_no_matches(search_dir):
    """Test search with no matches."""
    tool = GrepTool(workspace_dir=search_dir)
    result = await tool.execute(pattern="nonexistent_string_xyz")

    assert result.success
    assert "No matches found" in result.content


@pytest.mark.asyncio
async def test_grep_glob_filter(search_dir):
    """Test file filtering with glob pattern."""
    tool = GrepTool(workspace_dir=search_dir)
    result = await tool.execute(pattern="hello", glob="*.py")

    assert result.success
    assert "hello" in result.content
    # Should not include json file matches
    assert "data.json" not in result.content


@pytest.mark.asyncio
async def test_grep_files_only_mode(search_dir):
    """Test files_only output mode."""
    tool = GrepTool(workspace_dir=search_dir)
    result = await tool.execute(pattern="hello", output_mode="files_only")

    assert result.success
    assert "hello.py" in result.content


@pytest.mark.asyncio
async def test_grep_count_mode(search_dir):
    """Test count output mode."""
    tool = GrepTool(workspace_dir=search_dir)
    result = await tool.execute(pattern="world", output_mode="count")

    assert result.success
    # Should have counts for files containing "world"


@pytest.mark.asyncio
async def test_grep_with_context(search_dir):
    """Test search with context lines."""
    tool = GrepTool(workspace_dir=search_dir)
    result = await tool.execute(pattern="print", context=1)

    assert result.success
    # Context should include surrounding lines (def or return)
    assert "def" in result.content or "return" in result.content


@pytest.mark.asyncio
async def test_grep_max_results(search_dir):
    """Test max_results limiting."""
    tool = GrepTool(workspace_dir=search_dir)
    result = await tool.execute(pattern=".", max_results=3)

    assert result.success
    # Should have at most 3 result lines (may have truncation notice)


@pytest.mark.asyncio
async def test_grep_specific_path(search_dir):
    """Test searching in a specific file."""
    tool = GrepTool(workspace_dir=search_dir)
    result = await tool.execute(pattern="hello", path="hello.py")

    assert result.success
    assert "hello" in result.content


@pytest.mark.asyncio
async def test_grep_schema():
    """Test GrepTool schema generation."""
    tool = GrepTool()
    schema = tool.to_schema()

    assert schema["name"] == "grep"
    assert "input_schema" in schema
    assert "pattern" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["required"] == ["pattern"]

    openai_schema = tool.to_openai_schema()
    assert openai_schema["type"] == "function"
    assert openai_schema["function"]["name"] == "grep"
```

- [ ] **Step 2: Run tests to verify they fail (module not found)**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/test_grep_tool.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'mini_agent.tools.grep_tool'`

- [ ] **Step 3: Implement GrepTool**

Create `mini_agent/tools/grep_tool.py`:

```python
"""Search tool using ripgrep or grep."""

import asyncio
import shutil
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult
from .file_tools import truncate_text_by_tokens


GREP_MAX_OUTPUT_TOKENS = 16000
GREP_TIMEOUT = 30


class GrepTool(Tool):
    """Search file contents using regex patterns via rg/grep subprocess."""

    _search_tool_cache: str | None = None

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

    def _resolve_path(self, path: str | None) -> Path:
        """Resolve search path relative to workspace."""
        if path is None:
            return self.workspace_dir
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace_dir / p
        return p

    @classmethod
    def _detect_search_tool(cls) -> str:
        """Detect available search tool (rg preferred, grep fallback)."""
        if cls._search_tool_cache is not None:
            return cls._search_tool_cache

        if shutil.which("rg"):
            cls._search_tool_cache = "rg"
        else:
            cls._search_tool_cache = "grep"

        return cls._search_tool_cache

    def _build_command(
        self,
        pattern: str,
        search_path: Path,
        glob: str | None,
        context: int,
        output_mode: str,
    ) -> list[str]:
        """Build search command based on available tool."""
        tool = self._detect_search_tool()

        if tool == "rg":
            cmd = ["rg", "--no-heading", "-n"]
            if output_mode == "files_only":
                cmd.append("-l")
            elif output_mode == "count":
                cmd.append("-c")
            if glob:
                cmd.extend(["--glob", glob])
            if context > 0:
                cmd.extend(["-C", str(context)])
            cmd.append(pattern)
            cmd.append(str(search_path))
        else:
            cmd = ["grep", "-rn"]
            if output_mode == "files_only":
                cmd.append("-l")
            elif output_mode == "count":
                cmd.append("-c")
            if glob:
                cmd.extend(["--include", glob])
            if context > 0:
                cmd.extend(["-C", str(context)])
            cmd.append(pattern)
            cmd.append(str(search_path))

        return cmd

    async def execute(  # pylint: disable=arguments-differ
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        context: int = 0,
        max_results: int = 50,
        output_mode: str = "content",
    ) -> ToolResult:
        """Execute search."""
        try:
            search_path = self._resolve_path(path)
            cmd = self._build_command(pattern, search_path, glob, context, output_mode)

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_dir),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=GREP_TIMEOUT
                )
            except asyncio.TimeoutError:
                process.kill()
                return ToolResult(
                    success=False, content="", error=f"Search timed out after {GREP_TIMEOUT}s"
                )

            output = stdout.decode("utf-8", errors="replace")

            # Total result limiting
            lines = output.splitlines()
            if len(lines) > max_results:
                output = "\n".join(lines[:max_results])
                output += f"\n\n[Truncated: showing {max_results} of {len(lines)} result lines]"

            # Token truncation as safety net
            output = truncate_text_by_tokens(output, GREP_MAX_OUTPUT_TOKENS)

            if not output.strip():
                return ToolResult(success=True, content="No matches found.")

            return ToolResult(success=True, content=output)

        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/test_grep_tool.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mini_agent/tools/grep_tool.py tests/test_grep_tool.py
git commit -m "feat: add GrepTool with ripgrep/grep backend, glob filtering, output modes"
```

---

### Task 7: TodoTool

**Files:**
- Create: `mini_agent/tools/todo_tool.py`
- Create: `tests/test_todo_tool.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_todo_tool.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/test_todo_tool.py -v`

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement TodoTool**

Create `mini_agent/tools/todo_tool.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/test_todo_tool.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mini_agent/tools/todo_tool.py tests/test_todo_tool.py
git commit -m "feat: add TodoTool with session-scoped task tracking"
```

---

## Chunk 4: Logging Enhancement

### Task 8: Logging Enhancement

**Files:**
- Modify: `mini_agent/logger.py`
- Modify: `mini_agent/agent.py` (constructor)
- Create: `tests/test_logging.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_logging.py`:

```python
"""Test cases for AgentLogger enhancements."""

import tempfile
from pathlib import Path

import pytest

from mini_agent.config import LogLevel
from mini_agent.logger import AgentLogger


@pytest.fixture
def logger_dir():
    """Create a temp directory for log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestLogLevel:
    def test_log_level_values(self):
        assert LogLevel.MINIMAL == "minimal"
        assert LogLevel.STANDARD == "standard"
        assert LogLevel.VERBOSE == "verbose"

    def test_log_level_from_string(self):
        assert LogLevel("minimal") == LogLevel.MINIMAL
        assert LogLevel("standard") == LogLevel.STANDARD
        assert LogLevel("verbose") == LogLevel.VERBOSE


class TestAgentLogger:
    def test_default_levels(self):
        logger = AgentLogger()
        assert logger.file_level == LogLevel.STANDARD
        assert logger.console_level == LogLevel.MINIMAL

    def test_custom_levels(self):
        logger = AgentLogger(file_level=LogLevel.VERBOSE, console_level=LogLevel.STANDARD)
        assert logger.file_level == LogLevel.VERBOSE
        assert logger.console_level == LogLevel.STANDARD

    def test_start_new_run_creates_log_file(self):
        logger = AgentLogger()
        logger.start_new_run()
        assert logger.log_file is not None
        assert logger.log_file.exists()

    def test_minimal_level_skips_request_details(self):
        """Minimal level should only log tool names and status, not full messages."""
        logger = AgentLogger(file_level=LogLevel.MINIMAL)
        logger.start_new_run()

        from mini_agent.schema import Message
        messages = [Message(role="user", content="test message")]
        logger.log_request(messages=messages)

        log_content = logger.log_file.read_text()
        # Minimal should NOT contain full message content
        assert "test message" not in log_content

    def test_standard_level_logs_full_content(self):
        """Standard level should log full messages (current behavior)."""
        logger = AgentLogger(file_level=LogLevel.STANDARD)
        logger.start_new_run()

        from mini_agent.schema import Message
        messages = [Message(role="user", content="test message")]
        logger.log_request(messages=messages)

        log_content = logger.log_file.read_text()
        assert "test message" in log_content

    def test_verbose_level_logs_tool_schemas(self):
        """Verbose level should log tool schema definitions."""
        logger = AgentLogger(file_level=LogLevel.VERBOSE)
        logger.start_new_run()

        from mini_agent.schema import Message
        from mini_agent.tools.base import Tool, ToolResult

        class MockTool(Tool):
            @property
            def name(self):
                return "mock"
            @property
            def description(self):
                return "mock tool"
            @property
            def parameters(self):
                return {"type": "object", "properties": {}}
            async def execute(self, **kwargs):
                return ToolResult(success=True, content="ok")

        messages = [Message(role="user", content="test")]
        logger.log_request(messages=messages, tools=[MockTool()])

        log_content = logger.log_file.read_text()
        # Verbose should include tool schema details
        assert "mock" in log_content


class TestLogRotation:
    def test_rotate_logs(self, logger_dir):
        """Test that old log files are deleted when exceeding max_files."""
        logger = AgentLogger(max_files=3)
        # Override log_dir to use temp directory
        logger.log_dir = Path(logger_dir)

        # Create 5 log files
        for i in range(5):
            (Path(logger_dir) / f"agent_run_2026010{i}_120000.log").write_text(f"log {i}")

        logger._rotate_logs()

        remaining = list(Path(logger_dir).glob("agent_run_*.log"))
        assert len(remaining) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/test_logging.py -v`

Expected: FAIL (AgentLogger doesn't accept file_level param yet)

- [ ] **Step 3: Implement logging enhancement**

Replace `mini_agent/logger.py` entirely:

```python
"""Agent run logger with configurable levels."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import LogLevel
from .schema import Message, ToolCall


class AgentLogger:
    """Agent run logger with configurable file and console log levels.

    Log levels:
    - MINIMAL: tool names + success/fail + finish_reason only
    - STANDARD: full messages, responses, tool results (default)
    - VERBOSE: additionally includes tool schemas, token usage, system prompt
    """

    def __init__(
        self,
        file_level: LogLevel = LogLevel.STANDARD,
        console_level: LogLevel = LogLevel.MINIMAL,
        max_files: int = 50,
    ):
        self.file_level = file_level
        self.console_level = console_level

        # Numeric ordering for level comparison (string comparison is alphabetical, not severity)
        self._LEVEL_ORDER = {LogLevel.MINIMAL: 0, LogLevel.STANDARD: 1, LogLevel.VERBOSE: 2}
        self.max_files = max_files

        self.log_dir = Path.home() / ".mini-agent" / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = None
        self.log_index = 0

        self._console_logger = logging.getLogger("mini_agent.trace")

    def start_new_run(self):
        """Start new run, create new log file."""
        self._rotate_logs()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"agent_run_{timestamp}.log"
        self.log_file = self.log_dir / log_filename
        self.log_index = 0

        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Agent Run Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Log Level: file={self.file_level.value}, console={self.console_level.value}\n")
            f.write("=" * 80 + "\n\n")

    def log_request(self, messages: list[Message], tools: list[Any] | None = None):
        """Log LLM request."""
        self.log_index += 1

        # MINIMAL: skip request details entirely
        if self.file_level == LogLevel.MINIMAL:
            return

        # STANDARD: log messages + tool names (current behavior)
        request_data: dict[str, Any] = {"messages": []}

        for msg in messages:
            msg_dict: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.thinking:
                msg_dict["thinking"] = msg.thinking
            if msg.tool_calls:
                msg_dict["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if msg.name:
                msg_dict["name"] = msg.name
            request_data["messages"].append(msg_dict)

        # Tool schemas: names only for STANDARD, full schemas for VERBOSE
        if tools:
            if self.file_level == LogLevel.VERBOSE:
                request_data["tools"] = [tool.to_schema() for tool in tools]
            else:
                request_data["tools"] = [tool.name for tool in tools]

        content = "LLM Request:\n\n"
        content += json.dumps(request_data, indent=2, ensure_ascii=False)

        self._write_log("REQUEST", content)

        # Console output
        if self._LEVEL_ORDER[self.console_level] >= self._LEVEL_ORDER[LogLevel.STANDARD]:
            self._write_console("REQUEST", f"Sending {len(messages)} messages")

    def log_response(
        self,
        content: str,
        thinking: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        finish_reason: str | None = None,
        usage: dict | None = None,
    ):
        """Log LLM response."""
        self.log_index += 1

        if self.file_level == LogLevel.MINIMAL:
            # Minimal: only log finish_reason and tool call names
            minimal_data: dict[str, Any] = {}
            if finish_reason:
                minimal_data["finish_reason"] = finish_reason
            if tool_calls:
                minimal_data["tool_calls"] = [tc.function.name for tc in tool_calls]
            self._write_log("RESPONSE", json.dumps(minimal_data, indent=2, ensure_ascii=False))
            return

        # STANDARD+: full response
        response_data: dict[str, Any] = {"content": content}
        if thinking:
            response_data["thinking"] = thinking
        if tool_calls:
            response_data["tool_calls"] = [tc.model_dump() for tc in tool_calls]
        if finish_reason:
            response_data["finish_reason"] = finish_reason

        # VERBOSE: add usage breakdown
        if self.file_level == LogLevel.VERBOSE and usage:
            response_data["usage"] = usage

        log_content = "LLM Response:\n\n"
        log_content += json.dumps(response_data, indent=2, ensure_ascii=False)

        self._write_log("RESPONSE", log_content)

    def log_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result_success: bool,
        result_content: str | None = None,
        result_error: str | None = None,
    ):
        """Log tool execution result."""
        self.log_index += 1

        if self.file_level == LogLevel.MINIMAL:
            # Minimal: tool name + success/fail only
            status = "SUCCESS" if result_success else "FAIL"
            self._write_log("TOOL_RESULT", f"{tool_name}: {status}")
            return

        # STANDARD+: full tool result
        tool_result_data: dict[str, Any] = {
            "tool_name": tool_name,
            "arguments": arguments,
            "success": result_success,
        }
        if result_success:
            tool_result_data["result"] = result_content
        else:
            tool_result_data["error"] = result_error

        content = "Tool Execution:\n\n"
        content += json.dumps(tool_result_data, indent=2, ensure_ascii=False)

        self._write_log("TOOL_RESULT", content)

        # Console output for tool results
        if self._LEVEL_ORDER[self.console_level] >= self._LEVEL_ORDER[LogLevel.STANDARD]:
            status = "OK" if result_success else "FAIL"
            self._write_console("TOOL", f"{tool_name}: {status}")

    def _rotate_logs(self):
        """Keep only the most recent max_files log files."""
        log_files = sorted(
            self.log_dir.glob("agent_run_*.log"),
            key=lambda f: f.stat().st_mtime,
        )
        if len(log_files) > self.max_files:
            for f in log_files[: -self.max_files]:
                f.unlink(missing_ok=True)

    def _write_log(self, log_type: str, content: str):
        """Write log entry to file."""
        if self.log_file is None:
            return

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"[{self.log_index}] {log_type}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write("-" * 80 + "\n")
            f.write(content + "\n")

    def _write_console(self, log_type: str, content: str):
        """Write log entry to console via Python logging."""
        self._console_logger.info("[%s] %s", log_type, content)

    def get_log_file_path(self) -> Path:
        """Get current log file path."""
        return self.log_file
```

- [ ] **Step 4: Update Agent constructor**

In `mini_agent/agent.py`, update the `__init__` signature and body:

Add parameters:

```python
def __init__(
    self,
    llm_client: LLMClient,
    system_prompt: str,
    tools: list[Tool],
    max_steps: int = 50,
    workspace_dir: str = "./workspace",
    token_limit: int = 80000,
    log_file_level: str = "standard",
    log_console_level: str = "minimal",
    log_max_files: int = 50,
):
```

Replace the logger initialization line:

```python
# Initialize logger
from .config import LogLevel
self.logger = AgentLogger(
    file_level=LogLevel(log_file_level),
    console_level=LogLevel(log_console_level),
    max_files=log_max_files,
)
```

Also update the `log_response` call in `run_stream()` to pass usage data. Find the line:

```python
self.logger.log_response(content=text, thinking=thinking or None,
                          tool_calls=tool_calls or None, finish_reason=finish_reason)
```

Replace with:

```python
usage_data = None
if hasattr(self, 'api_total_tokens') and self.api_total_tokens:
    usage_data = {"total_tokens": self.api_total_tokens}
self.logger.log_response(content=text, thinking=thinking or None,
                          tool_calls=tool_calls or None, finish_reason=finish_reason,
                          usage=usage_data)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/test_logging.py -v`

Expected: All PASS

- [ ] **Step 6: Run all existing tests for regressions**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/ -v --ignore=tests/test_agent.py --ignore=tests/test_integration.py --ignore=tests/test_session_integration.py --ignore=tests/test_acp.py -x`

Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add mini_agent/logger.py mini_agent/agent.py tests/test_logging.py
git commit -m "feat(logging): configurable log levels (minimal/standard/verbose), log rotation, console output"
```

---

## Chunk 5: CLI Integration + System Prompt

### Task 9: CLI Registration + System Prompt

**Files:**
- Modify: `mini_agent/cli.py`
- Modify: `mini_agent/config/system_prompt.md`
- Modify: `mini_agent/tools/__init__.py` (if exists, to export new tools)

- [ ] **Step 1: Update tools/__init__.py exports**

In `mini_agent/tools/__init__.py`, add imports and exports for the new tools:

```python
from .grep_tool import GrepTool
from .todo_tool import TodoTool
```

Add to `__all__`:

```python
    "GrepTool",
    "TodoTool",
```

- [ ] **Step 2: Register GrepTool in cli.py**

In `mini_agent/cli.py`, add import:

```python
from mini_agent.tools.grep_tool import GrepTool
```

In `add_workspace_tools()`, add after file tools registration:

```python
# Grep tool - needs workspace to resolve relative search paths
if config.tools.enable_grep:
    tools.append(GrepTool(workspace_dir=str(workspace_dir)))
    print(f"{Colors.GREEN}✅ Loaded Grep tool{Colors.RESET}")
```

- [ ] **Step 3: Register TodoTool in cli.py**

Add import:

```python
from mini_agent.tools.todo_tool import TodoTool
```

In `add_workspace_tools()`, add after note tool:

```python
# Todo tool - session-scoped task tracking
if config.tools.enable_todo:
    tools.append(TodoTool())
    print(f"{Colors.GREEN}✅ Loaded Todo tool{Colors.RESET}")
```

- [ ] **Step 4: Pass logging config to Agent in cli.py**

In `run_agent()`, update the Agent construction:

```python
agent = Agent(
    llm_client=llm_client,
    system_prompt=system_prompt,
    tools=tools,
    max_steps=config.agent.max_steps,
    workspace_dir=str(workspace_dir),
    log_file_level=config.logging.file_level.value,
    log_console_level=config.logging.console_level.value,
    log_max_files=config.logging.max_files,
)
```

- [ ] **Step 5: Update system_prompt.md**

In `mini_agent/config/system_prompt.md`, update the "Basic Tools" section:

```markdown
### 1. **Basic Tools**
- **File Operations**: Read, write, edit files with full path support
- **Search**: Use `grep` tool to search file contents by regex (do NOT use bash for file searching)
- **Bash Execution**: Run commands, manage git, packages, and system operations
- **MCP Tools**: Access additional tools from configured MCP servers
```

Add a new section after "Specialized Skills":

```markdown
### 3. **Task Tracking**
- For complex tasks (3+ steps), use `todo` tool to create a task list first
- Mark tasks as `in_progress` before starting, `completed` when done
- Use `todo(operation="list")` to review progress and stay on track
```

- [ ] **Step 6: Run full test suite (excluding API-dependent tests)**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && pytest tests/ -v --ignore=tests/test_agent.py --ignore=tests/test_integration.py --ignore=tests/test_session_integration.py --ignore=tests/test_acp.py -x`

Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add mini_agent/cli.py mini_agent/config/system_prompt.md mini_agent/tools/__init__.py
git commit -m "feat: register GrepTool/TodoTool in CLI, pass logging config, update system prompt"
```

- [ ] **Step 8: Final smoke test (manual)**

Run: `cd /Users/liujia/workspace/laoliu/ai/MiniAgent && python -c "
from mini_agent.tools.grep_tool import GrepTool
from mini_agent.tools.todo_tool import TodoTool
from mini_agent.config import Config, LogLevel, LoggingConfig
print('GrepTool schema:', GrepTool().to_schema()['name'])
print('TodoTool schema:', TodoTool().to_schema()['name'])
print('LogLevel:', LogLevel.STANDARD)
print('All imports OK')
"`

Expected: All imports succeed, prints tool names and LogLevel.

---

## Summary

| Task | Files | Dependencies |
|------|-------|-------------|
| 1. Config | `config.py`, `config-example.yaml` | First (foundation) |
| 2. ReadTool | `file_tools.py`, `test_tools.py` | After Task 1 |
| 3. WriteTool | `file_tools.py`, `test_tools.py` | After Task 2 (shared files) |
| 4. EditTool | `file_tools.py`, `test_tools.py` | After Task 3 (shared files) |
| 5. BashTool | `bash_tool.py`, `test_bash_tool.py` | After Task 1 (parallel with 2→3→4) |
| 6. GrepTool | `grep_tool.py` (new), `test_grep_tool.py` (new) | After Task 1 (parallel) |
| 7. TodoTool | `todo_tool.py` (new), `test_todo_tool.py` (new) | After Task 1 (parallel) |
| 8. Logging | `logger.py`, `agent.py`, `test_logging.py` (new) | After Task 1 (parallel) |
| 9. CLI + Prompt | `cli.py`, `system_prompt.md` | After all above |

**Total: 9 tasks, ~45 steps. Tasks 2→3→4 are sequential (shared files). Tasks 5, 6, 7, 8 can run in parallel with each other and the 2→3→4 chain.**
