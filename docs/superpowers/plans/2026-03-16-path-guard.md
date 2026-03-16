# PathGuard Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the agent from reading/writing files outside its workspace or modifying its own source code, with configurable exceptions.

**Architecture:** A centralized `PathGuard` class validates all file access. Each tool receives an optional `PathGuard` instance and calls `check()` before any I/O. Bash commands get best-effort auditing via `audit_command()`. Configuration lives under `tools.path_guard` in config.yaml.

**Tech Stack:** Python 3.10+, Pydantic, shlex, re, pytest

**Spec:** `docs/superpowers/specs/2026-03-16-path-guard-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `mini_agent/tools/path_guard.py` (create) | PathGuard class + PathGuardError exception |
| `mini_agent/config.py` (modify) | Add PathGuardConfig model, parse in from_yaml() |
| `mini_agent/tools/file_tools.py` (modify) | Add path_guard param to ReadTool/WriteTool/EditTool |
| `mini_agent/tools/bash_tool.py` (modify) | Add path_guard param to BashTool |
| `mini_agent/tools/grep_tool.py` (modify) | Add path_guard param to GrepTool |
| `mini_agent/cli.py` (modify) | Create PathGuard, pass to tools, inject system prompt section |
| `mini_agent/agent.py` (modify) | Append path policy to system prompt |
| `tests/test_path_guard.py` (create) | Unit tests for PathGuard core + bash auditing |
| `tests/test_path_guard_integration.py` (create) | Integration tests with actual tools |

---

## Chunk 1: Core PathGuard + Config

### Task 1: PathGuardConfig and Config Integration

**Files:**
- Modify: `mini_agent/config.py:65-83` (ToolsConfig) and `mini_agent/config.py:159-181` (from_yaml)
- Test: `tests/test_path_guard.py` (create)

- [ ] **Step 1: Write failing test for PathGuardConfig**

```python
# tests/test_path_guard.py
"""Tests for PathGuard file access control."""
import pytest
from mini_agent.config import PathGuardConfig, ToolsConfig


def test_path_guard_config_defaults():
    """PathGuardConfig has sensible defaults."""
    cfg = PathGuardConfig()
    assert cfg.enabled is True
    assert cfg.extra_readable_paths == []
    assert cfg.extra_writable_paths == []
    assert cfg.source_whitelist == []


def test_path_guard_config_custom():
    """PathGuardConfig accepts custom values."""
    cfg = PathGuardConfig(
        enabled=False,
        extra_readable_paths=["/tmp"],
        extra_writable_paths=["/tmp/out"],
        source_whitelist=["config/config.yaml:rw"],
    )
    assert cfg.enabled is False
    assert cfg.extra_readable_paths == ["/tmp"]


def test_tools_config_has_path_guard():
    """ToolsConfig includes path_guard with defaults."""
    tc = ToolsConfig()
    assert isinstance(tc.path_guard, PathGuardConfig)
    assert tc.path_guard.enabled is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_path_guard.py -v`
Expected: FAIL with `ImportError: cannot import name 'PathGuardConfig'`

- [ ] **Step 3: Add PathGuardConfig to config.py and update ToolsConfig**

In `mini_agent/config.py`, add before the `ToolsConfig` class (around line 64):

```python
class PathGuardConfig(BaseModel):
    """Path access control configuration."""
    enabled: bool = True
    extra_readable_paths: list[str] = []
    extra_writable_paths: list[str] = []
    source_whitelist: list[str] = []
```

Add field to `ToolsConfig`:
```python
class ToolsConfig(BaseModel):
    # ... existing fields ...
    path_guard: PathGuardConfig = Field(default_factory=PathGuardConfig)
```

Update `from_yaml()` — after the existing `mcp_config` parsing (around line 168) and inside the `ToolsConfig(...)` constructor call (line 170-181), add `path_guard` parsing:

```python
        # Parse path guard configuration
        path_guard_data = tools_data.get("path_guard", {})
        path_guard_config = PathGuardConfig(**path_guard_data)

        tools_config = ToolsConfig(
            enable_file_tools=tools_data.get("enable_file_tools", True),
            enable_bash=tools_data.get("enable_bash", True),
            enable_note=tools_data.get("enable_note", True),
            enable_grep=tools_data.get("enable_grep", True),
            enable_todo=tools_data.get("enable_todo", True),
            enable_skills=tools_data.get("enable_skills", True),
            skills_dir=tools_data.get("skills_dir", "./skills"),
            enable_mcp=tools_data.get("enable_mcp", True),
            mcp_config_path=tools_data.get("mcp_config_path", "mcp.json"),
            mcp=mcp_config,
            path_guard=path_guard_config,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_path_guard.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add mini_agent/config.py tests/test_path_guard.py
git commit -m "feat(path_guard): add PathGuardConfig model and config parsing"
```

---

### Task 2: PathGuard Core — check() method

**Files:**
- Create: `mini_agent/tools/path_guard.py`
- Test: `tests/test_path_guard.py` (append)

- [ ] **Step 1: Write failing tests for PathGuard.check()**

Append to `tests/test_path_guard.py`:

```python
from pathlib import Path
from mini_agent.tools.path_guard import PathGuard, PathGuardError
from mini_agent.config import PathGuardConfig


@pytest.fixture
def guard_dirs(tmp_path):
    """Create workspace and source directories for testing."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    # Create a config subdir inside source for whitelist tests
    (source / "config").mkdir()
    (source / "config" / "config.yaml").touch()
    (source / "config" / "prompt.md").touch()
    (source / "core.py").touch()
    return workspace, source


def make_guard(workspace, source, **kwargs):
    """Helper to create PathGuard with defaults."""
    cfg = PathGuardConfig(**kwargs)
    return PathGuard(cfg, workspace, source)


def test_workspace_path_allowed(guard_dirs):
    workspace, source = guard_dirs
    guard = make_guard(workspace, source)
    # Should not raise
    guard.check(workspace / "file.py", "r")
    guard.check(workspace / "file.py", "w")


def test_workspace_subdir_allowed(guard_dirs):
    workspace, source = guard_dirs
    guard = make_guard(workspace, source)
    guard.check(workspace / "sub" / "file.py", "r")
    guard.check(workspace / "sub" / "file.py", "w")


def test_source_path_denied(guard_dirs):
    workspace, source = guard_dirs
    guard = make_guard(workspace, source)
    with pytest.raises(PathGuardError, match="agent source code"):
        guard.check(source / "core.py", "r")


def test_outside_path_denied(guard_dirs):
    workspace, source = guard_dirs
    guard = make_guard(workspace, source)
    with pytest.raises(PathGuardError, match="outside the workspace"):
        guard.check(Path("/etc/passwd"), "r")


def test_disabled_allows_everything(guard_dirs):
    workspace, source = guard_dirs
    guard = make_guard(workspace, source, enabled=False)
    # Should not raise even for source and outside paths
    guard.check(source / "core.py", "r")
    guard.check(Path("/etc/passwd"), "r")


def test_extra_readable_read_allowed(guard_dirs):
    workspace, source = guard_dirs
    guard = make_guard(workspace, source, extra_readable_paths=["/tmp"])
    guard.check(Path("/tmp/data.txt"), "r")


def test_extra_readable_write_denied(guard_dirs):
    workspace, source = guard_dirs
    guard = make_guard(workspace, source, extra_readable_paths=["/tmp"])
    with pytest.raises(PathGuardError, match="read-only"):
        guard.check(Path("/tmp/data.txt"), "w")


def test_extra_writable_both_allowed(guard_dirs):
    workspace, source = guard_dirs
    guard = make_guard(workspace, source, extra_writable_paths=["/tmp/out"])
    guard.check(Path("/tmp/out/file.txt"), "r")
    guard.check(Path("/tmp/out/file.txt"), "w")


def test_source_whitelist_read(guard_dirs):
    workspace, source = guard_dirs
    guard = make_guard(workspace, source, source_whitelist=["config/config.yaml:r"])
    guard.check(source / "config" / "config.yaml", "r")


def test_source_whitelist_read_only_denies_write(guard_dirs):
    workspace, source = guard_dirs
    guard = make_guard(workspace, source, source_whitelist=["config/config.yaml:r"])
    with pytest.raises(PathGuardError, match="read-only in source whitelist"):
        guard.check(source / "config" / "config.yaml", "w")


def test_source_whitelist_rw(guard_dirs):
    workspace, source = guard_dirs
    guard = make_guard(workspace, source, source_whitelist=["config/config.yaml:rw"])
    guard.check(source / "config" / "config.yaml", "r")
    guard.check(source / "config" / "config.yaml", "w")


def test_source_whitelist_directory_prefix(guard_dirs):
    workspace, source = guard_dirs
    guard = make_guard(workspace, source, source_whitelist=["config/:r"])
    guard.check(source / "config" / "config.yaml", "r")
    guard.check(source / "config" / "prompt.md", "r")


def test_source_inside_workspace(tmp_path):
    """When source_dir is inside workspace_dir, source check takes priority."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    source = workspace / "mini_agent"
    source.mkdir()
    (source / "agent.py").touch()
    guard = make_guard(workspace, source)
    # Source file inside workspace should still be denied
    with pytest.raises(PathGuardError, match="agent source code"):
        guard.check(source / "agent.py", "r")
    # Non-source workspace file should be allowed
    guard.check(workspace / "README.md", "r")


def test_path_traversal_normalized(guard_dirs):
    """../.. paths are resolved before checking."""
    workspace, source = guard_dirs
    guard = make_guard(workspace, source)
    # workspace/../../../etc/passwd should resolve to /etc/passwd → denied
    with pytest.raises(PathGuardError, match="outside the workspace"):
        guard.check(workspace / ".." / ".." / ".." / "etc" / "passwd", "r")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_path_guard.py -v -k "not config"`
Expected: FAIL with `ModuleNotFoundError: No module named 'mini_agent.tools.path_guard'`

- [ ] **Step 3: Implement PathGuard core**

Create `mini_agent/tools/path_guard.py`:

```python
"""PathGuard: Agent file access control.

Prevents the agent from accessing files outside its workspace
or modifying its own source code. Best-effort protection against
well-intentioned mistakes, not adversarial attacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from mini_agent.config import PathGuardConfig


class PathGuardError(PermissionError):
    """Raised when path access is denied by PathGuard."""
    pass


class PathGuard:
    """Validates file access against workspace boundaries and source protection."""

    def __init__(
        self,
        config: PathGuardConfig,
        workspace_dir: Path,
        source_dir: Path,
    ):
        self.enabled = config.enabled
        self.workspace_dir = workspace_dir.resolve()
        self.source_dir = source_dir.resolve()
        self._extra_readable = [
            Path(p).expanduser().resolve() for p in config.extra_readable_paths
        ]
        self._extra_writable = [
            Path(p).expanduser().resolve() for p in config.extra_writable_paths
        ]
        self._source_whitelist = self._parse_whitelist(config.source_whitelist)

    def check(self, path: Path, mode: Literal["r", "w"]) -> None:
        """Check if path access is allowed. Raises PathGuardError if denied."""
        if not self.enabled:
            return
        resolved = path.resolve()

        # 1. Source dir → check whitelist (BEFORE workspace check,
        #    because source_dir may be inside workspace_dir)
        if resolved.is_relative_to(self.source_dir):
            self._check_source_whitelist(resolved, mode)
            return

        # 2. Workspace → allow
        if resolved.is_relative_to(self.workspace_dir):
            return

        # 3. Extra writable → allow read+write
        if self._matches_extra(resolved, self._extra_writable):
            return

        # 4. Extra readable → allow read only
        if self._matches_extra(resolved, self._extra_readable):
            if mode == "r":
                return
            raise PathGuardError(
                f"Write denied: {resolved} is read-only"
            )

        # 5. Default deny
        raise PathGuardError(
            f"Access denied: {resolved} is outside the workspace "
            f"and not in the allowed list"
        )

    def _parse_whitelist(self, entries: list[str]) -> list[tuple[Path, str]]:
        """Parse 'relative/path:mode' entries. Supports files and directories."""
        result = []
        for entry in entries:
            parts = entry.rsplit(":", 1)
            rel_path = parts[0]
            mode = parts[1] if len(parts) > 1 else "r"
            result.append((self.source_dir / rel_path, mode))
        return result

    def _check_source_whitelist(self, resolved: Path, mode: str) -> None:
        """Check if a source path is allowed by the whitelist."""
        for wl_path, allowed_mode in self._source_whitelist:
            if resolved == wl_path or resolved.is_relative_to(wl_path):
                if mode == "w" and allowed_mode != "rw":
                    raise PathGuardError(
                        f"Write denied: {resolved} is read-only in source whitelist"
                    )
                return  # Allowed
        raise PathGuardError(
            f"Access denied: {resolved} is agent source code"
        )

    @staticmethod
    def _matches_extra(resolved: Path, extra_paths: list[Path]) -> bool:
        """Check if resolved path is inside any of the extra paths."""
        return any(resolved.is_relative_to(p) for p in extra_paths)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_path_guard.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add mini_agent/tools/path_guard.py tests/test_path_guard.py
git commit -m "feat(path_guard): implement PathGuard core with check() method"
```

---

### Task 3: PathGuard Bash Auditing — audit_command() and _extract_paths()

**Files:**
- Modify: `mini_agent/tools/path_guard.py`
- Test: `tests/test_path_guard.py` (append)

- [ ] **Step 1: Write failing tests for bash auditing**

Append to `tests/test_path_guard.py`:

```python
class TestBashAuditing:
    """Tests for PathGuard.audit_command() and _extract_paths()."""

    def test_read_command_detected(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source)
        paths = guard._extract_paths("cat /etc/passwd")
        assert any(str(p).endswith("passwd") and m == "r" for p, m in paths)

    def test_write_redirect_detected(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source)
        paths = guard._extract_paths("echo hello > /tmp/out.txt")
        assert any(str(p).endswith("out.txt") and m == "w" for p, m in paths)

    def test_append_redirect_detected(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source)
        paths = guard._extract_paths("echo hello >> /tmp/log.txt")
        assert any(str(p).endswith("log.txt") and m == "w" for p, m in paths)

    def test_write_command_detected(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source)
        paths = guard._extract_paths("cp /tmp/a.txt /tmp/b.txt")
        assert any(m == "w" for _, m in paths)

    def test_rm_detected_as_write(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source)
        paths = guard._extract_paths("rm /tmp/secret.txt")
        assert any(str(p).endswith("secret.txt") and m == "w" for p, m in paths)

    def test_sed_inplace_detected(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source)
        paths = guard._extract_paths("sed -i 's/old/new/' /tmp/file.txt")
        assert any(str(p).endswith("file.txt") and m == "w" for p, m in paths)

    def test_pipe_segments_analyzed(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source)
        paths = guard._extract_paths("cat /tmp/a.txt | grep foo")
        assert any(str(p).endswith("a.txt") and m == "r" for p, m in paths)

    def test_semicolon_segments_analyzed(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source)
        paths = guard._extract_paths("cd /tmp; rm ./file.txt")
        assert any(m == "w" for _, m in paths)

    def test_relative_path_resolved_against_workspace(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source)
        paths = guard._extract_paths("cat ./file.py")
        # Should resolve to workspace/file.py, not CWD/file.py
        assert any(str(p).startswith(str(workspace)) for p, _ in paths)

    def test_malformed_quotes_fallback(self, guard_dirs):
        """shlex.split failure falls back to str.split."""
        workspace, source = guard_dirs
        guard = make_guard(workspace, source)
        # Unclosed quote — should not raise, falls back gracefully
        paths = guard._extract_paths('echo "hello /tmp/test.txt')
        # Best-effort: may or may not detect the path, but should not crash
        assert isinstance(paths, list)

    def test_non_path_tokens_skipped(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source)
        paths = guard._extract_paths("echo hello world 42")
        # None of these are path-like
        assert len(paths) == 0

    def test_flags_skipped(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source)
        paths = guard._extract_paths("ls -la /tmp")
        path_strs = [str(p) for p, _ in paths]
        assert not any("-la" in s for s in path_strs)

    def test_audit_command_denies_source_access(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source)
        with pytest.raises(PathGuardError, match="agent source code"):
            guard.audit_command(f"cat {source}/core.py")

    def test_audit_command_allows_workspace(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source)
        # Should not raise
        guard.audit_command(f"cat {workspace}/file.py")

    def test_audit_command_disabled(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source, enabled=False)
        # Should not raise even with source path
        guard.audit_command(f"cat {source}/core.py")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_path_guard.py::TestBashAuditing -v`
Expected: FAIL with `AttributeError: 'PathGuard' object has no attribute '_extract_paths'`

- [ ] **Step 3: Add bash auditing to PathGuard**

Add to `mini_agent/tools/path_guard.py`, at the top (imports):

```python
import re
import shlex
```

Add these class attributes and methods to the `PathGuard` class:

```python
    # Commands that write to their path arguments
    _WRITE_COMMANDS = {
        "tee", "mv", "cp", "install", "rsync",
        "rm", "mkdir", "touch", "ln", "dd", "chmod", "chown",
    }
    # Commands with in-place modification flags
    _INPLACE_FLAGS = {"sed": "-i", "perl": "-i", "sort": "-o"}

    _WRITE_REDIRECTS = re.compile(r'>{1,2}\s*([~./][\w./_-]*|/[\w./_-]+)')

    def audit_command(self, command: str) -> None:
        """Best-effort check of bash command string."""
        if not self.enabled:
            return
        for path, mode in self._extract_paths(command):
            self.check(path, mode)

    def _extract_paths(self, command: str) -> list[tuple[Path, str]]:
        """Extract path-like tokens from a bash command string."""
        results: list[tuple[Path, str]] = []

        # 1. Redirect targets (> file, >> file) → write
        for m in self._WRITE_REDIRECTS.finditer(command):
            p = Path(m.group(1)).expanduser()
            if not p.is_absolute():
                p = self.workspace_dir / p
            results.append((p, "w"))

        # 2. Split on pipes/semicolons, analyze each segment
        segments = re.split(r'[|;]|&&', command)
        for segment in segments:
            try:
                tokens = shlex.split(segment, posix=True)
            except ValueError:
                tokens = segment.split()  # fallback for malformed quotes
            if not tokens:
                continue
            cmd = Path(tokens[0]).name  # strip /usr/bin/ prefix
            is_write_cmd = cmd in self._WRITE_COMMANDS
            has_inplace = (
                cmd in self._INPLACE_FLAGS
                and self._INPLACE_FLAGS[cmd] in tokens
            )

            for token in tokens[1:]:
                if token.startswith("-"):
                    continue  # skip flags
                p = Path(token).expanduser()
                if not (
                    str(p).startswith("/")
                    or str(p).startswith(".")
                    or str(p).startswith("~")
                ):
                    continue  # not a path-like token
                # Resolve relative paths against workspace_dir (BashTool's CWD)
                if not p.is_absolute():
                    p = self.workspace_dir / p
                mode = "w" if (is_write_cmd or has_inplace) else "r"
                results.append((p, mode))

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_path_guard.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add mini_agent/tools/path_guard.py tests/test_path_guard.py
git commit -m "feat(path_guard): add bash command auditing with _extract_paths()"
```

---

### Task 4: PathGuard Violation Logging

**Files:**
- Modify: `mini_agent/tools/path_guard.py`
- Test: `tests/test_path_guard.py` (append)

- [ ] **Step 1: Write failing test for logging**

Append to `tests/test_path_guard.py`:

```python
from unittest.mock import MagicMock


def test_check_logs_denial(guard_dirs):
    """PathGuard logs violations when a logger is provided."""
    workspace, source = guard_dirs
    mock_logger = MagicMock()
    guard = make_guard(workspace, source)
    guard.logger = mock_logger
    with pytest.raises(PathGuardError):
        guard.check(source / "core.py", "r")
    mock_logger.log_context_event.assert_called_once()
    call_args = mock_logger.log_context_event.call_args
    assert "path_guard" in call_args[0][0]
    assert "DENIED" in call_args[0][1]


def test_check_no_log_when_allowed(guard_dirs):
    """PathGuard does not log when access is allowed."""
    workspace, source = guard_dirs
    mock_logger = MagicMock()
    guard = make_guard(workspace, source)
    guard.logger = mock_logger
    guard.check(workspace / "file.py", "r")
    mock_logger.log_context_event.assert_not_called()


def test_check_no_log_without_logger(guard_dirs):
    """PathGuard works without logger (backward compat)."""
    workspace, source = guard_dirs
    guard = make_guard(workspace, source)
    assert guard.logger is None
    with pytest.raises(PathGuardError):
        guard.check(source / "core.py", "r")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_path_guard.py -v -k "log"`
Expected: FAIL with `AttributeError: 'PathGuard' object has no attribute 'logger'`

- [ ] **Step 3: Add logger support to PathGuard**

In `mini_agent/tools/path_guard.py`, add `logger` attribute to `__init__`:

```python
    def __init__(
        self,
        config: PathGuardConfig,
        workspace_dir: Path,
        source_dir: Path,
        logger=None,
    ):
        self.enabled = config.enabled
        self.workspace_dir = workspace_dir.resolve()
        self.source_dir = source_dir.resolve()
        self.logger = logger
        # ... rest unchanged ...
```

In the `check()` method, add logging before each `raise PathGuardError(...)`:

```python
    def _deny(self, mode: str, resolved: Path, message: str) -> None:
        """Log denial and raise PathGuardError."""
        if self.logger:
            self.logger.log_context_event("path_guard", f"DENIED {mode} {resolved}")
        raise PathGuardError(message)
```

Then replace each `raise PathGuardError(msg)` in `check()` and `_check_source_whitelist()` with `self._deny(mode, resolved, msg)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_path_guard.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add mini_agent/tools/path_guard.py tests/test_path_guard.py
git commit -m "feat(path_guard): log violations via AgentLogger"
```

---

## Chunk 2: Tool Integration

### Task 5: Add path_guard to File Tools (ReadTool, WriteTool, EditTool)

**Files:**
- Modify: `mini_agent/tools/file_tools.py:72-78` (ReadTool.__init__), `184-190` (WriteTool.__init__), `244-250` (EditTool.__init__), and their `execute()` methods
- Test: `tests/test_path_guard_integration.py` (create)

- [ ] **Step 1: Write failing integration tests**

Create `tests/test_path_guard_integration.py`:

```python
"""Integration tests: tools with PathGuard wired up."""
import pytest
import tempfile
from pathlib import Path

from mini_agent.config import PathGuardConfig
from mini_agent.tools.path_guard import PathGuard, PathGuardError
from mini_agent.tools.file_tools import ReadTool, WriteTool, EditTool


@pytest.fixture
def guarded_env(tmp_path):
    """Create a guarded environment with workspace and source dirs."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    (source / "agent.py").write_text("# agent code")

    cfg = PathGuardConfig()
    guard = PathGuard(cfg, workspace, source)
    return workspace, source, guard


async def test_read_tool_workspace_allowed(guarded_env):
    workspace, source, guard = guarded_env
    (workspace / "test.txt").write_text("hello")
    tool = ReadTool(workspace_dir=str(workspace), path_guard=guard)
    result = await tool.execute(path=str(workspace / "test.txt"))
    assert result.success


async def test_read_tool_source_denied(guarded_env):
    workspace, source, guard = guarded_env
    tool = ReadTool(workspace_dir=str(workspace), path_guard=guard)
    result = await tool.execute(path=str(source / "agent.py"))
    assert not result.success
    assert "agent source code" in result.error


async def test_read_tool_outside_denied(guarded_env):
    workspace, source, guard = guarded_env
    tool = ReadTool(workspace_dir=str(workspace), path_guard=guard)
    result = await tool.execute(path="/etc/hostname")
    assert not result.success
    assert "outside the workspace" in result.error


async def test_write_tool_workspace_allowed(guarded_env):
    workspace, source, guard = guarded_env
    tool = WriteTool(workspace_dir=str(workspace), path_guard=guard)
    result = await tool.execute(path=str(workspace / "new.txt"), content="data")
    assert result.success


async def test_write_tool_source_denied(guarded_env):
    workspace, source, guard = guarded_env
    tool = WriteTool(workspace_dir=str(workspace), path_guard=guard)
    result = await tool.execute(path=str(source / "agent.py"), content="hacked")
    assert not result.success
    assert "agent source code" in result.error


async def test_write_tool_outside_denied(guarded_env):
    workspace, source, guard = guarded_env
    tool = WriteTool(workspace_dir=str(workspace), path_guard=guard)
    result = await tool.execute(path="/etc/test_output.txt", content="data")
    assert not result.success
    assert "outside the workspace" in result.error


async def test_edit_tool_workspace_allowed(guarded_env):
    workspace, source, guard = guarded_env
    (workspace / "editable.txt").write_text("old content")
    tool = EditTool(workspace_dir=str(workspace), path_guard=guard)
    result = await tool.execute(
        path=str(workspace / "editable.txt"),
        old_str="old content",
        new_str="new content",
    )
    assert result.success


async def test_edit_tool_source_denied(guarded_env):
    workspace, source, guard = guarded_env
    tool = EditTool(workspace_dir=str(workspace), path_guard=guard)
    result = await tool.execute(
        path=str(source / "agent.py"),
        old_str="# agent code",
        new_str="# modified",
    )
    assert not result.success
    assert "agent source code" in result.error


async def test_edit_tool_outside_denied(guarded_env):
    workspace, source, guard = guarded_env
    tool = EditTool(workspace_dir=str(workspace), path_guard=guard)
    result = await tool.execute(
        path="/etc/hostname",
        old_str="old",
        new_str="new",
    )
    assert not result.success
    assert "outside the workspace" in result.error


async def test_tools_work_without_path_guard():
    """Tools still work when path_guard is None (backward compat)."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "test.txt").write_text("content")
        tool = ReadTool(workspace_dir=td)
        result = await tool.execute(path=str(td_path / "test.txt"))
        assert result.success
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_path_guard_integration.py -v`
Expected: FAIL with `TypeError: ReadTool.__init__() got an unexpected keyword argument 'path_guard'`

- [ ] **Step 3: Add path_guard parameter to file tools**

In `mini_agent/tools/file_tools.py`:

**ReadTool.__init__** (line 72-78): Add `path_guard` parameter:
```python
    def __init__(self, workspace_dir: str = ".", path_guard=None):
        self.workspace_dir = Path(workspace_dir).absolute()
        self.path_guard = path_guard
```

**ReadTool.execute** (line 114-178): Add check after path resolution (after line 120):
```python
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.workspace_dir / file_path

            if self.path_guard:
                self.path_guard.check(file_path, "r")

            if not file_path.exists():
```

**WriteTool.__init__** (line 184-190): Same pattern:
```python
    def __init__(self, workspace_dir: str = ".", path_guard=None):
        self.workspace_dir = Path(workspace_dir).absolute()
        self.path_guard = path_guard
```

**WriteTool.execute** (line 221-238): Add check after path resolution (after line 227):
```python
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.workspace_dir / file_path

            if self.path_guard:
                self.path_guard.check(file_path, "w")

            file_path.parent.mkdir(parents=True, exist_ok=True)
```

**EditTool.__init__** (line 244-250): Same pattern:
```python
    def __init__(self, workspace_dir: str = ".", path_guard=None):
        self.workspace_dir = Path(workspace_dir).absolute()
        self.path_guard = path_guard
```

**EditTool.execute** (line 290-334): Add check after path resolution (after line 295):
```python
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.workspace_dir / file_path

            if self.path_guard:
                self.path_guard.check(file_path, "w")

            if not file_path.exists():
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_path_guard_integration.py -v`
Expected: All PASSED

- [ ] **Step 5: Run existing tool tests to ensure no regression**

Run: `uv run pytest tests/test_tools.py -v`
Expected: All PASSED (existing tests don't pass path_guard, tools work without it)

- [ ] **Step 6: Commit**

```bash
git add mini_agent/tools/file_tools.py tests/test_path_guard_integration.py
git commit -m "feat(path_guard): integrate PathGuard into ReadTool/WriteTool/EditTool"
```

---

### Task 6: Add path_guard to BashTool and GrepTool

**Files:**
- Modify: `mini_agent/tools/bash_tool.py:228-238` (BashTool.__init__)
- Modify: `mini_agent/tools/grep_tool.py:21-22` (GrepTool.__init__)
- Test: `tests/test_path_guard_integration.py` (append)

- [ ] **Step 1: Write failing integration tests**

Append to `tests/test_path_guard_integration.py`:

```python
from mini_agent.tools.bash_tool import BashTool
from mini_agent.tools.grep_tool import GrepTool


async def test_bash_tool_source_access_denied(guarded_env):
    workspace, source, guard = guarded_env
    tool = BashTool(workspace_dir=str(workspace), path_guard=guard)
    result = await tool.execute(command=f"cat {source}/agent.py")
    assert not result.success
    assert "agent source code" in result.error


async def test_bash_tool_workspace_allowed(guarded_env):
    workspace, source, guard = guarded_env
    (workspace / "hello.txt").write_text("hello")
    tool = BashTool(workspace_dir=str(workspace), path_guard=guard)
    result = await tool.execute(command=f"cat {workspace}/hello.txt")
    assert result.success


async def test_bash_tool_outside_denied(guarded_env):
    workspace, source, guard = guarded_env
    tool = BashTool(workspace_dir=str(workspace), path_guard=guard)
    result = await tool.execute(command="cat /etc/passwd")
    assert not result.success
    assert "outside the workspace" in result.error


async def test_grep_tool_source_denied(guarded_env):
    workspace, source, guard = guarded_env
    tool = GrepTool(workspace_dir=str(workspace), path_guard=guard)
    result = await tool.execute(pattern="agent", path=str(source))
    assert not result.success
    assert "agent source code" in result.error


async def test_grep_tool_workspace_allowed(guarded_env):
    workspace, source, guard = guarded_env
    (workspace / "code.py").write_text("def hello(): pass")
    tool = GrepTool(workspace_dir=str(workspace), path_guard=guard)
    result = await tool.execute(pattern="hello", path=str(workspace))
    assert result.success
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_path_guard_integration.py -v -k "bash or grep"`
Expected: FAIL with `TypeError: BashTool.__init__() got an unexpected keyword argument 'path_guard'`

- [ ] **Step 3: Add path_guard to BashTool**

In `mini_agent/tools/bash_tool.py`, modify `BashTool.__init__` (line 228-238):

```python
    def __init__(self, workspace_dir: str | None = None, path_guard=None):
        self.is_windows = platform.system() == "Windows"
        self.shell_name = "PowerShell" if self.is_windows else "bash"
        self.workspace_dir = workspace_dir
        self.path_guard = path_guard
```

In `BashTool.execute` (line 312-329), add audit after the `try:` (line 329) and before timeout validation (line 330):

```python
        try:
            if self.path_guard:
                self.path_guard.audit_command(command)

            # Validate timeout
```

- [ ] **Step 4: Add path_guard to GrepTool**

In `mini_agent/tools/grep_tool.py`, modify `GrepTool.__init__` (line 21-22):

```python
    def __init__(self, workspace_dir: str = ".", path_guard=None):
        self.workspace_dir = Path(workspace_dir).absolute()
        self.path_guard = path_guard
```

In `GrepTool.execute`, add check after path resolution. Find the line where `_resolve_path` is called and add after it:

```python
        search_path = self._resolve_path(path)
        if self.path_guard:
            self.path_guard.check(search_path, "r")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_path_guard_integration.py -v`
Expected: All PASSED

- [ ] **Step 6: Run existing tests for regression**

Run: `uv run pytest tests/test_bash_tool.py tests/test_tools.py -v`
Expected: All PASSED

- [ ] **Step 7: Commit**

```bash
git add mini_agent/tools/bash_tool.py mini_agent/tools/grep_tool.py tests/test_path_guard_integration.py
git commit -m "feat(path_guard): integrate PathGuard into BashTool and GrepTool"
```

---

## Chunk 3: CLI/Agent Integration + System Prompt

### Task 7: Create PathGuard in CLI and pass to tools

**Files:**
- Modify: `mini_agent/cli.py:434-477` (add_workspace_tools function)

- [ ] **Step 1: Add PathGuard creation and tool wiring in add_workspace_tools()**

In `mini_agent/cli.py`, add import near the top (with other tool imports):

```python
from mini_agent.tools.path_guard import PathGuard
```

Modify `add_workspace_tools()` (line 434-477). Add PathGuard creation at the start of the function, after `workspace_dir.mkdir()` (line 445):

```python
    # Create PathGuard for file access control
    source_dir = Config.get_package_dir()  # mini_agent/
    path_guard = PathGuard(config.tools.path_guard, workspace_dir, source_dir)
    if config.tools.path_guard.enabled:
        print(f"{Colors.GREEN}✅ PathGuard enabled (workspace: {workspace_dir}){Colors.RESET}")
```

Then update each tool instantiation to pass `path_guard`:

```python
    if config.tools.enable_bash:
        bash_tool = BashTool(workspace_dir=str(workspace_dir), path_guard=path_guard)
        # ...

    if config.tools.enable_file_tools:
        tools.extend([
            ReadTool(workspace_dir=str(workspace_dir), path_guard=path_guard),
            WriteTool(workspace_dir=str(workspace_dir), path_guard=path_guard),
            EditTool(workspace_dir=str(workspace_dir), path_guard=path_guard),
        ])
        # ...

    if config.tools.enable_grep:
        tools.append(GrepTool(workspace_dir=str(workspace_dir), path_guard=path_guard))
        # ...
```

- [ ] **Step 2: Run all tests to verify no regression**

Run: `uv run pytest tests/ -v`
Expected: All PASSED

- [ ] **Step 3: Commit**

```bash
git add mini_agent/cli.py
git commit -m "feat(path_guard): wire PathGuard into CLI tool initialization"
```

---

### Task 8: Add path policy to system prompt

**Files:**
- Modify: `mini_agent/agent.py:55-60` (system prompt injection section)

- [ ] **Step 1: Add path policy section to system prompt**

In `mini_agent/agent.py`, after the workspace info injection (line 58), add path policy:

```python
        if "Current Workspace" not in system_prompt:
            workspace_info = f"\n\n## Current Workspace\nYou are currently working in: `{self.workspace_dir.absolute()}`\nAll relative paths will be resolved relative to this directory."
            system_prompt = system_prompt + workspace_info

        # Append path access policy
        path_policy = (
            "\n\n## Path Access Policy\n"
            "You operate under file access restrictions:\n"
            "- Full read/write access within the workspace directory\n"
            "- Access outside the workspace is restricted\n"
            "- Your own source code is not accessible\n\n"
            "If a file operation is denied, inform the user about the restriction.\n"
            "Do NOT attempt to work around restrictions via bash, alternative paths, "
            "or encoded commands."
        )
        system_prompt = system_prompt + path_policy

        self.system_prompt = system_prompt
```

- [ ] **Step 2: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All PASSED

- [ ] **Step 3: Commit**

```bash
git add mini_agent/agent.py
git commit -m "feat(path_guard): inject path access policy into system prompt"
```

---

### Task 9: Export PathGuard from tools package

**Files:**
- Modify: `mini_agent/tools/__init__.py`

- [ ] **Step 1: Add PathGuard and PathGuardError to exports**

In `mini_agent/tools/__init__.py`, add:

```python
from .path_guard import PathGuard, PathGuardError
```

And update `__all__` to include them.

- [ ] **Step 2: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All PASSED

- [ ] **Step 3: Commit**

```bash
git add mini_agent/tools/__init__.py
git commit -m "feat(path_guard): export PathGuard from tools package"
```

---

### Task 10: Final — Run full test suite

- [ ] **Step 1: Run entire test suite with coverage**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests pass, no regressions.

- [ ] **Step 2: Verify PathGuard test count**

Run: `uv run pytest tests/test_path_guard.py tests/test_path_guard_integration.py -v --tb=short`
Expected: ~40+ tests covering core checks, bash auditing, logging, and tool integration.
