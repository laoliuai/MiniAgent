# PathGuard: Agent File Access Control

## Background

MiniAgent's tools (ReadTool, WriteTool, EditTool, BashTool, GrepTool) currently have **no path restrictions**. The agent can read and modify any file the process has OS-level access to — including its own source code. This was discovered when the agent, asked about adding a feature, inspected its own implementation and then modified it.

### Problem

1. **Self-modification** — Agent can read and write `mini_agent/` source files, potentially breaking itself
2. **Path traversal** — File tools accept absolute paths and `../` without validation; workspace boundary is not enforced
3. **Bash escape** — Even with file tool restrictions, Bash provides unrestricted filesystem access
4. **No configurability** — No way to grant limited access to paths outside workspace

### Threat Model

This is **not** adversarial security (defending against prompt injection attacks). The goal is preventing the agent from "helpfully" accessing or modifying things it shouldn't — the "good intentions, bad judgment" scenario. Best-effort protection is acceptable.

## Design

### Permission Zones

```
┌─────────────────────────────────────────────┐
│  Filesystem                                 │
│                                             │
│  ┌─ workspace_dir ──────────────────────┐   │
│  │  Full read+write access              │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  ┌─ Agent source dir (mini_agent/) ─────┐   │
│  │  Default: denied (read and write)    │   │
│  │  Whitelist entries: per-file r / rw  │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  All other paths:                           │
│    Default: denied                          │
│    Configurable extra_readable_paths (r)    │
│    Configurable extra_writable_paths (r+w)  │
└─────────────────────────────────────────────┘
```

### Configuration

Under `tools.path_guard` in `config.yaml`:

```yaml
tools:
  path_guard:
    enabled: true  # false disables all checks (for development/debugging)

    # Paths outside workspace that agent can read
    extra_readable_paths:
      - "/tmp"
      - "~/.config/some-app"

    # Paths outside workspace that agent can read+write
    extra_writable_paths:
      - "/tmp/agent-output"

    # Agent source files allowed through (relative to mini_agent/)
    # Format: "relative/path:r" or "relative/path:rw"
    source_whitelist:
      - "config/config.yaml:rw"
      - "config/system_prompt.md:r"
```

**Pydantic config model:**

```python
class PathGuardConfig(BaseModel):
    enabled: bool = True
    extra_readable_paths: list[str] = []
    extra_writable_paths: list[str] = []
    source_whitelist: list[str] = []
```

Integrated into existing `ToolsConfig`:

```python
class ToolsConfig(BaseModel):
    # ... existing fields ...
    path_guard: PathGuardConfig = PathGuardConfig()
```

**Config.from_yaml() update:** The `from_yaml()` method must parse `path_guard` from `tools_data`:

```python
# In Config.from_yaml(), after existing tools parsing
path_guard_data = tools_data.get("path_guard", {})
path_guard_config = PathGuardConfig(**path_guard_data)

tools_config = ToolsConfig(
    # ... existing fields ...
    path_guard=path_guard_config,
)
```

### PathGuard Module

**File:** `mini_agent/tools/path_guard.py`

```python
class PathGuard:
    def __init__(self, config: PathGuardConfig,
                 workspace_dir: Path, source_dir: Path):
        self.enabled = config.enabled
        self.workspace_dir = workspace_dir.resolve()
        self.source_dir = source_dir.resolve()
        self._extra_readable = [Path(p).expanduser().resolve()
                                for p in config.extra_readable_paths]
        self._extra_writable = [Path(p).expanduser().resolve()
                                for p in config.extra_writable_paths]
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
                f"Write denied: {resolved} is read-only")

        # 5. Default deny
        raise PathGuardError(
            f"Access denied: {resolved} is outside the workspace "
            f"and not in the allowed list")

    def audit_command(self, command: str) -> None:
        """Best-effort check of bash command string."""
        if not self.enabled:
            return
        for path, mode in self._extract_paths(command):
            self.check(path, mode)
```

**Key implementation details:**

- `path.resolve()` called first — normalizes `../`, resolves symlinks (including symlinks within workspace that point outside it)
- **Source check runs before workspace check** — prevents bypass when `source_dir` is a subdirectory of `workspace_dir` (common case: workspace is the repo root, `mini_agent/` is inside it)
- Write permission implies read — `extra_writable_paths` entries are auto-readable
- `source_dir` discovered via `Config.get_package_dir()` (returns `Path(__file__).parent` from `config.py`, i.e. `mini_agent/`)
- `_matches_extra()` checks `is_relative_to()` — a path matches if it's inside any listed directory
- **Immutable after construction** — all config is resolved in `__init__`. No runtime mutation, safe to share across concurrent tool calls

### Source Whitelist Parsing

```python
def _parse_whitelist(self, entries: list[str]) -> list[tuple[Path, str]]:
    """Parse 'relative/path:mode' entries. Supports files and directories."""
    result = []
    for entry in entries:
        parts = entry.rsplit(":", 1)
        rel_path = parts[0]
        mode = parts[1] if len(parts) > 1 else "r"
        result.append((self.source_dir / rel_path, mode))  # "r" or "rw"
    return result

def _check_source_whitelist(self, resolved: Path, mode: str):
    # Check both exact match and directory prefix match
    for wl_path, allowed_mode in self._source_whitelist:
        if resolved == wl_path or resolved.is_relative_to(wl_path):
            if mode == "w" and allowed_mode != "rw":
                raise PathGuardError(
                    f"Write denied: {resolved} is read-only in source whitelist")
            return  # Allowed
    raise PathGuardError(
        f"Access denied: {resolved} is agent source code")
```

Whitelist supports both exact file paths and directory prefixes:
- `"config/config.yaml:rw"` — matches only that file
- `"config/:r"` — matches all files under `mini_agent/config/`

### Bash Command Auditing

**File:** `mini_agent/tools/path_guard.py` (part of PathGuard class)

```python
# Commands that write to their last path argument
_WRITE_COMMANDS = {
    "tee", "mv", "cp", "install", "rsync",
    "rm", "mkdir", "touch", "ln", "dd", "chmod", "chown",
}
# sed -i is a special case (modifies files in-place)
_INPLACE_FLAGS = {"sed": "-i", "perl": "-i", "sort": "-o"}

_WRITE_REDIRECTS = re.compile(r'>{1,2}\s*([~./][\w./_-]*|/[\w./_-]+)')

def _extract_paths(self, command: str) -> list[tuple[Path, str]]:
    results = []

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
        has_inplace = (cmd in self._INPLACE_FLAGS
                       and self._INPLACE_FLAGS[cmd] in tokens)

        for token in tokens[1:]:
            if token.startswith("-"):
                continue  # skip flags
            p = Path(token).expanduser()
            if not (str(p).startswith("/") or str(p).startswith(".")
                    or str(p).startswith("~")):
                continue  # not a path-like token
            # Resolve relative paths against workspace_dir (BashTool's CWD),
            # not the Python process CWD
            if not p.is_absolute():
                p = self.workspace_dir / p
            mode = "w" if (is_write_cmd or has_inplace) else "r"
            results.append((p, mode))

    return results
```

Uses `shlex.split()` for proper token parsing (handles quotes, escapes). Splits on `|`, `;`, `&&` to analyze each command segment independently.

**Known limitations (accepted by design):**
- Variable expansion (`$HOME/secret`) — not detected
- Pipe chains (`echo | python -c "open(...)"`) — not detected
- Quoted paths with special chars — partial detection (e.g., `> "/tmp/my file.txt"` not caught by redirect regex)
- Malformed quotes (unclosed `"`) — falls back to `str.split()`, best-effort
- This is best-effort: catches common `cat /path`, `vim /path`, `> /path` patterns

### Tool Integration

Each tool calls PathGuard at the start of `execute()`:

**File tools (ReadTool, WriteTool, EditTool):**
```python
class ReadTool(Tool):
    def __init__(self, workspace_dir: str, path_guard: PathGuard | None = None):
        self.path_guard = path_guard
        # ... existing init ...

    async def execute(self, file_path: str, ...):
        resolved = self._resolve_path(file_path)
        if self.path_guard:
            self.path_guard.check(resolved, "r")  # raises PathGuardError
        # ... existing logic ...
```

WriteTool/EditTool: same pattern with `check(path, "w")`.

**GrepTool:**
```python
async def execute(self, pattern: str, path: str = ".", ...):
    resolved = self._resolve_path(path)
    if self.path_guard:
        self.path_guard.check(resolved, "r")
    # ... existing logic ...
```

**Note:** GrepTool searches recursively within a directory. When `path="."` (default, resolves to workspace), the search could traverse into `source_dir` if it's inside the workspace. This is acceptable — GrepTool's underlying `ripgrep`/`grep` reads files but doesn't modify them, and the agent source code is not secret (it's open source). The primary goal of PathGuard is preventing **writes** to source code and preventing access to paths **outside** the workspace. If stricter grep isolation is needed later, GrepTool can pass `--glob='!mini_agent/'` to ripgrep.

**BashTool:**
```python
async def execute(self, command: str, ...):
    if self.path_guard:
        self.path_guard.audit_command(command)  # raises PathGuardError
    # ... existing logic ...
```

**PathGuard is optional** (`path_guard: PathGuard | None = None`) — tools work without it for backward compatibility and testing.

**Error propagation:** `PathGuardError` inherits from `PermissionError`. The existing tools already have `except Exception` handlers that convert exceptions to failed `ToolResult(success=False, error=str(e))`. No dedicated `except PathGuardError` block is needed — the existing handlers produce clear error messages since `PathGuardError` includes descriptive text. For BashTool, the error similarly propagates through its existing exception handling to return a failed `BashOutputResult`.

### Agent / CLI Integration

**Agent constructor** creates PathGuard and passes to tools:

```python
# In Agent.__init__ or CLI tool initialization
from mini_agent.tools.path_guard import PathGuard

source_dir = Path(__file__).parent  # mini_agent/
path_guard = PathGuard(config.tools.path_guard, workspace_dir, source_dir)

# Pass to each tool
tools = [
    ReadTool(workspace_dir=str(workspace_dir), path_guard=path_guard),
    WriteTool(workspace_dir=str(workspace_dir), path_guard=path_guard),
    # ...
]
```

### System Prompt Injection

Agent appends a path policy section to the system prompt (alongside existing workspace info):

```
## Path Access Policy
You operate under file access restrictions:
- Full read/write access within the workspace directory
- Access outside the workspace is restricted
- Your own source code is not accessible

If a file operation is denied, inform the user about the restriction.
Do NOT attempt to work around restrictions via bash, alternative paths,
or encoded commands.
```

**Does not list** white-listed paths or extra paths — avoids prompting the agent to explore edges.

### Error Handling

PathGuardError is a subclass of PermissionError. Tools catch it and return a failed ToolResult:

```python
class PathGuardError(PermissionError):
    """Raised when path access is denied by PathGuard."""
    pass
```

Error messages are descriptive but don't reveal the full ruleset:
- `"Access denied: /path/to/file is outside the workspace and not in the allowed list"`
- `"Write denied: /path/to/file is read-only"`
- `"Access denied: /path/to/file is agent source code"`

### Logging

PathGuard violations are logged via AgentLogger:
- File log at STANDARD level
- Console at VERBOSE level
- Format: `[CONTEXT] path_guard: DENIED w /path/to/file`

### Scope Exclusions

- **MCP tools** — run in external processes, not covered by PathGuard
- **Skill tools** — loaded from git submodule, not covered (they execute arbitrary code anyway)
- **Docker/seccomp sandboxing** — out of scope, this is application-level protection
- **Adversarial prompt injection defense** — out of scope, this prevents well-intentioned mistakes

## Testing Strategy

1. **Unit tests for PathGuard** — workspace paths allowed, source paths denied, whitelist entries, extra paths, mode checks
2. **Unit tests for Bash auditing** — redirect detection, path extraction, known limitation cases
3. **Integration tests** — ReadTool/WriteTool/BashTool with PathGuard wired up, verify denials
4. **Config tests** — PathGuardConfig parsing, enabled/disabled toggle
