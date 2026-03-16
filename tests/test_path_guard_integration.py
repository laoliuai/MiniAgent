"""Integration tests: tools with PathGuard wired up."""
import pytest
import tempfile
from pathlib import Path
from mini_agent.config import PathGuardConfig
from mini_agent.tools.path_guard import PathGuard, PathGuardError
from mini_agent.tools.file_tools import ReadTool, WriteTool, EditTool
from mini_agent.tools.bash_tool import BashTool
from mini_agent.tools.grep_tool import GrepTool


@pytest.fixture
def guarded_env(tmp_path):
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
    result = await tool.execute(path=str(workspace / "editable.txt"), old_str="old content", new_str="new content")
    assert result.success

async def test_edit_tool_source_denied(guarded_env):
    workspace, source, guard = guarded_env
    tool = EditTool(workspace_dir=str(workspace), path_guard=guard)
    result = await tool.execute(path=str(source / "agent.py"), old_str="# agent code", new_str="# modified")
    assert not result.success
    assert "agent source code" in result.error

async def test_edit_tool_outside_denied(guarded_env):
    workspace, source, guard = guarded_env
    tool = EditTool(workspace_dir=str(workspace), path_guard=guard)
    result = await tool.execute(path="/etc/hostname", old_str="old", new_str="new")
    assert not result.success
    assert "outside the workspace" in result.error

async def test_tools_work_without_path_guard():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "test.txt").write_text("content")
        tool = ReadTool(workspace_dir=td)
        result = await tool.execute(path=str(td_path / "test.txt"))
        assert result.success


# --- BashTool + GrepTool integration tests ---

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
