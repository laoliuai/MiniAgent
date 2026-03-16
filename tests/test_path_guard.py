"""Tests for PathGuard file access control."""
import pytest
from mini_agent.config import PathGuardConfig, ToolsConfig


def test_path_guard_config_defaults():
    cfg = PathGuardConfig()
    assert cfg.enabled is True
    assert cfg.extra_readable_paths == []
    assert cfg.extra_writable_paths == []
    assert cfg.source_whitelist == []


def test_path_guard_config_custom():
    cfg = PathGuardConfig(
        enabled=False,
        extra_readable_paths=["/tmp"],
        extra_writable_paths=["/tmp/out"],
        source_whitelist=["config/config.yaml:rw"],
    )
    assert cfg.enabled is False
    assert cfg.extra_readable_paths == ["/tmp"]


def test_tools_config_has_path_guard():
    tc = ToolsConfig()
    assert isinstance(tc.path_guard, PathGuardConfig)
    assert tc.path_guard.enabled is True


from pathlib import Path
from mini_agent.tools.path_guard import PathGuard, PathGuardError


@pytest.fixture
def guard_dirs(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    (source / "config").mkdir()
    (source / "config" / "config.yaml").touch()
    (source / "config" / "prompt.md").touch()
    (source / "core.py").touch()
    return workspace, source


def make_guard(workspace, source, **kwargs):
    cfg = PathGuardConfig(**kwargs)
    return PathGuard(cfg, workspace, source)


def test_workspace_path_allowed(guard_dirs):
    workspace, source = guard_dirs
    guard = make_guard(workspace, source)
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
    workspace = tmp_path / "repo"
    workspace.mkdir()
    source = workspace / "mini_agent"
    source.mkdir()
    (source / "agent.py").touch()
    guard = make_guard(workspace, source)
    with pytest.raises(PathGuardError, match="agent source code"):
        guard.check(source / "agent.py", "r")
    guard.check(workspace / "README.md", "r")


def test_path_traversal_normalized(guard_dirs):
    workspace, source = guard_dirs
    guard = make_guard(workspace, source)
    with pytest.raises(PathGuardError, match="outside the workspace"):
        guard.check(workspace / ".." / ".." / ".." / "etc" / "passwd", "r")
