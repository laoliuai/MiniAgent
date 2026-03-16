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


class TestBashAuditing:
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
        assert any(str(p).startswith(str(workspace)) for p, _ in paths)

    def test_malformed_quotes_fallback(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source)
        paths = guard._extract_paths('echo "hello /tmp/test.txt')
        assert isinstance(paths, list)

    def test_non_path_tokens_skipped(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source)
        paths = guard._extract_paths("echo hello world 42")
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
        guard.audit_command(f"cat {workspace}/file.py")

    def test_audit_command_disabled(self, guard_dirs):
        workspace, source = guard_dirs
        guard = make_guard(workspace, source, enabled=False)
        guard.audit_command(f"cat {source}/core.py")


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
