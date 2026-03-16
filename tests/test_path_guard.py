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
