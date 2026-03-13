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
