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

        # STANDARD: log messages + tool names
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
