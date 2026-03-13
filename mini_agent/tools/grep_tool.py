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
