"""PathGuard: Agent file access control."""
from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Literal

from mini_agent.config import PathGuardConfig


class PathGuardError(PermissionError):
    """Raised when a file access violates PathGuard policy."""

    pass


class PathGuard:
    """File access control that restricts agent operations to allowed paths."""

    _WRITE_COMMANDS = frozenset({
        "cp", "mv", "rm", "rmdir", "mkdir", "touch", "tee",
        "dd", "install", "rsync", "chmod", "chown", "chgrp",
        "ln", "unlink", "truncate",
    })

    _INPLACE_FLAGS = frozenset({"-i", "--in-place"})

    _WRITE_REDIRECTS = re.compile(r">{1,2}")

    def __init__(self, config: PathGuardConfig, workspace_dir: Path, source_dir: Path, logger=None):
        self.enabled = config.enabled
        self.workspace_dir = workspace_dir.resolve()
        self.source_dir = source_dir.resolve()
        self._extra_readable = [Path(p).expanduser().resolve() for p in config.extra_readable_paths]
        self._extra_writable = [Path(p).expanduser().resolve() for p in config.extra_writable_paths]
        self._source_whitelist = self._parse_whitelist(config.source_whitelist)
        self.logger = logger

    def _deny(self, mode: str, resolved: Path, message: str) -> None:
        if self.logger:
            self.logger.log_context_event("path_guard", f"DENIED {mode} {resolved}")
        raise PathGuardError(message)

    def check(self, path: Path, mode: Literal["r", "w"]) -> None:
        """Check whether a file access is allowed.

        Args:
            path: The file path to check.
            mode: "r" for read, "w" for write.

        Raises:
            PathGuardError: If the access is denied.
        """
        if not self.enabled:
            return

        resolved = path.resolve()

        # 1. Source dir check BEFORE workspace (source may be inside workspace)
        if resolved.is_relative_to(self.source_dir):
            self._check_source_whitelist(resolved, mode)
            return

        # 2. Workspace -> allow
        if resolved.is_relative_to(self.workspace_dir):
            return

        # 3. Extra writable -> allow r+w
        if self._matches_extra(resolved, self._extra_writable):
            return

        # 4. Extra readable -> allow r only
        if self._matches_extra(resolved, self._extra_readable):
            if mode == "r":
                return
            self._deny(mode, resolved, f"Write denied: {resolved} is read-only")

        # 5. Default deny
        self._deny(
            mode, resolved,
            f"Access denied: {resolved} is outside the workspace and not in the allowed list"
        )

    def _parse_whitelist(self, entries: list[str]) -> list[tuple[Path, str]]:
        """Parse source whitelist entries like 'config/config.yaml:rw'."""
        result = []
        for entry in entries:
            parts = entry.rsplit(":", 1)
            rel_path = parts[0]
            mode = parts[1] if len(parts) > 1 else "r"
            result.append((self.source_dir / rel_path, mode))
        return result

    def _check_source_whitelist(self, resolved: Path, mode: str) -> None:
        """Check if a source path is allowed via whitelist."""
        for wl_path, allowed_mode in self._source_whitelist:
            if resolved == wl_path or resolved.is_relative_to(wl_path):
                if mode == "w" and allowed_mode != "rw":
                    self._deny(mode, resolved, f"Write denied: {resolved} is read-only in source whitelist")
                return
        self._deny(mode, resolved, f"Access denied: {resolved} is agent source code")

    @staticmethod
    def _matches_extra(resolved: Path, extra_paths: list[Path]) -> bool:
        """Check if resolved path falls under any of the extra paths."""
        return any(resolved.is_relative_to(p) for p in extra_paths)

    def audit_command(self, command: str) -> None:
        """Audit a bash command for file access violations.

        Extracts file paths from the command and checks each one.

        Args:
            command: The shell command string to audit.

        Raises:
            PathGuardError: If any extracted path violates the policy.
        """
        if not self.enabled:
            return

        for path, mode in self._extract_paths(command):
            self.check(path, mode)

    def _extract_paths(self, command: str) -> list[tuple[Path, str]]:
        """Extract file paths and their access modes from a shell command.

        Splits on pipes and semicolons, then analyzes each segment.
        Returns a list of (Path, mode) tuples where mode is "r" or "w".
        """
        results: list[tuple[Path, str]] = []

        # Split command on pipes, semicolons, and &&
        segments = re.split(r"\s*[|;]\s*|\s*&&\s*", command)

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            results.extend(self._analyze_segment(segment))

        return results

    def _analyze_segment(self, segment: str) -> list[tuple[Path, str]]:
        """Analyze a single command segment for file paths."""
        results: list[tuple[Path, str]] = []

        # Check for write redirects: handle > and >> before shlex splitting
        redirect_match = self._WRITE_REDIRECTS.split(segment)
        if len(redirect_match) > 1:
            # Everything after the last redirect is a write target
            for part in redirect_match[1:]:
                part = part.strip()
                if part:
                    try:
                        tokens = shlex.split(part)
                    except ValueError:
                        tokens = part.split()
                    for token in tokens:
                        if self._looks_like_path(token):
                            results.append((self._resolve_path(token), "w"))
            # Analyze the part before the first redirect for read paths
            segment = redirect_match[0].strip()
            if not segment:
                return results

        # Parse the segment with shlex
        try:
            tokens = shlex.split(segment)
        except ValueError:
            # Malformed quotes — fallback to simple split
            tokens = segment.split()

        if not tokens:
            return results

        cmd = tokens[0]
        args = tokens[1:]

        # Determine if this is a write command
        is_write_cmd = cmd in self._WRITE_COMMANDS
        has_inplace_flag = bool(self._INPLACE_FLAGS & set(args))

        for token in args:
            if token.startswith("-"):
                continue
            # Skip sed substitution patterns
            if cmd == "sed" and ("/" in token and token.startswith("s")):
                continue
            if not self._looks_like_path(token):
                continue

            path = self._resolve_path(token)

            if is_write_cmd:
                results.append((path, "w"))
            elif has_inplace_flag:
                results.append((path, "w"))
            else:
                results.append((path, "r"))

        return results

    def _resolve_path(self, token: str) -> Path:
        """Resolve a path token, using workspace_dir for relative paths."""
        p = Path(token)
        if not p.is_absolute():
            p = self.workspace_dir / p
        return p.resolve()

    @staticmethod
    def _looks_like_path(token: str) -> bool:
        """Heuristic: does this token look like a file path?"""
        if token.startswith("-"):
            return False
        # Must contain a slash or dot-prefixed, or have a file extension
        if "/" in token or token.startswith("."):
            return True
        if "." in token and not token.replace(".", "").isdigit():
            # Has a dot but isn't a number like "3.14"
            parts = token.rsplit(".", 1)
            if len(parts) == 2 and len(parts[1]) <= 10:
                return True
        return False
