"""PathGuard: Agent file access control."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from mini_agent.config import PathGuardConfig


class PathGuardError(PermissionError):
    """Raised when a file access violates PathGuard policy."""

    pass


class PathGuard:
    """File access control that restricts agent operations to allowed paths."""

    def __init__(self, config: PathGuardConfig, workspace_dir: Path, source_dir: Path):
        self.enabled = config.enabled
        self.workspace_dir = workspace_dir.resolve()
        self.source_dir = source_dir.resolve()
        self._extra_readable = [Path(p).expanduser().resolve() for p in config.extra_readable_paths]
        self._extra_writable = [Path(p).expanduser().resolve() for p in config.extra_writable_paths]
        self._source_whitelist = self._parse_whitelist(config.source_whitelist)

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
            raise PathGuardError(f"Write denied: {resolved} is read-only")

        # 5. Default deny
        raise PathGuardError(
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
                    raise PathGuardError(
                        f"Write denied: {resolved} is read-only in source whitelist"
                    )
                return
        raise PathGuardError(f"Access denied: {resolved} is agent source code")

    @staticmethod
    def _matches_extra(resolved: Path, extra_paths: list[Path]) -> bool:
        """Check if resolved path falls under any of the extra paths."""
        return any(resolved.is_relative_to(p) for p in extra_paths)
