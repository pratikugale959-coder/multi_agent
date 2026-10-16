"""Path-safety helpers shared by the deterministic filesystem tools.

Every tool that accepts a path must funnel it through :func:`resolve_in_project`.
That single choke point is what prevents path traversal (``../../etc/passwd``)
and keeps the tools inside the configured target project.
"""

from __future__ import annotations

from pathlib import Path

from code_helper.config import Settings, get_settings


class PathSecurityError(ValueError):
    """Raised when a requested path escapes the allowed project root."""


def project_root(settings: Settings | None = None) -> Path:
    """Return the resolved absolute target-project root."""
    root = (settings or get_settings()).target_repo_path
    return Path(root).expanduser().resolve()


def is_within(path: Path, root: Path) -> bool:
    """Return True when *path* is inside *root*."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_in_project(file_path: str | Path, settings: Settings | None = None) -> Path:
    """Resolve *file_path* inside the target project, rejecting escapes.

    Relative paths are interpreted relative to the project root.

    Args:
        file_path: Relative or absolute path supplied by a user/agent.
        settings: Optional configuration override.

    Returns:
        The resolved absolute path.

    Raises:
        PathSecurityError: If the path escapes the project root, or if the path
            is empty.
        FileNotFoundError: If the project root does not exist.
    """
    if file_path is None or str(file_path).strip() == "":
        raise PathSecurityError("A non-empty file path is required")

    root = project_root(settings)
    if not root.exists():
        raise FileNotFoundError(f"Target project root does not exist: {root}")

    candidate = Path(str(file_path).strip())
    if not candidate.is_absolute():
        candidate = root / candidate

    resolved = candidate.resolve()
    if not is_within(resolved, root):
        raise PathSecurityError(
            f"Refusing to access '{file_path}': it is outside the allowed project root ({root})."
        )
    return resolved


def relative_path(file_path: str | Path, settings: Settings | None = None) -> str:
    """Return the POSIX relative path of *file_path* within the project root."""
    root = project_root(settings)
    resolved = resolve_in_project(file_path, settings)
    return resolved.relative_to(root).as_posix()
