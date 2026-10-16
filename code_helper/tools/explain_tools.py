"""Deterministic tool for the **Code Explainer** agent.

``read_project_file`` reads a *complete* source file safely. This works without
any AI (satisfying the "at least one tool must operate without AI" requirement)
and complements the read-only Filesystem MCP toolset that the Code Explainer also
has access to.

The public signature is deliberately minimal -- only ``file_path`` -- so Google
ADK never asks the model to invent internal arguments such as configuration
objects.
"""

from __future__ import annotations

from code_helper.config import get_settings
from code_helper.ingestion import IngestionError, read_source_file
from code_helper.paths import PathSecurityError, project_root


def read_project_file(file_path: str) -> str:
    """Read the full contents of a source file inside the target project.

    Use this to explain a whole file: its imports, functions, classes and control
    flow. The path is validated against the project root, so path traversal and
    secret files are rejected.

    Args:
        file_path: Path relative to the project root (e.g. ``"query_pipeline.py"``)
            or an absolute path that is still inside the project root.

    Returns:
        The file contents, prefixed with a one-line header giving the relative
        path and line count. Returns an explanatory error string on failure.
    """
    if not isinstance(file_path, str) or file_path.strip() == "":
        return "Error: read_project_file requires a non-empty file path."

    settings = get_settings()
    root = project_root(settings)
    try:
        source = read_source_file(file_path, root=root)
    except PathSecurityError as exc:
        return f"Error: {exc}"
    except IngestionError as exc:
        return f"Error: {exc}"

    header = f"File: {source.relative_path} ({source.line_count} lines)\n{'-' * 60}\n"
    return header + source.text
