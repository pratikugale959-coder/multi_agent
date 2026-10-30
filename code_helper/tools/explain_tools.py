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

import ast
from typing import Any

from code_helper.core.config import get_settings
from code_helper.core.paths import PathSecurityError, project_root
from code_helper.rag.ingestion import IngestionError, read_source_file

#: Bound the prompt size even when the ingestion layer accepts a large file.
MAX_READ_CHARS = 40_000


def _first_line(text: str | None) -> str:
    """Return a compact one-line summary of a docstring."""
    if not text:
        return ""
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line[:160]


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render a readable function signature without defaults or annotations."""
    args: list[str] = []
    for arg in node.args.posonlyargs + node.args.args:
        if arg.arg not in {"self", "cls"}:
            args.append(arg.arg)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    elif node.args.kwonlyargs:
        args.append("*")
    args.extend(arg.arg for arg in node.args.kwonlyargs)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return f"{prefix}{node.name}({', '.join(args)})"


def _value_preview(node: ast.AST) -> str:
    """Render a short, safe preview of a constant's value."""
    try:
        value = ast.unparse(node)
    except Exception:  # pragma: no cover - defensive for unusual AST nodes
        return "..."
    value = " ".join(value.split())
    return value[:80] + ("..." if len(value) > 80 else "")


def summarize_project_file(file_path: str) -> str:
    """Return a compact AST outline of a Python file.

    Use this first when explaining a whole file. It lists the module docstring,
    imports, module constants, classes, methods and functions with their
    docstrings and line numbers. The outline is much smaller than the source, so
    it is faster and avoids sending a whole file to the chat model. Call
    ``read_project_file`` only when the outline is insufficient or the user asks
    for line-level detail.

    Args:
        file_path: Path relative to the project root (e.g. ``"main.py"``) or an
            absolute path that is still inside the project root.

    Returns:
        A text outline, or an explanatory error string on failure.
    """
    if not isinstance(file_path, str) or file_path.strip() == "":
        return "Error: summarize_project_file requires a non-empty file path."

    settings = get_settings()
    root = project_root(settings)
    try:
        source = read_source_file(file_path, root=root)
    except PathSecurityError as exc:
        return f"Error: {exc}"
    except IngestionError as exc:
        return f"Error: {exc}"

    return _outline_for_source(source)


def _outline_for_source(source: Any) -> str:
    """Format an already-resolved source file as a compact AST outline."""
    header = f"File: {source.relative_path} ({source.line_count} lines)"
    try:
        tree = ast.parse(source.text)
    except SyntaxError as exc:
        preview = "\n".join(source.text.splitlines()[:40])
        return (
            f"{header}\n(Not valid Python: {exc.msg} at line {exc.lineno}; "
            f"showing the first 40 lines)\n{preview}"
        )

    lines = [header]
    module_doc = _first_line(ast.get_docstring(tree))
    if module_doc:
        lines.append(f"Module: {module_doc}")

    imports: list[str] = []
    constants: list[str] = []
    entities: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            names = ", ".join(alias.name for alias in node.names)
            imports.append(f"import {names}")
        elif isinstance(node, ast.ImportFrom):
            names = ", ".join(alias.name for alias in node.names)
            where = "." * node.level + (node.module or "")
            imports.append(f"from {where} import {names}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = ", ".join(target.id for target in targets if isinstance(target, ast.Name))
            if names and node.value is not None:
                constants.append(f"{names} = {_value_preview(node.value)}")
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(_value_preview(base) for base in node.bases)
            suffix = f"({bases})" if bases else ""
            doc = _first_line(ast.get_docstring(node))
            entities.append(
                f"class {node.name}{suffix} (line {node.lineno})" + (f" - {doc}" if doc else "")
            )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_doc = _first_line(ast.get_docstring(child))
                    entities.append(
                        f"  {_signature(child)} (line {child.lineno})"
                        + (f" - {method_doc}" if method_doc else "")
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = _first_line(ast.get_docstring(node))
            entities.append(
                f"{_signature(node)} (line {node.lineno})" + (f" - {doc}" if doc else "")
            )

    if imports:
        lines.append("Imports:")
        lines.extend(f"  {line}" for line in imports)
    if constants:
        lines.append("Constants:")
        lines.extend(f"  {line}" for line in constants)
    if entities:
        lines.append("Definitions:")
        lines.extend(entities)

    if not imports and not constants and not entities:
        lines.append("(No top-level imports, constants, classes or functions found.)")

    return "\n".join(lines)


def read_project_file(file_path: str) -> str:
    """Read a source file inside the target project.

    Use this to explain a whole file: its imports, functions, classes and control
    flow. The path is validated against the project root, so path traversal and
    secret files are rejected. Very large files are truncated so they cannot
    overflow the model's prompt; use ``summarize_project_file`` for a compact
    outline first.

    Args:
        file_path: Path relative to the project root (e.g. ``"query_pipeline.py"``)
            or an absolute path that is still inside the project root.

    Returns:
        The file contents (capped at :data:`MAX_READ_CHARS`), prefixed with a
        one-line header giving the relative path and line count. Returns an
        explanatory error string on failure.
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
    if len(source.text) <= MAX_READ_CHARS:
        return header + source.text
    return (
        header
        + source.text[:MAX_READ_CHARS]
        + f"\n\n[Truncated after {MAX_READ_CHARS} characters. "
        "Use summarize_project_file for a compact outline of the rest.]"
    )
