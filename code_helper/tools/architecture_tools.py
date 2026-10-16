"""Deterministic tools for the **Architecture Mapper** agent.

Both tools use plain Python (``pathlib`` and the ``ast`` module) and never call an
LLM, so their output is reproducible and cheap. Parsing imports with ``ast`` is
far more reliable than asking a model to guess dependencies.
"""

from __future__ import annotations

import ast

from code_helper.config import get_settings
from code_helper.ingestion import (
    DEFAULT_EXTENSIONS,
    IngestionError,
    discover_source_files,
    read_source_file,
)
from code_helper.paths import PathSecurityError, project_root


def list_project_files() -> str:
    """List the Python source files that make up the target project.

    Use this to see the project's file structure before analysing dependencies or
    entry points. Hidden directories, virtual environments, caches, secrets and
    generated files are excluded.

    Returns:
        A formatted list of relative file paths with line counts, or an error
        message when the project root is missing.
    """
    root = project_root(get_settings())
    if not root.exists():
        return f"Error: target project root does not exist: {root}"

    paths = discover_source_files(root, extensions=DEFAULT_EXTENSIONS)
    if not paths:
        return f"No Python source files found under {root}."

    lines = [f"Project root: {root}", f"Python files ({len(paths)}):"]
    for path in paths:
        relative = path.relative_to(root).as_posix()
        try:
            line_count = len(path.read_text(encoding="utf-8").splitlines())
        except (OSError, UnicodeDecodeError):
            line_count = -1
        suffix = f" ({line_count} lines)" if line_count >= 0 else " (unreadable)"
        lines.append(f"  - {relative}{suffix}")
    return "\n".join(lines)


def _relative_import(node: ast.ImportFrom) -> str:
    """Render a ``from ... import ...`` target, preserving relative dots."""
    module = node.module or ""
    prefix = "." * (node.level or 0)
    return f"{prefix}{module}"


def inspect_imports(file_path: str) -> str:
    """Extract imports and top-level definitions from a Python file using ``ast``.

    Use this to understand how a module connects to the rest of the project.

    Args:
        file_path: Path relative to the project root (or absolute inside it).

    Returns:
        A formatted report of imports, classes and functions. Invalid syntax,
        unreadable files and path escapes are reported as text, not exceptions.
    """
    if not isinstance(file_path, str) or file_path.strip() == "":
        return "Error: inspect_imports requires a non-empty file path."

    root = project_root(get_settings())
    try:
        source = read_source_file(file_path, root=root)
    except PathSecurityError as exc:
        return f"Error: {exc}"
    except IngestionError as exc:
        return f"Error: {exc}"

    try:
        tree = ast.parse(source.text)
    except SyntaxError as exc:
        return (
            f"Error: could not parse {source.relative_path} -- invalid Python syntax "
            f"at line {exc.lineno}: {exc.msg}"
        )

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
                )
        elif isinstance(node, ast.ImportFrom):
            names = ", ".join(alias.name for alias in node.names)
            imports.append(f"from {_relative_import(node)} import {names}")
    # The same import can appear in several functions; report each unique import once.
    imports = list(dict.fromkeys(imports))

    classes: list[str] = []
    functions: list[str] = []
    for node in tree.body:  # top-level only: keeps the report readable
        if isinstance(node, ast.ClassDef):
            classes.append(f"class {node.name} (line {node.lineno})")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            functions.append(f"{kind} {node.name} (line {node.lineno})")

    docstring = ast.get_docstring(tree)

    lines = [f"File: {source.relative_path}"]
    if docstring:
        first_line = docstring.strip().splitlines()[0]
        lines.append(f"Module docstring: {first_line}")

    def append_section(title: str, items: list[str]) -> None:
        lines.append(f"{title} ({len(items)}):")
        if items:
            lines.extend(f"  - {item}" for item in items)
        else:
            lines.append("  - (none)")

    append_section("Imports", imports)
    append_section("Top-level classes", classes)
    append_section("Top-level functions", functions)
    return "\n".join(lines)


def collect_definitions(source_text: str) -> dict[str, list[str]]:
    """Return top-level class and function names from *source_text*.

    A small pure function kept separate so it can be unit-tested without files.
    """
    result: dict[str, list[str]] = {"classes": [], "functions": []}
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return result
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            result["classes"].append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result["functions"].append(node.name)
    return result
