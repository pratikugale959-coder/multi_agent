"""Deterministic import-graph answers for the coordinator fast path.

The language model is useful for open-ended questions, but "what does
``main.py`` import?" and "how does ``main.py`` connect to ``query_pipeline.py``?"
have exact answers in the source. This module builds a small import graph with
Python's ``ast`` module and renders those answers without a model call, so an
obvious structural question still works when the hosted chat endpoint is down.

Only static imports are reported. The renderer never claims that one module
*calls* another at runtime; it says what the import statements prove.
"""

from __future__ import annotations

import ast
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from code_helper.core.config import Settings, get_settings
from code_helper.core.paths import project_root
from code_helper.rag.ingestion import (
    IngestionError,
    discover_source_files,
    read_source_file,
)

#: How far to follow project imports when explaining a connection.
MAX_IMPORT_PATH_DEPTH = 5


@dataclass(frozen=True)
class ImportEdge:
    """One import statement, resolved to a project file when possible."""

    source: str
    target: str
    line: int
    statement: str
    internal: bool
    stdlib: bool


@dataclass(frozen=True)
class ImportGraph:
    """The project's import edges, keyed by source-relative POSIX path."""

    root: Path
    files: tuple[str, ...]
    edges: dict[str, tuple[ImportEdge, ...]]


def _module_index(files: tuple[str, ...]) -> dict[str, str]:
    """Map dotted module names (``pkg.mod``) to relative file paths."""
    index: dict[str, str] = {}
    for relative in files:
        if relative.endswith("/__init__.py"):
            module = relative[: -len("/__init__.py")].replace("/", ".")
        elif relative == "__init__.py":
            module = ""
        elif relative.endswith(".py"):
            module = relative[:-3].replace("/", ".")
        else:
            continue
        if module:
            index[module] = relative
    return index


def _absolute_module(current: str, node: ast.ImportFrom) -> str:
    """Resolve a ``from ... import ...`` module, including relative dots."""
    module = node.module or ""
    if not node.level:
        return module

    parent = Path(current).parent
    parts = [] if parent == Path(".") else list(parent.parts)
    up = node.level - 1
    if up:
        parts = parts[:-up] if up <= len(parts) else []
    if module:
        parts.extend(module.split("."))
    return ".".join(parts)


def _import_candidates(current: str, node: ast.Import | ast.ImportFrom) -> list[str]:
    """Return every dotted module a statement could refer to."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]

    module = _absolute_module(current, node)
    candidates = [module] if module else []
    for alias in node.names:
        if alias.name == "*":
            continue
        candidates.append(f"{module}.{alias.name}" if module else alias.name)
    return list(dict.fromkeys(candidates))


def _is_stdlib(module: str) -> bool:
    """Return True for a Python standard-library module."""
    root = module.split(".", 1)[0]
    return root in sys.stdlib_module_names


def _statement(node: ast.AST) -> str:
    """Render one import statement compactly, with a safe fallback."""
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - ast.unparse is very reliable
        return "<import statement>"


def build_import_graph(settings: Settings | None = None) -> ImportGraph:
    """Parse every project Python file and return its import graph."""
    settings = settings or get_settings()
    root = project_root(settings)
    try:
        paths = discover_source_files(root)
    except IngestionError:
        return ImportGraph(root=root, files=(), edges={})

    files = tuple(sorted(path.relative_to(root).as_posix() for path in paths))
    index = _module_index(files)
    edges: dict[str, set[ImportEdge]] = {}

    for relative in files:
        try:
            source = read_source_file(relative, root=root)
            tree = ast.parse(source.text)
        except (IngestionError, SyntaxError, ValueError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue

            statement = _statement(node)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = index.get(alias.name)
                    edges.setdefault(relative, set()).add(
                        ImportEdge(
                            source=relative,
                            target=target or alias.name,
                            line=node.lineno,
                            statement=statement,
                            internal=target is not None,
                            stdlib=target is None and _is_stdlib(alias.name),
                        )
                    )
                continue

            internal_targets: list[str] = []
            for candidate in _import_candidates(relative, node):
                target = index.get(candidate)
                if target and target != relative and target not in internal_targets:
                    internal_targets.append(target)

            if internal_targets:
                for target in internal_targets:
                    edges.setdefault(relative, set()).add(
                        ImportEdge(
                            source=relative,
                            target=target,
                            line=node.lineno,
                            statement=statement,
                            internal=True,
                            stdlib=False,
                        )
                    )
                continue

            candidates = _import_candidates(relative, node)
            display = _absolute_module(relative, node) or (candidates[0] if candidates else "")
            if display:
                edges.setdefault(relative, set()).add(
                    ImportEdge(
                        source=relative,
                        target=display,
                        line=node.lineno,
                        statement=statement,
                        internal=False,
                        stdlib=_is_stdlib(display),
                    )
                )

    ordered = {
        source: tuple(sorted(items, key=lambda edge: (edge.line, edge.target)))
        for source, items in sorted(edges.items())
    }
    return ImportGraph(root=root, files=files, edges=ordered)


def _resolve_user_file(name: str, graph: ImportGraph) -> str | None:
    """Resolve a user-supplied filename to a path in the import graph."""
    try:
        return read_source_file(name, root=graph.root).relative_path
    except (IngestionError, ValueError, OSError):
        pass

    normalised = name.replace("\\", "/").strip().lstrip("./")
    if normalised in graph.files:
        return normalised
    matches = [
        relative
        for relative in graph.files
        if relative.endswith(f"/{normalised}") or Path(relative).name == normalised
    ]
    return matches[0] if len(matches) == 1 else None


def _internal_targets(graph: ImportGraph, source: str) -> set[str]:
    """Return the project files imported directly by *source*."""
    return {edge.target for edge in graph.edges.get(source, ()) if edge.internal}


def _first_edge(graph: ImportGraph, source: str, target: str) -> ImportEdge | None:
    """Return the first import edge from *source* to *target*."""
    for edge in graph.edges.get(source, ()):
        if edge.internal and edge.target == target:
            return edge
    return None


def _import_path(
    graph: ImportGraph, start: str, goal: str, max_depth: int = MAX_IMPORT_PATH_DEPTH
) -> list[str] | None:
    """Return a shortest project-import path from *start* to *goal*."""
    if start == goal:
        return [start]
    queue: deque[list[str]] = deque([[start]])
    seen = {start}
    while queue:
        path = queue.popleft()
        if len(path) - 1 >= max_depth:
            continue
        for target in sorted(_internal_targets(graph, path[-1])):
            if target in seen:
                continue
            next_path = [*path, target]
            if target == goal:
                return next_path
            seen.add(target)
            queue.append(next_path)
    return None


def _edge_line(edge: ImportEdge | None) -> str:
    if edge is None:
        return "import statement not found"
    return f"line {edge.line}: `{edge.statement}`"


def render_imports(name: str, settings: Settings | None = None) -> str | None:
    """Render what one project file imports, or ``None`` if it is not a file."""
    graph = build_import_graph(settings)
    relative = _resolve_user_file(name, graph)
    if relative is None:
        return None

    edges = graph.edges.get(relative, ())
    internal = [edge for edge in edges if edge.internal]
    stdlib = sorted({edge.target for edge in edges if edge.stdlib})
    third_party = sorted({edge.target for edge in edges if not edge.internal and not edge.stdlib})

    lines = [f"`{relative}` imports (Python AST):"]
    lines.append("- Project modules:")
    if internal:
        lines.extend(f"  - `{edge.target}` ({_edge_line(edge)})" for edge in internal)
    else:
        lines.append("  - (none)")
    lines.append(f"- Standard library: {', '.join(stdlib) or '(none)'}")
    lines.append(f"- Third-party: {', '.join(third_party) or '(none)'}")
    if not edges:
        lines.append("No import statements were found in this file.")
    return "\n".join(lines)


def render_importers(name: str, settings: Settings | None = None) -> str | None:
    """Render which project files import *name*, or ``None`` if not a file."""
    graph = build_import_graph(settings)
    relative = _resolve_user_file(name, graph)
    if relative is None:
        return None

    incoming: list[ImportEdge] = []
    for edges in graph.edges.values():
        incoming.extend(edge for edge in edges if edge.internal and edge.target == relative)
    incoming.sort(key=lambda edge: (edge.source, edge.line))

    if not incoming:
        return (
            f"No Python file in the project imports `{relative}` "
            f"(checked {len(graph.files)} file(s); Python AST)."
        )

    lines = [f"Files that import `{relative}` (Python AST):"]
    lines.extend(f"- `{edge.source}` ({_edge_line(edge)})" for edge in incoming)
    return "\n".join(lines)


def render_relationship(first: str, second: str, settings: Settings | None = None) -> str | None:
    """Render how two project files are connected through imports."""
    graph = build_import_graph(settings)
    source = _resolve_user_file(first, graph)
    target = _resolve_user_file(second, graph)
    if source is None or target is None:
        return None

    path = _import_path(graph, source, target)
    reverse = False
    if path is None:
        path = _import_path(graph, target, source)
        reverse = True

    lines = [f"Import relationship between `{source}` and `{target}` (Python AST):"]
    if path:
        if reverse:
            lines.append(
                f"- `{source}` is imported by `{target}` along this path: "
                + " -> ".join(f"`{item}`" for item in path)
            )
        else:
            lines.append("- Import path: " + " -> ".join(f"`{item}`" for item in path))
        for start, end in zip(path, path[1:], strict=False):
            lines.append(f"  - `{start}` {_edge_line(_first_edge(graph, start, end))}")
    else:
        lines.append(f"- No import path between `{source}` and `{target}` was found.")
        common = sorted(_internal_targets(graph, source) & _internal_targets(graph, target))
        if common:
            lines.append("- Both import: " + ", ".join(f"`{item}`" for item in common))

    lines.append(
        "This is static import evidence only; it does not prove runtime calls between the modules."
    )
    return "\n".join(lines)
