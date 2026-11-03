"""Deterministic answers for narrow, high-frequency questions.

The language model is excellent for open-ended questions, but four question
shapes have a fully deterministic answer already available in the repository:

* ``"Where is main.py?"`` -- locate a literally named file.
* ``"What does query_pipeline.py do?"`` -- show the AST outline of that file.
* ``"What does main.py import?"`` / ``"How does main.py connect to
  query_pipeline.py?"`` -- describe real imports from the AST graph.
* ``"Tell me about this codebase in 5 lines"`` -- a short evidence-based overview.
* ``"Explain how the coordinator agent works"`` -- explain the indexed scope and
  point at the helper's own source, which is outside that scope by default.

Answering these without a model call makes the agent usable even when the hosted
chat endpoint is slow or unavailable. Everything else still goes through the
coordinator and the specialised workers.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from google.genai import types

from code_helper.core.config import Settings, get_settings
from code_helper.core.paths import project_root
from code_helper.rag.ingestion import IngestionError, read_source_file
from code_helper.tools.explain_tools import _outline_for_source
from code_helper.tools.import_analysis import (
    render_importers,
    render_imports,
    render_relationship,
)
from code_helper.tools.onboarding_tools import ProjectOverview, collect_project_overview
from code_helper.tools.search_tools import _named_files
from code_helper.tools.symbol_locator import locate_symbol

_WHERE_RE = re.compile(r"^(where\s+is|where\s+are|where's|locate|find)\b", re.IGNORECASE)
_EXPLAIN_RE = re.compile(r"^(what\s+does|what\s+is|explain|describe|summari[sz]e)\b", re.IGNORECASE)
_ALLOWED_WORDS = frozenset(
    {
        "about",
        "code",
        "describe",
        "do",
        "does",
        "explain",
        "file",
        "find",
        "in",
        "is",
        "locate",
        "located",
        "me",
        "module",
        "please",
        "project",
        "source",
        "summarise",
        "summarize",
        "tell",
        "the",
        "this",
        "what",
        "where",
    }
)

#: Names that identify the helper itself. Its source is not in the indexed
#: target repository by default, so routing these to RAG only produces timeouts
#: or a misleading "which file?" question.
_SCOPE_RE = re.compile(
    r"\b(?:coordinator(?:\s+agent)?|code_helper|code\s+helper|"
    r"multi[- ]agent\s+(?:helper|system)|root_agent|code_searcher|code_explainer|"
    r"architecture_mapper|onboarding_guide|this\s+assistant|this\s+helper|"
    r"your\s+(?:own\s+)?(?:code|source|codebase|implementation))\b",
    re.IGNORECASE,
)
_SCOPE_QUERY_RE = re.compile(
    r"\bwhat\s+(?:is|are)\s+(?:being\s+)?indexed\b"
    r"|\bwhat\s+(?:files|repos?|repositories|projects?)\s+(?:are|is)\s+(?:being\s+)?indexed\b"
    r"|\bwhat\s+do\s+you\s+index\b"
    r"|\bwhat\s+is\s+(?:in\s+)?scope\b"
    r"|\bwhich\s+(?:repo|repository|project)\s+(?:is|are)\s+(?:indexed|you)\b",
    re.IGNORECASE,
)

#: Project-level nouns plus an overview/usage verb. A query such as
#: "where is this project configured?" must still go to the code searcher.
_OVERVIEW_TARGET_RE = re.compile(
    r"\b(?:this|the)\s+(?:project|codebase|repo|repository|code)\b"
    r"|\b(?:project|codebase|repo|repository)\s+overview\b"
    r"|\bwhat'?s\s+in\s+(?:this|the)\s+(?:repo|repository|project|codebase)\b",
    re.IGNORECASE,
)
_OVERVIEW_INTENT_RE = re.compile(
    r"^(?:what\s+(?:does|is|are)|explain|describe|summari[sz]e|tell\s+me|give\s+me|"
    r"show\s+me|overview|where\s+do\s+i\s+start|getting\s+started|"
    r"how\s+(?:do|can|should)\s+i\s+(?:run|start)|what'?s\s+in)\b",
    re.IGNORECASE,
)
_BARE_RUN_RE = re.compile(
    r"^(?:how\s+(?:do|can|should)\s+i\s+(?:run|start)\s+(?:it|this)"
    r"|where\s+do\s+i\s+start|getting\s+started)\b",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    """Normalise backticks and quotes around a filename."""
    return re.sub(r"[`'\"]", "", text).strip()


def _module_purpose(source: Any) -> str:
    """Return a one-line purpose for a file, using real source only."""
    try:
        tree = ast.parse(source.text)
    except SyntaxError:
        return f"text file ({source.line_count} lines)"

    doc = ast.get_docstring(tree)
    if doc:
        purpose = " ".join(doc.strip().splitlines()[0].split())
    else:
        purpose = ""
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                node_doc = ast.get_docstring(node)
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                purpose = f"defines the {kind} `{node.name}`"
                if node_doc:
                    purpose += f" ({' '.join(node_doc.strip().splitlines()[0].split())})"
                break

    if not purpose:
        for line in source.text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") and not stripped.startswith("#!"):
                purpose = stripped.lstrip("# ").strip()
                break

    if not purpose:
        purpose = f"Python module ({source.line_count} lines)"

    # Drop a redundant "main.py -- " prefix from module docstrings.
    name = Path(source.relative_path).name
    purpose = re.sub(rf"^{re.escape(name)}\s*(?:--|—|-)\s*", "", purpose)
    return purpose


def _named_file_is_readable(name: str, settings: Settings) -> bool:
    """Return True when *name* can be read inside the configured project."""
    try:
        read_source_file(name, root=project_root(settings))
    except (IngestionError, ValueError):
        return False
    return True


def _is_overview_question(text: str) -> bool:
    """Return True for project-level "what/run/overview" questions."""
    if _BARE_RUN_RE.match(text):
        return True
    return bool(_OVERVIEW_TARGET_RE.search(text) and _OVERVIEW_INTENT_RE.match(text))


#: Relation words that mark an import/connection question rather than a pure
#: "where is this file?" lookup.
_RELATION_RE = re.compile(
    r"\b(?:imports?|imported|uses?|used|depends?|dependencies|connect(?:s|ed|ion)?|"
    r"relat(?:e|es|ed|ionship)?|calls?|called|references?|between)\b",
    re.IGNORECASE,
)
_INCOMING_IMPORT_RE = re.compile(
    r"\b(?:what|which)\s+(?:(?:files?|modules?)\s+)?imports?\b"
    r"|\bwho\s+imports\b"
    r"|\bwhere\s+is\b.+\bused\b"
    r"|\bwho\s+uses\b"
    r"|\bimported\s+by\b",
    re.IGNORECASE,
)
_OUTGOING_IMPORT_RE = re.compile(
    r"\b(?:what|which)\s+(?:files?|modules?)\s+does\b.+\bimport\b"
    r"|\bwhat\s+does\b.+\bimport\b"
    r"|\bimports?\s+of\b"
    r"|\bdepends?\s+on\b"
    r"|\bdependenc(?:y|ies)\b"
    r"|\bimports?\b",
    re.IGNORECASE,
)


def _relation_answer(text: str, names: list[str], settings: Settings) -> str | None:
    """Answer import/connection questions from the AST import graph.

    Two named files describe a connection; one named file answers either
    "what does it import?" or "what imports it?". Anything less specific is left
    to the worker agents so the fast path never guesses at intent.
    """
    if not names:
        return None

    if len(names) >= 2:
        if not _RELATION_RE.search(text):
            return None
        return render_relationship(names[0], names[1], settings)

    # Incoming direction is checked first: "what imports X?" also contains the
    # word "imports", but means the opposite of "what does X import?".
    if _INCOMING_IMPORT_RE.search(text):
        return render_importers(names[0], settings)
    if _OUTGOING_IMPORT_RE.search(text):
        return render_imports(names[0], settings)
    return None


def _first_sentence(text: str) -> str:
    """Return the first sentence of *text* (used to keep the summary short)."""
    collapsed = " ".join(text.split())
    match = re.match(r"(.+?[.!?])(?:\s|$)", collapsed)
    return match.group(1) if match else collapsed


def _run_summary(overview: ProjectOverview) -> tuple[str, str]:
    """Pick the most useful documented (or inferred) run commands."""
    documented = list(overview.documented_commands)
    if documented:
        chosen: list[str] = []
        for pattern in (r"^(?:pip|pip3)\s+install\b", r"^(?:python|py|npx|adk|uvicorn|docker)\b"):
            for command in documented:
                if command not in chosen and re.match(pattern, command, re.IGNORECASE):
                    chosen.append(command)
                    break
        commands = chosen or documented[:2]
        return ", ".join(f"`{command}`" for command in commands), "documented in README.md"
    if overview.inferred_commands:
        return f"`{overview.inferred_commands[0]}`", "inferred from an __main__ guard"
    return "not documented", ""


def _overview_answer(overview: ProjectOverview) -> str:
    """Render a short, evidence-only project summary (about five lines)."""
    name = overview.root.name or str(overview.root)
    if overview.readme_found and overview.purpose:
        purpose = _first_sentence(overview.purpose)
    else:
        purpose = "purpose is not documented (no README.md found)"

    files = list(overview.files)
    shown = ", ".join(files[:8])
    if len(files) > 8:
        shown += f", and {len(files) - 8} more"

    entry_points = list(
        dict.fromkeys([*overview.named_entry_points, *overview.guarded_entry_points])
    )
    dependencies = ", ".join(overview.dependencies) or "none documented"
    commands, command_source = _run_summary(overview)
    run_line = f"- Run: {commands}" + (f" ({command_source})." if command_source else ".")
    reading = ", ".join(overview.reading_order[:3]) or "no files to read"

    return "\n".join(
        [
            f"`{name}/` — {purpose}",
            f"- Python files ({len(files)}): {shown or '(none)'}.",
            f"- Entry point(s): {', '.join(entry_points) or 'none detected'}; "
            f"dependencies: {dependencies}.",
            run_line,
            f"- Start reading: {reading}.",
        ]
    )


def _scope_answer(settings: Settings) -> str:
    """Explain what is indexed and where the helper's own source lives."""
    target = project_root(settings)
    return (
        f"I answer questions about the indexed repository, `{target.name}/` "
        f"({target}), not about my own source.\n\n"
        "The coordinator agent is defined in `code_helper/agents/coordinator.py`: it "
        "has no tools of its own and delegates each request to one of four workers "
        "(`code_searcher`, `code_explainer`, `architecture_mapper`, "
        "`onboarding_guide`). That file is outside the indexed scope, so I cannot "
        "quote it here.\n\n"
        "To make the helper itself searchable, set `TARGET_REPO_PATH=./` in `.env`, "
        "run `python -m code_helper.index_codebase --prune --progress`, and restart "
        "`adk run code_helper` (or `adk web`)."
    )


def deterministic_answer(text: str, settings: Settings | None = None) -> str | None:
    """Return a deterministic answer for a narrow query, or ``None``.

    The match is intentionally strict. Scope questions and project-level
    overview/setup questions are handled first; filename questions must contain
    exactly one filename and start with a location or explanation verb. Mixed
    questions (for example "how does main.py use config.py?") are left to the
    workers.
    """
    cleaned = _clean_text(text)
    settings = settings or get_settings()
    names = _named_files(cleaned)

    # Questions about the helper itself: its source is not in the indexed target
    # repository by default, so answering from RAG would be a guess or a timeout.
    # If the user names a file that really is readable (for example after setting
    # TARGET_REPO_PATH=./), let the normal filename path answer instead.
    if (_SCOPE_RE.search(cleaned) or _SCOPE_QUERY_RE.search(cleaned)) and not any(
        _named_file_is_readable(name, settings) for name in names
    ):
        return _scope_answer(settings)

    # Import and connection questions about named files are answered from the
    # AST graph. This covers "what does main.py import?", "where is alpha.py
    # used?" and "how does main.py connect to query_pipeline.py?" without a model
    # call, so an obvious relationship never depends on a slow endpoint.
    relation = _relation_answer(cleaned, names, settings)
    if relation is not None:
        return relation

    # A "where is ...?" question without a filename is answered from real
    # function/class definitions when the match is unambiguous. This prevents
    # the coordinator from asking "which document?" for an obvious operation
    # such as "Where is the document loaded?".
    if not names and _WHERE_RE.match(cleaned):
        located = locate_symbol(cleaned, settings)
        if located is not None:
            return located

    # Project-level overview/setup questions are answered from the same
    # evidence-based overview the onboarding worker uses, without a model call.
    if not names and _is_overview_question(cleaned):
        try:
            overview = collect_project_overview(settings)
        except FileNotFoundError as exc:
            return f"Error: {exc}"
        return _overview_answer(overview)

    if len(names) != 1:
        return None
    if not (_WHERE_RE.match(cleaned) or _EXPLAIN_RE.match(cleaned)):
        return None

    # Reject mixed questions such as "where is main.py used?" -- the fast path
    # only answers pure location/summary questions.
    name = names[0]
    remainder = cleaned.replace(name, " ")
    if any(
        token not in _ALLOWED_WORDS for token in re.findall(r"[a-z_][a-z0-9_]+", remainder.lower())
    ):
        return None

    if _WHERE_RE.match(cleaned):
        try:
            source = read_source_file(name, root=project_root(settings))
        except (IngestionError, ValueError) as exc:
            return (
                f"Could not read `{name}` under the project root: {exc}. "
                "The file was not found; do not infer its contents from imports."
            )
        purpose = _module_purpose(source)
        return (
            f"`{source.relative_path}` — {purpose}\n\n"
            f"Path: `{source.relative_path}` ({source.line_count} lines)."
        )

    try:
        source = read_source_file(name, root=project_root(settings))
    except (IngestionError, ValueError) as exc:
        return f"Could not read `{name}` under the project root: {exc}."
    return _outline_for_source(source)


def fast_path_callback(callback_context: Any) -> types.Content | None:
    """ADK ``before_agent_callback`` that answers narrow queries directly."""
    content = getattr(callback_context, "user_content", None)
    parts = getattr(content, "parts", None) or []
    text = "\n".join(getattr(part, "text", "") or "" for part in parts).strip()
    if not text:
        return None

    answer = deterministic_answer(text)
    if answer is None:
        return None
    return types.Content(role="model", parts=[types.Part(text=answer)])
