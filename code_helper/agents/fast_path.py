"""Deterministic answers for narrow, high-frequency questions.

The language model is excellent for open-ended questions, but two question
shapes have a fully deterministic answer already available in the repository:

* ``"Where is main.py?"`` -- locate a literally named file.
* ``"What does query_pipeline.py do?"`` -- show the AST outline of that file.

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
from code_helper.tools.search_tools import _named_files

_WHERE_RE = re.compile(r"^(where\s+is|where's|locate|find)\b", re.IGNORECASE)
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


def deterministic_answer(text: str, settings: Settings | None = None) -> str | None:
    """Return a deterministic answer for a narrow query, or ``None``.

    The match is intentionally strict: the query must contain exactly one
    filename and start with a location or explanation verb. Mixed questions
    (for example "how does main.py use config.py?") are left to the workers.
    """
    cleaned = _clean_text(text)
    names = _named_files(cleaned)
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

    settings = settings or get_settings()

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
