"""Deterministic locator for obvious "where is ...?" questions.

The coordinator model occasionally treats a question such as "Where is the
document loaded?" as ambiguous and asks "which document?" instead of routing it
to a search. That is a bad answer for a small project where the relevant
function is easy to find. This module locates the operation with a conservative
lexical search over real Python definitions (functions and classes) and their
docstrings, then reports the file and line range without any model call.

It is intentionally narrow:

* It only handles questions with at least one meaningful term that appears in a
  real definition.
* Two-term questions must match at least two terms.
* A single-term question must match the definition's *name* (for example
  "upsert" -> ``upsert_to_vector_db``).
* If the best match is tied across files, the answer is left to the workers.

Every answer cites the source file and line range. Nothing is inferred from a
file's name alone.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from code_helper.core.config import Settings, get_settings
from code_helper.core.paths import project_root
from code_helper.rag.ingestion import (
    IngestionError,
    discover_source_files,
    read_source_file,
)
from code_helper.tools.search_tools import _STOPWORDS

#: Query words that add no locating value on their own.
_EXTRA_STOPWORDS = frozenset(
    {
        "actually",
        "anywhere",
        "contained",
        "defined",
        "definition",
        "exactly",
        "happen",
        "happens",
        "inside",
        "located",
        "operation",
        "performed",
        "please",
        "specifically",
        "stored",
        "stuff",
        "thing",
    }
)


@dataclass(frozen=True)
class SymbolMatch:
    """One real function or class that matches the question's terms."""

    file_path: str
    name: str
    kind: str
    line_start: int
    line_end: int
    docstring: str
    matched_terms: tuple[str, ...]
    name_terms: tuple[str, ...]
    score: float


def _ordered_terms(query: str) -> list[str]:
    """Return content words in question order, de-duplicated.

    Order matters for the small ranking heuristic below: in "where are the
    document chunks embedded?" the final term ("embedded") names the operation,
    while "document" and "chunks" describe its input.
    """
    return list(
        dict.fromkeys(
            token
            for token in re.findall(r"[a-z_][a-z0-9_]{2,}", query.lower())
            if token not in _STOPWORDS and token not in _EXTRA_STOPWORDS
        )
    )


def _meaningful_terms(query: str) -> set[str]:
    """Return content words from *query*, excluding conversational filler."""
    return set(_ordered_terms(query))


def _term_variants(term: str) -> tuple[str, ...]:
    """Return simple plural/past/gerund variants so "loaded" matches "load"."""
    variants = {term}
    if len(term) >= 5:
        if term.endswith("ied"):
            variants.add(f"{term[:-3]}y")
        if term.endswith("ed"):
            base = term[:-2]
            variants.add(base)
            variants.add(term[:-1])
            if len(base) >= 4 and base[-1] == base[-2]:
                variants.add(base[:-1])
        if term.endswith("ing"):
            variants.add(term[:-3])
        if term.endswith("es"):
            variants.add(term[:-2])
        if term.endswith("s"):
            variants.add(term[:-1])
    return tuple(value for value in variants if len(value) >= 3)


def _mentions(text: str, term: str) -> bool:
    """Return True when *text* contains *term* or a common variant."""
    return any(variant in text for variant in _term_variants(term))


def _first_doc_line(node: ast.AST) -> str:
    """Return the first docstring line for a definition, or ``""``."""
    doc = ast.get_docstring(node)  # type: ignore[arg-type]
    if not doc:
        return ""
    return " ".join(doc.strip().splitlines()[0].split())


def _candidates(
    source_text: str, relative_path: str, terms: set[str], last_term: str | None
) -> list[SymbolMatch]:
    """Score every function/class in one file against the query terms."""
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return []

    matches: list[SymbolMatch] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            kind = "class"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "function"
        else:
            continue

        docstring = _first_doc_line(node)
        segment = ast.get_source_segment(source_text, node) or ""
        surface = f"{node.name}\n{docstring}\n{segment}".lower()
        name_surface = node.name.lower()

        matched = {term for term in terms if _mentions(surface, term)}
        name_matched = {term for term in matched if _mentions(name_surface, term)}
        if not matched:
            continue

        if len(terms) == 1 and not name_matched:
            continue
        if len(terms) >= 2 and len(matched) < 2:
            continue

        # A definition whose *name* contains the operation is a stronger answer
        # than one that only mentions the words in its body. Matching the final
        # query term gets an extra point because "where are X embedded?" asks for
        # ``embed_chunks``, not a caller such as ``run_indexing_pipeline``.
        score = len(matched) + (0.5 * len(name_matched))
        if last_term is not None and last_term in name_matched:
            score += 1.0

        matches.append(
            SymbolMatch(
                file_path=relative_path,
                name=node.name,
                kind=kind,
                line_start=node.lineno,
                line_end=getattr(node, "end_lineno", None) or node.lineno,
                docstring=docstring,
                matched_terms=tuple(sorted(matched)),
                name_terms=tuple(sorted(name_matched)),
                score=score,
            )
        )
    return matches


def locate_symbol(query: str, settings: Settings | None = None) -> str | None:
    """Return a grounded file/function answer for an obvious location question.

    ``None`` means the question is not specific enough to answer this way; the
    normal coordinator/worker path then handles it.
    """
    ordered_terms = _ordered_terms(query)
    terms = set(ordered_terms)
    if not terms:
        return None

    settings = settings or get_settings()
    root = project_root(settings)
    try:
        paths = discover_source_files(root)
    except IngestionError:
        return None

    candidates: list[SymbolMatch] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        try:
            source = read_source_file(relative, root=root)
        except (IngestionError, ValueError):
            continue
        candidates.extend(_candidates(source.text, relative, terms, ordered_terms[-1]))

    if not candidates:
        return None

    # Rank by the strongest kind of evidence, not by a blended score:
    #   1. the definition's *name* mentions the question's final term, which in
    #      "where are document chunks embedded?" is the operation ("embedded");
    #   2. how many question terms the definition mentions at all;
    #   3. how many of those terms appear in its name -- but only when some
    #      definition actually names the operation. Otherwise a helper such as
    #      ``clear_local_store`` would win on the incidental word "local" even
    #      though ``upsert_to_vector_db`` and ``query_vector_db`` are the real
    #      users of the fallback.
    # Definitions tied on all three are alternative answers and are reported
    # together. A tie across files is ambiguous, so the workers handle it.
    last_term = ordered_terms[-1]
    operation_is_named = any(last_term in candidate.name_terms for candidate in candidates)

    def rank(candidate: SymbolMatch) -> tuple[bool, int, int]:
        return (
            last_term in candidate.name_terms,
            len(candidate.matched_terms),
            len(candidate.name_terms) if operation_is_named else 0,
        )

    top_rank = max(rank(candidate) for candidate in candidates)
    winners = [candidate for candidate in candidates if rank(candidate) == top_rank]
    winner_files = {candidate.file_path for candidate in winners}
    if len(winner_files) != 1:
        return None

    winners.sort(key=lambda candidate: (candidate.line_start, candidate.name))
    file_path = winners[0].file_path
    lines = [f"`{file_path}` matches this question (Python AST):"]
    for candidate in winners:
        doc = f" — {candidate.docstring}" if candidate.docstring else ""
        lines.append(
            f"- `{candidate.name}()` ({candidate.kind}, lines "
            f"{candidate.line_start}-{candidate.line_end}){doc}"
        )
    matched_terms = sorted({term for candidate in winners for term in candidate.matched_terms})
    lines.append(f"Matched query terms: {', '.join(matched_terms)}.")
    lines.append(
        "This is a lexical match over real definitions, not a semantic guess; "
        "read the cited lines before relying on them."
    )
    return "\n".join(lines)
