"""Tool for the **Code Searcher** agent.

``rag_search`` answers two kinds of questions:

* **Named file** (e.g. "where is ``main.py``?") -- resolved *deterministically*
  by reading that file from the target project. Semantic search is skipped
  because a small index can omit the one file the user asked about.
* **Everything else** -- embedded with the NVIDIA NIM model and matched against
  the Pinecone index, returning filenames, line ranges and similarity scores.

Neither path invents evidence: a missing file or an empty result set is reported
as such so the agent can say "not found" instead of guessing.
"""

from __future__ import annotations

import re

from code_helper.core.config import ConfigError, Settings, get_settings
from code_helper.core.paths import project_root
from code_helper.rag.chunking import chunk_code
from code_helper.rag.embeddings import EmbeddingError
from code_helper.rag.ingestion import IngestionError, read_source_file
from code_helper.rag.retrieval import RetrievedChunk, format_results, retrieve
from code_helper.rag.vector_store import VectorStoreError

#: File extensions recognised when the user literally names a file in a query.
_NAMED_FILE_RE = re.compile(
    r"(?<![\w/.-])([\w./-]+\.(?:py|pyi|md|txt|toml|cfg|ini|json|ya?ml))(?![\w.])",
    re.IGNORECASE,
)

#: Named-file lookups are capped so a mixed question about a large file cannot
#: flood the chat prompt. The outline tool remains the way to inspect a whole file.
_EXACT_LOOKUP_MAX_CHUNKS = 4
_EXACT_LOOKUP_SNIPPET_CHARS = 1500

#: Common words ignored by the lexical relevance check below.
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "all",
        "also",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "by",
        "can",
        "code",
        "codebase",
        "could",
        "did",
        "do",
        "does",
        "done",
        "each",
        "file",
        "files",
        "find",
        "for",
        "from",
        "give",
        "had",
        "has",
        "have",
        "here",
        "how",
        "implement",
        "implemented",
        "into",
        "is",
        "it",
        "its",
        "just",
        "located",
        "may",
        "might",
        "more",
        "most",
        "must",
        "not",
        "of",
        "on",
        "or",
        "other",
        "our",
        "over",
        "project",
        "repo",
        "repository",
        "same",
        "should",
        "show",
        "some",
        "such",
        "tell",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "under",
        "used",
        "using",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)


def _named_files(query: str) -> list[str]:
    """Return the file paths mentioned literally in *query*, de-duplicated."""
    return list(dict.fromkeys(_NAMED_FILE_RE.findall(query)))


def _exact_file_lookup(
    query: str, settings: Settings
) -> tuple[list[RetrievedChunk], list[str]] | None:
    """Read files named literally in *query*, without semantic search.

    Returns:
        ``None`` when the query names no file. Otherwise ``(chunks, missing)``
        where *chunks* are the real file contents and *missing* lists names that
        could not be read (missing, excluded, binary or outside the root).
    """
    names = _named_files(query)
    if not names:
        return None

    root = project_root(settings)
    chunks: list[RetrievedChunk] = []
    missing: list[str] = []

    for name in names:
        try:
            source = read_source_file(name, root=root)
        except (IngestionError, ValueError):
            missing.append(name)
            continue

        for chunk in chunk_code(
            source.text,
            source.relative_path,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        ):
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    file_path=chunk.file_path,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    score=1.0,
                    text=chunk.text,
                )
            )

    return chunks, missing


def _lexically_supported(query: str, results: list[RetrievedChunk]) -> bool:
    """Return ``True`` when a result mentions at least one meaningful query term.

    Similarity scores are not comparable across questions, so a low score alone
    cannot prove irrelevance. This cheap lexical check catches the important
    failure mode: returning confident chunks for a topic the project does not
    contain (for example "Kubernetes") and inviting the model to speculate.
    Genuine semantic matches almost always share at least one content term, so
    the gate stays conservative.
    """
    tokens = {
        token
        for token in re.findall(r"[a-z_][a-z0-9_]{2,}", query.lower())
        if token not in _STOPWORDS
    }
    if not tokens:
        return True
    for result in results:
        haystack = f"{result.file_path}\n{result.text}".lower()
        if any(token in haystack for token in tokens):
            return True
    return False


def _meaningful_terms(query: str) -> set[str]:
    """Return the query's content terms, using the same rule as the gate."""
    return {
        token
        for token in re.findall(r"[a-z_][a-z0-9_]{2,}", query.lower())
        if token not in _STOPWORDS
    }


def _best_term_match(
    query: str, results: list[RetrievedChunk]
) -> tuple[RetrievedChunk, set[str]] | None:
    """Return the sole result that contains every meaningful query term.

    Vector order is usually good, but a shortened or rephrased worker query can
    rank a storage helper above the file that actually defines the operation.
    When exactly one result mentions *every* query term, that is strong,
    deterministic evidence for where the operation lives. Ambiguous cases
    (several full matches, or no full match) return ``None`` and keep normal
    retrieval order.
    """
    terms = _meaningful_terms(query)
    if len(terms) < 2:
        return None

    full_matches: list[RetrievedChunk] = []
    for result in results:
        haystack = f"{result.file_path}\n{result.text}".lower()
        if all(term in haystack for term in terms):
            full_matches.append(result)
    if len(full_matches) != 1:
        return None
    return full_matches[0], terms


def rag_search(query: str, top_k: int = 5) -> str:
    """Search the indexed codebase for code that answers *query*.

    Use this to locate where a function, feature, variable or operation is
    implemented. If the query names a file (e.g. ``main.py``), the file is read
    directly instead of searched semantically. Named-file output is capped
    (first few chunks, truncated snippets) so a large file cannot flood the
    prompt. Results include the source filename and line range so the answer
    can be traced back to real code.

    Args:
        query: A natural-language description of what to find.
        top_k: Maximum number of chunks to return (default 5).

    Returns:
        A formatted, human-readable list of matching code chunks, or a clear
        "no evidence" / error message. Never raises for user-input problems.
    """
    if not isinstance(query, str) or query.strip() == "":
        return "Error: rag_search requires a non-empty query string."

    # Language models sometimes send numbers as strings (e.g. "5"); coerce safely.
    try:
        resolved_top_k = int(top_k)
    except (TypeError, ValueError):
        return f"Error: top_k must be an integer, got {top_k!r}."
    if resolved_top_k <= 0:
        return f"Error: top_k must be a positive integer, got {resolved_top_k}."

    settings = get_settings()

    # A literal filename is answered by reading that file, not by similarity:
    # with a small index top-k search can omit the very file the user named.
    exact = _exact_file_lookup(query, settings)
    if exact is not None:
        chunks, missing = exact
        if chunks:
            found = list(dict.fromkeys(chunk.file_path for chunk in chunks))
            shown = chunks[:_EXACT_LOOKUP_MAX_CHUNKS]
            header = (
                f"Exact file match for {query!r}: {', '.join(found)}.\n"
                "This is a deterministic file lookup, not a semantic guess.\n"
                "Report the relative path and a one-sentence purpose. Do not paste "
                "the file; quote at most 5 relevant lines.\n"
            )
            if len(shown) < len(chunks):
                header += (
                    f"Showing the first {len(shown)} of {len(chunks)} chunk(s); use "
                    "summarize_project_file or read_project_file for the rest.\n"
                )
            header += "\n"
            return header + format_results(shown, snippet_chars=_EXACT_LOOKUP_SNIPPET_CHARS)
        names = ", ".join(repr(name) for name in missing)
        return (
            f"No readable file named {names} was found under the project root "
            f"({project_root(settings)}).\n"
            "Do NOT infer its contents from imports or other files. Report that the "
            "file was not found."
        )

    try:
        results = retrieve(query, top_k=resolved_top_k, settings=settings)
    except ConfigError as exc:
        return f"Search unavailable -- configuration problem: {exc}"
    except EmbeddingError as exc:
        return f"Search failed while embedding the query: {exc}"
    except VectorStoreError as exc:
        return f"Search failed while querying the vector database: {exc}"
    except ValueError as exc:
        return f"Search failed: {exc}"

    if not results:
        return (
            f"No relevant evidence found for: {query!r}. "
            "The codebase may not be indexed yet, or the question may not match any code. "
            "Try rephrasing, or run the indexer first."
        )

    if not _lexically_supported(query, results):
        return (
            f"No relevant evidence found for: {query!r}. "
            "None of the retrieved code chunks mention the requested topic, so the "
            "project does not appear to contain it. Report that it was not found; "
            "do NOT speculate."
        )

    header = f"Query: {query}\n"
    best_match = _best_term_match(query, results)
    if best_match is not None:
        best_result, terms = best_match
        if best_result is not results[0]:
            # Put the sole full-term match first. The hint alone was not always
            # enough: a model tends to cite the top-ranked block, which may be a
            # storage helper. Ordering is deterministic evidence, not a guess.
            results = [best_result, *(result for result in results if result is not best_result)]
            header += (
                f"\nBest query-term match (listed first): `{best_result.file_path}` "
                f"(lines {best_result.line_range}) mentions every meaningful query "
                f"term ({', '.join(sorted(terms))}). Prefer this file when explaining "
                "where the operation is defined; do not cite a helper that merely "
                "receives its output.\n"
            )

    return f"{header}\n{format_results(results)}"
