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

from code_helper.chunking import chunk_code
from code_helper.config import ConfigError, Settings, get_settings
from code_helper.embeddings import EmbeddingError
from code_helper.ingestion import IngestionError, read_source_file
from code_helper.paths import project_root
from code_helper.retrieval import RetrievedChunk, format_results, retrieve
from code_helper.vector_store import VectorStoreError

#: File extensions recognised when the user literally names a file in a query.
_NAMED_FILE_RE = re.compile(
    r"(?<![\w/.-])([\w./-]+\.(?:py|pyi|md|txt|toml|cfg|ini|json|ya?ml))(?![\w.])",
    re.IGNORECASE,
)

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


def rag_search(query: str, top_k: int = 5) -> str:
    """Search the indexed codebase for code that answers *query*.

    Use this to locate where a function, feature, variable or operation is
    implemented. If the query names a file (e.g. ``main.py``), the file is read
    directly and returned in full -- no semantic guessing. Results include the
    source filename, line range and similarity score so the answer can be
    traced back to real code.

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
            header = (
                f"Exact file match for {query!r}: {', '.join(found)}.\n"
                "This is a deterministic file lookup, not a semantic guess.\n\n"
            )
            return header + format_results(chunks, snippet_chars=None)
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

    return f"Query: {query}\n\n{format_results(results)}"
