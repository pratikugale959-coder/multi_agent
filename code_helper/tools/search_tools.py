"""Tool for the **Code Searcher** agent.

``rag_search`` embeds the question with the NVIDIA NIM embedding model, queries
Pinecone, and returns the matching code together with filenames, line ranges and
similarity scores. It never invents evidence: an empty result set is reported as
such.
"""

from __future__ import annotations

from code_helper.config import ConfigError, get_settings
from code_helper.embeddings import EmbeddingError
from code_helper.retrieval import format_results, retrieve
from code_helper.vector_store import VectorStoreError


def rag_search(query: str, top_k: int = 5) -> str:
    """Search the indexed codebase for code that answers *query*.

    Use this to locate where a function, feature, variable or operation is
    implemented. Results include the source filename, line range and similarity
    score so the answer can be traced back to real code.

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

    try:
        results = retrieve(query, top_k=resolved_top_k, settings=get_settings())
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

    return f"Query: {query}\n\n{format_results(results)}"
