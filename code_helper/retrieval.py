"""Stage 5 of the RAG pipeline -- retrieval.

This module is intentionally independent of ADK. It can be imported and tested on
its own: give it a natural-language query and it returns the most similar code
chunks, each carrying the source filename, the line range and the similarity
score. Agents simply call :func:`retrieve` and render the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from code_helper.config import Settings, get_settings
from code_helper.embeddings import EmbeddingClient
from code_helper.vector_store import PineconeVectorStore

#: How many characters of a chunk to show in a rendered result.
SNIPPET_CHARS = 1200


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved chunk with its provenance."""

    chunk_id: str
    file_path: str
    line_start: int
    line_end: int
    score: float
    text: str

    @property
    def line_range(self) -> str:
        """Human-readable line range, e.g. ``"12-45"``."""
        return f"{self.line_start}-{self.line_end}"


def _to_retrieved_chunk(match: dict[str, Any]) -> RetrievedChunk | None:
    """Convert a raw Pinecone match into a :class:`RetrievedChunk`."""
    metadata = match.get("metadata") or {}
    file_path = metadata.get("file_path")
    if not file_path:
        # Without provenance the match is useless for grounded answers.
        return None
    return RetrievedChunk(
        chunk_id=str(metadata.get("chunk_id") or match.get("id") or ""),
        file_path=str(file_path),
        line_start=int(metadata.get("line_start", 0) or 0),
        line_end=int(metadata.get("line_end", 0) or 0),
        score=float(match.get("score", 0.0) or 0.0),
        text=str(metadata.get("text", "") or ""),
    )


def retrieve(
    query: str,
    top_k: int | None = None,
    settings: Settings | None = None,
    embedder: EmbeddingClient | None = None,
    store: PineconeVectorStore | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the *top_k* most similar code chunks for *query*.

    Args:
        query: Natural-language question.
        top_k: Number of results (defaults to ``TOP_K``, initially 5).
        settings: Optional configuration override (useful in tests).
        embedder: Optional pre-built embedder (dependency injection).
        store: Optional pre-built vector store (dependency injection).

    Returns:
        Matching chunks, best first. Empty when nothing matches.

    Raises:
        ValueError: If *query* is empty/whitespace or *top_k* is not positive.
        EmbeddingError / VectorStoreError: Propagated from the stages above.
    """
    if not isinstance(query, str) or query.strip() == "":
        raise ValueError("Query must be a non-empty string")

    settings = settings or get_settings()
    resolved_top_k = top_k if top_k is not None else settings.top_k
    if resolved_top_k <= 0:
        raise ValueError(f"top_k must be positive, got {resolved_top_k}")

    embedder = embedder or EmbeddingClient(settings)
    store = store or PineconeVectorStore(settings)

    query_vector = embedder.embed_query(query)
    matches = store.query(query_vector, top_k=resolved_top_k)

    results: list[RetrievedChunk] = []
    for match in matches:
        converted = _to_retrieved_chunk(match)
        if converted is not None:
            results.append(converted)
    return results


def format_results(results: list[RetrievedChunk], max_results: int | None = None) -> str:
    """Render retrieved chunks as a plain-text block for an agent to read."""
    if not results:
        return "No relevant evidence was found in the indexed codebase."

    shown = results[:max_results] if max_results else results
    lines: list[str] = [f"Found {len(results)} relevant code chunk(s):", ""]
    for rank, result in enumerate(shown, start=1):
        snippet = result.text[:SNIPPET_CHARS]
        lines.append(
            f"[{rank}] {result.file_path} (lines {result.line_range}) similarity={result.score:.4f}"
        )
        lines.append("```python")
        lines.append(snippet)
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip()
