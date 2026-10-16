"""Stage 2 of the RAG pipeline -- chunking.

We implement a **custom** line-based sliding-window chunker. LangChain and
LlamaIndex splitter classes are deliberately not used (the assignment forbids
framework-managed RAG).

Why line-based windows?
-----------------------
Code is naturally organised in lines, so reporting ``line_start``/``line_end``
gives the reader a precise place to look. A fixed number of lines per chunk is
simple and predictable, which matters more for a teaching project than the
slightly better boundaries a syntax-aware splitter would produce.

Limitations (also documented in the README)
-------------------------------------------
* A chunk can start in the middle of a function or end mid-statement, so a
  retrieved snippet is a *pointer* into the file, not a self-contained unit.
* A very long function may be split across several chunks; the overlap window
  mitigates this but cannot eliminate it.
* Blank-line-heavy files produce chunks with little information, and highly
  repetitive files produce near-duplicate chunks.
* Overlap increases storage and embedding cost proportionally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Pinecone stores at most ~40 KB of metadata per vector. We keep the chunk text
#: below that so an upsert never fails on metadata size.
MAX_METADATA_TEXT_CHARS: int = 40_000

#: Default experimental chunking parameters (per the assignment starting point).
DEFAULT_CHUNK_SIZE: int = 100
DEFAULT_CHUNK_OVERLAP: int = 20


class ChunkingError(ValueError):
    """Raised when chunking parameters are invalid."""


@dataclass(frozen=True)
class Chunk:
    """One overlapping window of a source file."""

    chunk_id: str
    file_path: str
    line_start: int
    line_end: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_vector_metadata(self) -> dict[str, Any]:
        """Return the metadata payload stored alongside the vector in Pinecone.

        Only primitives are used, because Pinecone metadata must be scalar or a
        list of scalars.
        """
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "text": self.text[:MAX_METADATA_TEXT_CHARS],
        }


def chunk_code(
    text: str,
    relative_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split *text* into overlapping line-window :class:`Chunk` objects.

    Args:
        text: Original file text.
        relative_path: POSIX-style path used in chunk ids and metadata.
        chunk_size: Lines per chunk (must be > 0).
        overlap: Lines shared with the next chunk (``0 <= overlap < chunk_size``).

    Returns:
        A list of chunks in file order. Empty or whitespace-only windows are
        dropped, and windows whose text duplicates an earlier window are skipped.

    Raises:
        ChunkingError: If ``chunk_size`` or ``overlap`` are out of range.
    """
    if chunk_size <= 0:
        raise ChunkingError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ChunkingError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ChunkingError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size}) "
            "or the window would never advance"
        )

    if text == "":
        return []

    lines = text.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    seen_texts: set[str] = set()

    start = 0
    while start < total_lines:
        end = min(start + chunk_size, total_lines)  # exclusive
        window = lines[start:end]
        body = "\n".join(window)

        if body.strip() and body not in seen_texts:
            seen_texts.add(body)
            line_start = start + 1  # 1-based, inclusive
            line_end = end  # 1-based, inclusive
            chunk_id = f"{relative_path}:{line_start}-{line_end}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    file_path=relative_path,
                    line_start=line_start,
                    line_end=line_end,
                    text=body,
                )
            )

        if end >= total_lines:
            break
        start += step

    return chunks


def chunk_source_files(
    sources,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk an iterable of :class:`~code_helper.ingestion.SourceFile` objects."""
    chunks: list[Chunk] = []
    for source in sources:
        chunks.extend(
            chunk_code(
                source.text,
                relative_path=source.relative_path,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    return chunks
