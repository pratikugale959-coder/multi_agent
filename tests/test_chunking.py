"""Tests for stage 2 -- chunk boundaries, overlap and stable ids."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from code_helper.chunking import Chunk, ChunkingError, chunk_code, chunk_source_files
from code_helper.ingestion import SourceFile


def _numbered_text(n: int) -> str:
    return "\n".join(f"line_{i}" for i in range(1, n + 1))


def test_chunk_boundaries_and_line_numbers() -> None:
    chunks = chunk_code(_numbered_text(250), "f.py", chunk_size=100, overlap=20)
    assert [(c.line_start, c.line_end) for c in chunks] == [(1, 100), (81, 180), (161, 250)]


def test_chunk_overlap_is_twenty_lines() -> None:
    chunks = chunk_code(_numbered_text(250), "f.py", chunk_size=100, overlap=20)
    first, second = chunks[0], chunks[1]
    # Second chunk starts 20 lines before the first ends.
    assert first.line_end - second.line_start == 19  # inclusive ranges -> 20 shared lines


def test_small_file_produces_single_chunk() -> None:
    chunks = chunk_code(_numbered_text(30), "small.py", chunk_size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].line_start == 1
    assert chunks[0].line_end == 30


def test_chunk_ids_are_stable_and_deterministic() -> None:
    first = chunk_code(_numbered_text(120), "pkg/mod.py", chunk_size=50, overlap=10)
    second = chunk_code(_numbered_text(120), "pkg/mod.py", chunk_size=50, overlap=10)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first[0].chunk_id == "pkg/mod.py:1-50"


def test_empty_text_produces_no_chunks() -> None:
    assert chunk_code("", "empty.py") == []


def test_whitespace_only_text_produces_no_chunks() -> None:
    assert chunk_code("\n\n   \n\t\n", "blank.py") == []


def test_no_duplicate_or_empty_chunks() -> None:
    text = "\n".join(["x = 1"] * 300)
    chunks = chunk_code(text, "repeat.py", chunk_size=100, overlap=20)
    texts = [c.text for c in chunks]
    assert len(texts) == len(set(texts)), "duplicate chunk text produced"


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [(0, 0), (-5, 0), (10, 10), (10, 20), (10, -1)],
)
def test_invalid_parameters_raise(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ChunkingError):
        chunk_code(_numbered_text(50), "f.py", chunk_size=chunk_size, overlap=overlap)


def test_metadata_payload_is_scalar_only() -> None:
    chunk = chunk_code(_numbered_text(10), "f.py")[0]
    metadata = chunk.to_vector_metadata()
    assert metadata["file_path"] == "f.py"
    assert metadata["line_start"] == 1
    assert metadata["chunk_id"] == "f.py:1-10"
    assert all(isinstance(v, (str, int, float, bool)) for v in metadata.values())


def test_metadata_text_is_truncated_to_pinecone_limit() -> None:
    huge = "\n".join("a" * 200 for _ in range(500))
    chunk = chunk_code(huge, "huge.py", chunk_size=500, overlap=0)[0]
    assert len(chunk.to_vector_metadata()["text"]) <= 40_000


def test_chunk_source_files_preserves_order() -> None:
    sources = [
        SourceFile("a.py", None, "x = 1\ny = 2"),  # type: ignore[arg-type]
        SourceFile("b.py", None, "z = 3"),  # type: ignore[arg-type]
    ]
    chunks = chunk_source_files(sources, chunk_size=10, overlap=0)
    assert [c.file_path for c in chunks] == ["a.py", "b.py"]


def test_chunk_is_frozen_dataclass() -> None:
    chunk = Chunk("id", "f.py", 1, 2, "text")
    with pytest.raises(FrozenInstanceError):
        chunk.file_path = "other.py"  # type: ignore[misc]
