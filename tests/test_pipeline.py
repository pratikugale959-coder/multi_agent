"""End-to-end tests for the RAG pipeline orchestration (services faked)."""

from __future__ import annotations

from pathlib import Path

from code_helper.config import Settings
from code_helper.rag_pipeline import build_chunks, index_repository


def test_build_chunks_reads_and_splits(sample_project: Path, settings: Settings) -> None:
    chunks, skipped, errors = build_chunks(sample_project, settings)
    files = {c.file_path for c in chunks}
    assert {"alpha.py", "beta.py", "big.py", "broken.py"} <= files
    # big.py is 171 lines -> two chunks at size 100 / overlap 20
    assert len([c for c in chunks if c.file_path == "big.py"]) == 2
    assert skipped >= 1  # binary.py cannot be read
    assert errors


def test_index_repository_orchestrates_all_stages(
    sample_project: Path, settings: Settings, fake_embedder, fake_store
) -> None:
    report = index_repository(
        sample_project, settings=settings, embedder=fake_embedder, store=fake_store
    )
    assert report.chunks_created > 0
    assert report.chunks_indexed == report.chunks_created
    assert fake_store.ensure_called is True
    assert len(fake_store.upserts) == 1
    chunks, vectors = fake_store.upserts[0]
    assert len(chunks) == len(vectors)
    # Deterministic ids: file:start-end
    assert chunks[0].chunk_id.startswith(chunks[0].file_path + ":")


def test_index_repository_embeds_file_path_with_each_chunk(
    sample_project: Path, settings: Settings, fake_embedder, fake_store
) -> None:
    index_repository(sample_project, settings=settings, embedder=fake_embedder, store=fake_store)
    # Every chunk is embedded exactly once, with its relative path prepended.
    chunks = fake_store.upserts[0][0]
    embedded_texts = [t for batch in fake_embedder.document_batches for t in batch]
    assert len(embedded_texts) == len(chunks)
    for chunk, embedded in zip(chunks, embedded_texts, strict=True):
        assert embedded.startswith(f"# file: {chunk.file_path}\n")
        assert chunk.text in embedded


def test_index_repository_handles_empty_project(
    tmp_path: Path, settings: Settings, fake_embedder, fake_store
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    report = index_repository(empty, settings=settings, embedder=fake_embedder, store=fake_store)
    assert report.chunks_created == 0
    assert report.chunks_indexed == 0
    assert report.errors
