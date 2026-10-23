"""Tests for stage 5 -- retrieval and result formatting (no network)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from code_helper.config import Settings
from code_helper.rag_pipeline import answer_question
from code_helper.retrieval import RetrievedChunk, format_results, retrieve


def _match(file_path: str, start: int, end: int, score: float, text: str = "print('hi')") -> dict:
    return {
        "id": f"{file_path}:{start}-{end}",
        "score": score,
        "metadata": {
            "chunk_id": f"{file_path}:{start}-{end}",
            "file_path": file_path,
            "line_start": start,
            "line_end": end,
            "text": text,
        },
    }


def test_retrieve_returns_provenance(settings: Settings, fake_embedder, fake_store) -> None:
    fake_store.matches = [_match("vector_store.py", 40, 95, 0.87)]
    results = retrieve(
        "where is upsert", settings=settings, embedder=fake_embedder, store=fake_store
    )
    assert len(results) == 1
    chunk = results[0]
    assert chunk.file_path == "vector_store.py"
    assert chunk.line_range == "40-95"
    assert chunk.score == pytest.approx(0.87)
    assert fake_embedder.queries == ["where is upsert"]


def test_retrieve_empty_query_raises(settings: Settings, fake_embedder, fake_store) -> None:
    with pytest.raises(ValueError):
        retrieve("   ", settings=settings, embedder=fake_embedder, store=fake_store)


def test_retrieve_invalid_top_k_raises(settings: Settings, fake_embedder, fake_store) -> None:
    with pytest.raises(ValueError):
        retrieve("q", top_k=0, settings=settings, embedder=fake_embedder, store=fake_store)


def test_retrieve_empty_store_returns_empty(settings: Settings, fake_embedder, fake_store) -> None:
    fake_store.matches = []
    assert retrieve("q", settings=settings, embedder=fake_embedder, store=fake_store) == []


def test_retrieve_skips_matches_without_provenance(
    settings: Settings, fake_embedder, fake_store
) -> None:
    fake_store.matches = [{"id": "x", "score": 0.9, "metadata": {"text": "no file"}}]
    assert retrieve("q", settings=settings, embedder=fake_embedder, store=fake_store) == []


def test_format_results_reports_no_evidence() -> None:
    assert "No relevant evidence" in format_results([])


def test_format_results_includes_file_lines_and_score() -> None:
    chunk = RetrievedChunk("id", "main.py", 1, 10, 0.5, "def main(): pass")
    rendered = format_results([chunk])
    assert "main.py" in rendered
    assert "1-10" in rendered
    assert "0.5000" in rendered
    assert "def main" in rendered


def test_answer_question_payload(settings: Settings, fake_embedder, fake_store) -> None:
    fake_store.matches = [_match("a.py", 1, 5, 0.9), _match("b.py", 6, 9, 0.8)]
    payload = answer_question("q", settings=settings, embedder=fake_embedder, store=fake_store)
    assert payload["has_evidence"] is True
    assert payload["source_files"] == ["a.py", "b.py"]
    assert len(payload["results"]) == 2


def test_answer_question_without_evidence(settings: Settings, fake_embedder, fake_store) -> None:
    fake_store.matches = []
    payload = answer_question("q", settings=settings, embedder=fake_embedder, store=fake_store)
    assert payload["has_evidence"] is False
    assert payload["source_files"] == []


def test_shared_clients_are_reused_for_same_settings(settings: Settings) -> None:
    """A cached Pinecone handle avoids a multi-second control-plane lookup."""
    from code_helper.retrieval import _shared_embedder, _shared_store

    assert _shared_embedder(settings) is _shared_embedder(settings)
    assert _shared_store(settings) is _shared_store(settings)
    # Different settings must not share a client.
    other = replace(settings, pinecone_index_name="another-index")
    assert _shared_store(other) is not _shared_store(settings)
