"""Tests for stage 3 -- NVIDIA NIM embeddings (LiteLLM mocked, no network)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from code_helper import embeddings as embeddings_module
from code_helper.config import ConfigError, Settings
from code_helper.embeddings import EmbeddingClient, EmbeddingError, embed_documents, embed_query


class ApiError(Exception):
    """Stand-in for a LiteLLM HTTP error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _response(vectors: list[list[float]], shuffle: bool = False):
    data = [{"embedding": v, "index": i} for i, v in enumerate(vectors)]
    if shuffle:
        data = list(reversed(data))
    return SimpleNamespace(data=data)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove backoff delays so retry tests run instantly."""
    monkeypatch.setattr(embeddings_module.time, "sleep", lambda *_: None)


def test_embed_documents_returns_vectors_in_order(settings: Settings, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_embedding(**kwargs):
        calls.append(kwargs["input"])
        return _response([[float(i)] * 4 for i in range(len(kwargs["input"]))])

    monkeypatch.setattr(embeddings_module.litellm, "embedding", fake_embedding)
    client = EmbeddingClient(replace(settings, embedding_dimension=4))
    vectors = client.embed_documents(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(v) == 4 for v in vectors)
    assert calls == [["a", "b", "c"]]


def test_embed_documents_batches_requests(settings: Settings, monkeypatch) -> None:
    calls: list[int] = []

    def fake_embedding(**kwargs):
        calls.append(len(kwargs["input"]))
        return _response([[0.0] * 4 for _ in kwargs["input"]])

    monkeypatch.setattr(embeddings_module.litellm, "embedding", fake_embedding)
    client = EmbeddingClient(replace(settings, embedding_dimension=4))
    client.embed_documents(["a", "b", "c", "d", "e"], batch_size=2)
    assert calls == [2, 2, 1]


def test_response_order_is_restored_from_index(settings: Settings, monkeypatch) -> None:
    def fake_embedding(**kwargs):
        return _response([[1.0] * 4, [2.0] * 4], shuffle=True)

    monkeypatch.setattr(embeddings_module.litellm, "embedding", fake_embedding)
    client = EmbeddingClient(replace(settings, embedding_dimension=4))
    vectors = client.embed_documents(["a", "b"])
    # index 0 -> [1.0...], index 1 -> [2.0...] regardless of arrival order
    assert vectors[0][0] == 1.0
    assert vectors[1][0] == 2.0


def test_dimension_mismatch_raises(settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr(
        embeddings_module.litellm,
        "embedding",
        lambda **kw: _response([[0.0] * 3 for _ in kw["input"]]),
    )
    client = EmbeddingClient(replace(settings, embedding_dimension=2048))
    with pytest.raises(EmbeddingError, match="dimension mismatch"):
        client.embed_documents(["a"])


def test_malformed_response_raises(settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr(
        embeddings_module.litellm, "embedding", lambda **kw: SimpleNamespace(data=None)
    )
    client = EmbeddingClient(replace(settings, embedding_dimension=4))
    with pytest.raises(EmbeddingError, match="no data"):
        client.embed_documents(["a"])


def test_count_mismatch_raises(settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr(embeddings_module.litellm, "embedding", lambda **kw: _response([[0.0] * 4]))
    client = EmbeddingClient(replace(settings, embedding_dimension=4))
    with pytest.raises(EmbeddingError, match="count mismatch"):
        client.embed_documents(["a", "b"])


def test_missing_api_key_raises_config_error(settings: Settings, monkeypatch) -> None:
    def should_not_be_called(**kwargs):  # pragma: no cover - must not run
        raise AssertionError("embedding must not be called without a key")

    monkeypatch.setattr(embeddings_module.litellm, "embedding", should_not_be_called)
    client = EmbeddingClient(replace(settings, nvidia_api_key=""))
    with pytest.raises(ConfigError):
        client.embed_documents(["a"])


def test_authentication_error_fails_fast(settings: Settings, monkeypatch) -> None:
    calls = {"n": 0}

    def fake_embedding(**kwargs):
        calls["n"] += 1
        raise ApiError("401 unauthorized", status_code=401)

    monkeypatch.setattr(embeddings_module.litellm, "embedding", fake_embedding)
    client = EmbeddingClient(replace(settings, embedding_dimension=4))
    with pytest.raises(EmbeddingError, match="credentials"):
        client.embed_documents(["a"])
    assert calls["n"] == 1, "auth errors must not be retried"


def test_rate_limit_is_retried_then_succeeds(settings: Settings, monkeypatch) -> None:
    calls = {"n": 0}

    def fake_embedding(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ApiError("429 rate limited", status_code=429)
        return _response([[0.5] * 4 for _ in kwargs["input"]])

    monkeypatch.setattr(embeddings_module.litellm, "embedding", fake_embedding)
    client = EmbeddingClient(replace(settings, embedding_dimension=4))
    vectors = client.embed_documents(["a"])
    assert calls["n"] == 2
    assert vectors[0][0] == 0.5


def test_non_retryable_error_fails_immediately(settings: Settings, monkeypatch) -> None:
    calls = {"n": 0}

    def fake_embedding(**kwargs):
        calls["n"] += 1
        raise ApiError("400 bad request", status_code=400)

    monkeypatch.setattr(embeddings_module.litellm, "embedding", fake_embedding)
    client = EmbeddingClient(replace(settings, embedding_dimension=4))
    with pytest.raises(EmbeddingError):
        client.embed_documents(["a"])
    assert calls["n"] == 1


def test_never_fabricates_vectors_when_all_attempts_fail(settings: Settings, monkeypatch) -> None:
    def fake_embedding(**kwargs):
        raise ApiError("503 unavailable", status_code=503)

    monkeypatch.setattr(embeddings_module.litellm, "embedding", fake_embedding)
    client = EmbeddingClient(replace(settings, embedding_dimension=4))
    with pytest.raises(EmbeddingError):
        client.embed_documents(["a"])


def test_empty_document_text_is_rejected(settings: Settings) -> None:
    client = EmbeddingClient(settings)
    with pytest.raises(EmbeddingError):
        client.embed_documents(["ok", "   "])


def test_embed_query_rejects_empty_string(settings: Settings) -> None:
    client = EmbeddingClient(settings)
    with pytest.raises(ValueError):
        client.embed_query("   ")


def test_embed_query_is_cached_per_client(settings: Settings, monkeypatch) -> None:
    """Repeating a query must not repeat the network call."""
    calls = {"n": 0}

    def fake_embedding(**kwargs):
        calls["n"] += 1
        return _response([[0.25] * 4 for _ in kwargs["input"]])

    monkeypatch.setattr(embeddings_module.litellm, "embedding", fake_embedding)
    client = EmbeddingClient(replace(settings, embedding_dimension=4))
    first = client.embed_query("where is upsert")
    second = client.embed_query("where is upsert")
    assert first == second
    assert calls["n"] == 1


def test_module_level_helpers_delegate(settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr(
        embeddings_module.litellm,
        "embedding",
        lambda **kw: _response([[0.0] * 4 for _ in kw["input"]]),
    )
    small = replace(settings, embedding_dimension=4)
    assert len(embed_documents(["x"], small)) == 1
    assert len(embed_query("x", small)) == 4
