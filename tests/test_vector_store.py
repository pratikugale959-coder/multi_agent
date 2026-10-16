"""Tests for stage 4 -- Pinecone vector store (client faked, no network)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from code_helper.chunking import Chunk
from code_helper.config import ConfigError, Settings
from code_helper.vector_store import PineconeVectorStore, VectorStoreError


class FakeIndex:
    def __init__(self, matches=None) -> None:
        self.upserts: list[dict] = []
        self._matches = matches or []

    def upsert(self, vectors, namespace=""):
        self.upserts.append({"vectors": vectors, "namespace": namespace})

    def query(self, vector=None, top_k=5, include_metadata=True, namespace="", filter=None):
        return SimpleNamespace(matches=self._matches[:top_k])

    def describe_index_stats(self):
        return SimpleNamespace(
            total_vector_count=7,
            dimension=2048,
            namespaces={"test-ns": SimpleNamespace(vector_count=7)},
        )


class FakeClient:
    def __init__(self, names=(), dimension=2048, metric="cosine", ready=True) -> None:
        self._names = list(names)
        self.dimension = dimension
        self.metric = metric
        self.ready = ready
        self.created: list[dict] = []
        self.index_handle = FakeIndex()

    def list_indexes(self):
        return [SimpleNamespace(name=n) for n in self._names]

    def create_index(self, **kwargs):
        self.created.append(kwargs)
        self._names.append(kwargs["name"])

    def describe_index(self, name):
        return SimpleNamespace(
            name=name,
            dimension=self.dimension,
            metric=self.metric,
            status={"ready": self.ready},
            host="host",
        )

    def Index(self, name):
        return self.index_handle


def _store(settings: Settings, client: FakeClient, dimension: int = 2048) -> PineconeVectorStore:
    store = PineconeVectorStore(settings, dimension=dimension)
    store._pc = client  # inject the fake client
    return store


def test_creates_index_when_missing(settings: Settings) -> None:
    client = FakeClient(names=[])
    store = _store(settings, client)
    description = store.ensure_index()
    assert client.created, "index should have been created"
    created = client.created[0]
    assert created["name"] == "code-helper"
    assert created["dimension"] == 2048
    assert created["metric"] == "cosine"
    assert description["ready"] is True


def test_reuses_existing_index_without_creating(settings: Settings) -> None:
    client = FakeClient(names=["code-helper"], dimension=2048)
    store = _store(settings, client)
    store.ensure_index()
    assert client.created == []


def test_dimension_mismatch_raises(settings: Settings) -> None:
    client = FakeClient(names=["code-helper"], dimension=1536)
    store = _store(settings, client, dimension=2048)
    with pytest.raises(VectorStoreError, match="dimension"):
        store.ensure_index()


def test_metric_mismatch_raises(settings: Settings) -> None:
    client = FakeClient(names=["code-helper"], dimension=2048, metric="dotproduct")
    store = _store(settings, client)
    with pytest.raises(VectorStoreError, match="metric"):
        store.ensure_index()


def test_upsert_rejects_misaligned_inputs(settings: Settings) -> None:
    store = _store(settings, FakeClient(names=["code-helper"]))
    chunk = Chunk("c1", "f.py", 1, 2, "text")
    with pytest.raises(VectorStoreError, match="same length"):
        store.upsert_chunks([chunk], [])


def test_upsert_rejects_wrong_vector_dimension(settings: Settings) -> None:
    store = _store(settings, FakeClient(names=["code-helper"]), dimension=2048)
    chunk = Chunk("c1", "f.py", 1, 2, "text")
    with pytest.raises(VectorStoreError, match="dims"):
        store.upsert_chunks([chunk], [[0.0] * 8])


def test_upsert_payload_shape_and_deterministic_ids(settings: Settings) -> None:
    client = FakeClient(names=["code-helper"])
    store = _store(settings, client, dimension=4)
    chunk = Chunk("f.py:1-10", "f.py", 1, 10, "print(1)")
    stored = store.upsert_chunks([chunk], [[0.1, 0.2, 0.3, 0.4]])
    assert stored == 1
    payload = client.index_handle.upserts[0]["vectors"][0]
    assert payload["id"] == "f.py:1-10"
    assert payload["values"] == [0.1, 0.2, 0.3, 0.4]
    assert payload["metadata"]["file_path"] == "f.py"
    assert payload["metadata"]["line_start"] == 1


def test_query_normalises_matches(settings: Settings) -> None:
    matches = [
        SimpleNamespace(id="m1", score=0.9, metadata={"file_path": "a.py", "text": "x"}),
        SimpleNamespace(id="m2", score=0.5, metadata={"file_path": "b.py", "text": "y"}),
    ]
    client = FakeClient(names=["code-helper"])
    client.index_handle = FakeIndex(matches=matches)
    store = _store(settings, client, dimension=4)
    result = store.query([0.0] * 4, top_k=2)
    assert [m["id"] for m in result] == ["m1", "m2"]
    assert result[0]["metadata"]["file_path"] == "a.py"


def test_stats_reads_namespace_counts(settings: Settings) -> None:
    store = _store(settings, FakeClient(names=["code-helper"]))
    stats = store.stats()
    assert stats["total_vector_count"] == 7
    assert stats["namespaces"]["test-ns"] == 7


def test_missing_pinecone_key_raises_config_error(settings: Settings) -> None:
    store = PineconeVectorStore(replace(settings, pinecone_api_key=""))
    with pytest.raises(ConfigError):
        _ = store.client
