"""Stage 4 of the RAG pipeline -- Pinecone serverless vector storage.

Design notes
------------
* The index is created lazily and **never deleted**. If the index already exists
  its dimension and metric are validated against the configured values, and a
  mismatch raises a clear error instead of silently corrupting the index.
* Vector ids are deterministic (``file:start-end``), so re-indexing the same file
  overwrites the same records instead of accumulating duplicates.
* Deletion is not exposed by this module at all -- destroying data requires an
  explicit, deliberate action outside the normal pipeline.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from code_helper.chunking import Chunk
from code_helper.config import Settings, get_settings, require_pinecone_credentials

DEFAULT_UPSERT_BATCH = 100
INDEX_READY_TIMEOUT_SECONDS = 120


class VectorStoreError(RuntimeError):
    """Raised for any Pinecone configuration or request failure."""


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read *name* from an object attribute or a mapping."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


class PineconeVectorStore:
    """Serverless Pinecone index wrapper for code chunks."""

    def __init__(
        self,
        settings: Settings | None = None,
        dimension: int | None = None,
        index_name: str | None = None,
        namespace: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.dimension = dimension or self.settings.embedding_dimension
        self.index_name = index_name or self.settings.pinecone_index_name
        self.namespace = namespace if namespace is not None else self.settings.pinecone_namespace
        self.metric = "cosine"
        self._pc: Pinecone | None = None
        self._index = None

    # -- connection --------------------------------------------------------- #
    @property
    def client(self) -> Pinecone:
        """Lazily create the Pinecone client (requires a key)."""
        if self._pc is None:
            api_key = require_pinecone_credentials(self.settings)
            self._pc = Pinecone(api_key=api_key)
        return self._pc

    @property
    def index(self):
        """Lazily obtain the :class:`pinecone.Index` handle."""
        if self._index is None:
            self._index = self.client.Index(self.index_name)
        return self._index

    # -- index lifecycle ---------------------------------------------------- #
    def list_index_names(self) -> list[str]:
        """Return the names of all indexes in the Pinecone project."""
        return [_attr(item, "name") for item in self.client.list_indexes()]

    def describe(self) -> dict[str, Any]:
        """Return a normalised description of the index."""
        info = self.client.describe_index(self.index_name)
        status = _attr(info, "status", {})
        ready = status.get("ready") if isinstance(status, Mapping) else _attr(status, "ready")
        return {
            "name": _attr(info, "name", self.index_name),
            "dimension": _attr(info, "dimension"),
            "metric": _attr(info, "metric"),
            "host": _attr(info, "host"),
            "ready": ready,
        }

    def ensure_index(self) -> dict[str, Any]:
        """Create the serverless index if missing; validate it if it exists.

        Returns:
            The normalised index description.

        Raises:
            VectorStoreError: On dimension or metric mismatch.
        """
        if self.index_name not in self.list_index_names():
            try:
                self.client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud=self.settings.pinecone_cloud,
                        region=self.settings.pinecone_region,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(
                    f"Failed to create Pinecone index '{self.index_name}': {exc}"
                ) from exc
            self._wait_until_ready()
            self._index = None  # force a fresh handle
            return self.describe()

        description = self.describe()
        existing_dim = description.get("dimension")
        existing_metric = description.get("metric")
        if existing_dim is not None and int(existing_dim) != int(self.dimension):
            raise VectorStoreError(
                f"Pinecone index '{self.index_name}' has dimension {existing_dim}, but "
                f"the embedding model is configured for {self.dimension}. Refusing to use "
                "a mismatched index. Either point EMBEDDING_DIMENSION at the correct model "
                "or create a new index."
            )
        if existing_metric is not None and str(existing_metric) != self.metric:
            raise VectorStoreError(
                f"Pinecone index '{self.index_name}' uses metric '{existing_metric}', but "
                f"'{self.metric}' is required."
            )
        return description

    def _wait_until_ready(self, timeout: float = INDEX_READY_TIMEOUT_SECONDS) -> None:
        """Block until the index reports ready (best effort)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.describe().get("ready"):
                    return
            except Exception:  # noqa: BLE001 - transient during provisioning
                pass
            time.sleep(1.0)

    # -- writes ------------------------------------------------------------- #
    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        batch_size: int = DEFAULT_UPSERT_BATCH,
    ) -> int:
        """Upsert chunk vectors and metadata. Returns the number stored.

        Chunks and vectors must be the same length and in the same order.
        """
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must be the same length"
            )
        if not chunks:
            return 0

        payload = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            values = [float(value) for value in vector]
            if len(values) != self.dimension:
                raise VectorStoreError(
                    f"Vector for {chunk.chunk_id} has {len(values)} dims, expected {self.dimension}"
                )
            payload.append(
                {
                    "id": chunk.chunk_id,
                    "values": values,
                    "metadata": chunk.to_vector_metadata(),
                }
            )

        stored = 0
        for start in range(0, len(payload), batch_size):
            batch = payload[start : start + batch_size]
            try:
                self.index.upsert(vectors=batch, namespace=self.namespace)
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"Pinecone upsert failed: {exc}") from exc
            stored += len(batch)
        return stored

    # -- reads -------------------------------------------------------------- #
    def query(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        filter: Mapping[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """Return raw Pinecone matches for a query vector."""
        values = [float(value) for value in vector]
        try:
            response = self.index.query(
                vector=values,
                top_k=top_k,
                include_metadata=include_metadata,
                namespace=self.namespace,
                filter=dict(filter) if filter else None,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Pinecone query failed: {exc}") from exc

        matches = _attr(response, "matches", []) or []
        normalised: list[dict[str, Any]] = []
        for match in matches:
            normalised.append(
                {
                    "id": _attr(match, "id"),
                    "score": float(_attr(match, "score", 0.0)),
                    "metadata": dict(_attr(match, "metadata", {}) or {}),
                }
            )
        return normalised

    def stats(self) -> dict[str, Any]:
        """Return index statistics (vector counts, dimension, namespaces)."""
        try:
            raw = self.index.describe_index_stats()
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Could not read index stats: {exc}") from exc
        namespaces = _attr(raw, "namespaces", {}) or {}
        return {
            "total_vector_count": _attr(raw, "total_vector_count", 0),
            "dimension": _attr(raw, "dimension", self.dimension),
            "namespaces": {
                name: _attr(value, "vector_count", 0) for name, value in dict(namespaces).items()
            },
            "metric": self.metric,
            "index_name": self.index_name,
        }


def upsert_chunks(
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    settings: Settings | None = None,
) -> int:
    """Convenience wrapper: ensure the index exists, then upsert."""
    store = PineconeVectorStore(settings)
    store.ensure_index()
    return store.upsert_chunks(chunks, vectors)


def iter_chunk_ids(chunks: Iterable[Chunk]) -> list[str]:
    """Return the deterministic ids for a sequence of chunks."""
    return [chunk.chunk_id for chunk in chunks]
