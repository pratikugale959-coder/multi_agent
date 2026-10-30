"""Stage 4 of the RAG pipeline -- Pinecone serverless vector storage.

Design notes
------------
* The index is created lazily and **never deleted**. If the index already exists
  its dimension and metric are validated against the configured values, and a
  mismatch raises a clear error instead of silently corrupting the index.
* Vector ids are deterministic (``file:start-end``), so re-indexing the same file
  overwrites the same records instead of accumulating duplicates.
* Deleting vectors is limited to the explicit :meth:`PineconeVectorStore.delete_ids`
  maintenance helper used by ``--prune``. Normal indexing never destroys data.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from code_helper.core.config import Settings, get_settings, require_pinecone_credentials
from code_helper.rag.chunking import Chunk

DEFAULT_UPSERT_BATCH = 100
DEFAULT_DELETE_BATCH = 1000
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
        """Block until the index reports ready, or raise on timeout.

        A created index that never becomes ready would otherwise accept writes
        that silently fail, so this fails loudly instead of returning.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.describe().get("ready"):
                    return
            except Exception:  # noqa: BLE001 - transient during provisioning
                pass
            time.sleep(1.0)
        raise VectorStoreError(
            f"Pinecone index '{self.index_name}' did not become ready within "
            f"{timeout:.0f} second(s)."
        )

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
        if batch_size <= 0:
            raise VectorStoreError(f"batch_size must be positive, got {batch_size}")
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

    # -- maintenance -------------------------------------------------------- #
    def list_vector_ids(self, page_size: int = 100) -> list[str]:
        """Return every vector id in the configured namespace.

        Pinecone paginates large namespaces. ``list`` returns a ``ListResponse``
        whose ``vectors`` items each carry an ``id``; older SDKs return a plain
        iterable. Both shapes are handled here.
        """
        ids: list[str] = []
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"namespace": self.namespace, "limit": page_size}
            if token:
                kwargs["pagination_token"] = token
            try:
                response = self.index.list(**kwargs)
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"Could not list vector ids: {exc}") from exc

            # Pinecone SDKs differ here: some return a ListResponse, some a
            # generator of ListResponse pages, and older ones a list of items.
            page_vectors = _attr(response, "vectors", None)
            if page_vectors is not None:
                pages = [response]
            elif isinstance(response, (list, tuple)):
                pages = list(response)
            elif hasattr(response, "__iter__"):
                pages = list(response)
            else:
                pages = [response]

            for page in pages:
                items = _attr(page, "vectors", None)
                if items is None:
                    items = [page]
                for item in items:
                    vector_id = _attr(item, "id", None)
                    if vector_id:
                        ids.append(str(vector_id))
                pagination = _attr(page, "pagination", None)
                token = _attr(pagination, "next", None) if pagination is not None else None

            if not token:
                return ids

    def delete_ids(self, ids: Sequence[str], batch_size: int = DEFAULT_DELETE_BATCH) -> int:
        """Delete specific vector ids from the namespace. Returns the count.

        Only used by the explicit ``--prune`` maintenance path; normal indexing
        never destroys data. Large deletions are split into requests because
        Pinecone caps the number of ids accepted per call.
        """
        if batch_size <= 0:
            raise VectorStoreError(f"batch_size must be positive, got {batch_size}")
        id_list = [str(vector_id) for vector_id in ids]
        if not id_list:
            return 0
        for start in range(0, len(id_list), batch_size):
            batch = id_list[start : start + batch_size]
            try:
                self.index.delete(ids=batch, namespace=self.namespace)
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"Could not delete stale vectors: {exc}") from exc
        return len(id_list)


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
