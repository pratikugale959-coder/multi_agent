"""Stage 3 of the RAG pipeline -- embeddings via NVIDIA NIM.

The assignment fixes the embedding model (``nvidia/nemotron-3-embed-1b``) and
requires the *same* model for documents and queries. LiteLLM routes the request
through its ``nvidia_nim`` provider.

Hard rules implemented here
---------------------------
* A missing API key raises :class:`~code_helper.config.ConfigError` *before* the
  network call.
* Transient failures (rate limit, timeout, 5xx) are retried with exponential
  backoff.
* Authentication and permission failures fail fast with a clear message.
* Malformed responses and unexpected vector counts raise :class:`EmbeddingError`.
* Dimensions are validated against the configured 2048 dimension.
* **No fabricated vectors.** There is no random/mock fallback anywhere in this
  module -- if the API cannot be reached we raise.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import litellm

from code_helper.config import (
    Settings,
    get_settings,
    require_nvidia_credentials,
)

#: HTTP statuses worth retrying.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

#: Substrings in an exception class name that indicate a transient failure.
_RETRYABLE_NAME_HINTS = (
    "ratelimit",
    "timeout",
    "connection",
    "serviceunavailable",
    "internalserver",
    "apierror",
)

#: Substrings indicating the credentials are wrong -- never retry these.
_AUTH_NAME_HINTS = ("authentication", "permissiondenied", "unauthorized")

DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_RETRIES = 4
DEFAULT_TIMEOUT_SECONDS = 60.0


class EmbeddingError(RuntimeError):
    """Raised when an embedding cannot be produced."""


def _status_code(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from an exception."""
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def _is_auth_error(exc: BaseException) -> bool:
    """True when *exc* looks like an authentication/permission problem."""
    if _status_code(exc) in (401, 403):
        return True
    name = type(exc).__name__.lower()
    return any(hint in name for hint in _AUTH_NAME_HINTS)


def _is_retryable(exc: BaseException) -> bool:
    """True when *exc* looks transient and worth retrying."""
    if _is_auth_error(exc):
        return False
    code = _status_code(exc)
    if code is not None:
        return code in _RETRYABLE_STATUS
    name = type(exc).__name__.lower()
    return any(hint in name for hint in _RETRYABLE_NAME_HINTS)


def _extract_vector(item: Any) -> list[float]:
    """Pull the embedding list out of a LiteLLM response item."""
    vector = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
    if not isinstance(vector, (list, tuple)) or len(vector) == 0:
        raise EmbeddingError(f"Malformed embedding payload: {type(item)!r}")
    try:
        return [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise EmbeddingError("Embedding contained non-numeric values") from exc


def _index_of(item: Any) -> int:
    """Return the ordering index of a response item."""
    value = item.get("index", 0) if isinstance(item, dict) else getattr(item, "index", 0)
    return int(value) if isinstance(value, int) else 0


class EmbeddingClient:
    """Thin wrapper around ``litellm.embedding`` for NVIDIA NIM.

    The instance is stateless apart from configuration, so it is safe to share.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.litellm_embed_model
        self.expected_dimension = self.settings.embedding_dimension

    # -- public API --------------------------------------------------------- #
    def embed_documents(
        self,
        texts: Sequence[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[list[float]]:
        """Embed many texts, returning one vector per input in the same order."""
        cleaned = self._clean_texts(texts)
        if batch_size <= 0:
            raise EmbeddingError(f"batch_size must be positive, got {batch_size}")

        vectors: list[list[float]] = []
        for start in range(0, len(cleaned), batch_size):
            batch = cleaned[start : start + batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query.

        Raises:
            ValueError: If *text* is empty or whitespace.
        """
        if not isinstance(text, str) or text.strip() == "":
            raise ValueError("Query text must be a non-empty string")
        return self._embed_batch([text])[0]

    def probe_dimension(self, sample: str = "dimension probe") -> int:
        """Return the actual embedding dimension from a single live call.

        Useful for verifying that the live model really returns 2048 values.
        """
        return len(self._embed_batch([sample])[0])

    # -- internals ---------------------------------------------------------- #
    @staticmethod
    def _clean_texts(texts: Sequence[str]) -> list[str]:
        cleaned: list[str] = []
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise EmbeddingError(f"Text at position {index} is not a string")
            if text.strip() == "":
                raise EmbeddingError(f"Text at position {index} is empty or whitespace")
            cleaned.append(text)
        return cleaned

    def _embed_batch(
        self,
        batch: Sequence[str],
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> list[list[float]]:
        """Embed one batch with retry/backoff. Never fabricates a vector."""
        api_key = require_nvidia_credentials(self.settings)

        last_error: BaseException | None = None
        for attempt in range(max_retries + 1):
            try:
                response = litellm.embedding(
                    model=self.model,
                    input=list(batch),
                    api_key=api_key,
                    api_base=self.settings.nvidia_base_url,
                    timeout=timeout,
                    num_retries=0,  # we control retries ourselves
                )
                return self._parse_response(response, expected_count=len(batch))
            except EmbeddingError:
                raise
            except Exception as exc:  # noqa: BLE001 - we classify below
                if _is_auth_error(exc):
                    raise EmbeddingError(
                        "NVIDIA NIM rejected the credentials (authentication/permission "
                        f"error). Check NVIDIA_API_KEY / NVIDIA_NIM_API_KEY. Details: {exc}"
                    ) from exc
                last_error = exc
                if attempt >= max_retries or not _is_retryable(exc):
                    break
                # Exponential backoff: 0.5s, 1s, 2s, 4s ...
                time.sleep(0.5 * (2**attempt))

        raise EmbeddingError(
            f"Embedding request failed after {max_retries + 1} attempt(s): {last_error}"
        )

    def _parse_response(self, response: Any, expected_count: int) -> list[list[float]]:
        """Validate and normalise a LiteLLM embedding response."""
        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data")
        if not isinstance(data, (list, tuple)) or len(data) == 0:
            raise EmbeddingError("Embedding response contained no data")

        ordered = sorted(data, key=_index_of)
        vectors = [_extract_vector(item) for item in ordered]

        if len(vectors) != expected_count:
            raise EmbeddingError(
                f"Embedding response count mismatch: expected {expected_count}, got {len(vectors)}"
            )

        for vector in vectors:
            if self.expected_dimension and len(vector) != self.expected_dimension:
                raise EmbeddingError(
                    "Embedding dimension mismatch: expected "
                    f"{self.expected_dimension}, got {len(vector)}. Update "
                    "EMBEDDING_DIMENSION and recreate the Pinecone index if the "
                    "model changed."
                )
        return vectors


# --------------------------------------------------------------------------- #
# Module-level convenience functions (used by tools and the pipeline)
# --------------------------------------------------------------------------- #
def embed_documents(texts: Sequence[str], settings: Settings | None = None) -> list[list[float]]:
    """Embed a sequence of documents using the shared embedding configuration."""
    return EmbeddingClient(settings).embed_documents(texts)


def embed_query(text: str, settings: Settings | None = None) -> list[float]:
    """Embed a single query using the **same** model as document embedding."""
    return EmbeddingClient(settings).embed_query(text)
