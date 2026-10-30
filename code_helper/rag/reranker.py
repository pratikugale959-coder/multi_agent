"""Reranking via NVIDIA Nemotron-3.5-Rerank-8B.

After the fast vector-similarity retrieval stage returns a broad candidate set
(``TOP_K`` chunks), this module passes those candidates to the cross-encoder
reranker which scores every (query, passage) pair jointly and returns them in
true relevance order.

Why a separate module
----------------------
Reranking uses a completely different NVIDIA endpoint
(``ai.api.nvidia.com/v1/retrieval/{model}/reranking``) and a different request
schema (``query`` / ``passages`` rather than the OpenAI embedding format), so
it cannot be wrapped by LiteLLM like the chat and embedding calls. A thin
``requests`` call is the cleanest, most transparent implementation.

Failure behaviour
-----------------
* A missing API key raises :class:`~code_helper.core.config.ConfigError` before any
  network call.
* HTTP errors and connection problems raise :class:`RerankError`.
* When the reranker is disabled (``NVIDIA_RERANK_MODEL`` is unset or empty)
  :func:`rerank` returns the input list unchanged so callers need no
  conditional logic.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from code_helper.core.config import Settings, get_settings, require_nvidia_credentials
from code_helper.rag.retrieval import RetrievedChunk

_log = logging.getLogger(__name__)

#: The NVIDIA retrieval endpoint base URL (different from the chat/embed base).
_RERANK_BASE = "https://ai.api.nvidia.com/v1/retrieval"

#: Seconds to wait for the reranking API before giving up.
_TIMEOUT = 8


class RerankError(RuntimeError):
    """Raised when reranking fails (network error, bad response, etc.)."""


def _rerank_endpoint(model: str) -> str:
    """Return the full reranking URL for *model*."""
    return f"{_RERANK_BASE}/{model}/reranking"


def _call_rerank_api(
    query: str,
    passages: list[str],
    model: str,
    api_key: str,
) -> list[dict[str, Any]]:
    """POST to the NVIDIA reranking endpoint and return the rankings list.

    Args:
        query: The user's question.
        passages: Text of each candidate chunk, in the same order as the
            caller's ``chunks`` list.
        model: NVIDIA model identifier, e.g. ``nvidia/nemotron-3.5-rerank-8b``.
        api_key: NVIDIA API key (``nvapi-...``).

    Returns:
        A list of ranking dicts, each with at least ``"index"`` (int) and
        ``"logit"`` (float) keys, sorted by relevance descending.

    Raises:
        RerankError: On HTTP error or unexpected response shape.
    """
    url = _rerank_endpoint(model)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "query": {"text": query},
        "passages": [{"text": p} for p in passages],
        "truncate": "END",
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        raise RerankError(f"Network error calling reranker: {exc}") from exc

    if not response.ok:
        raise RerankError(f"Reranker returned HTTP {response.status_code}: {response.text[:400]}")

    try:
        data = response.json()
    except ValueError as exc:
        raise RerankError(f"Reranker response was not valid JSON: {exc}") from exc

    rankings = data.get("rankings")
    if not isinstance(rankings, list):
        raise RerankError(f"Unexpected reranker response shape (no 'rankings' list): {data!r}")
    return rankings


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Re-order *chunks* by relevance to *query* using the NVIDIA reranker.

    When ``NVIDIA_RERANK_MODEL`` is not set the function returns *chunks*
    unchanged so the rest of the pipeline requires no conditional logic.

    Args:
        query: The natural-language question used to rank passages.
        chunks: Candidate chunks from the vector-similarity retrieval stage.
        top_n: If given, only this many chunks are returned after reranking.
            ``None`` returns all chunks in reranked order.
        settings: Optional configuration override (useful in tests).

    Returns:
        Chunks in reranked order (best first). Scores are replaced with the
        logit returned by the reranker so downstream formatters display
        meaningful values.

    Raises:
        ConfigError: If the API key is missing when the model is configured.
        RerankError: On API / network failure.
    """
    settings = settings or get_settings()
    model = settings.rerank_model_id

    if not model:
        _log.debug("Reranking disabled (NVIDIA_RERANK_MODEL not set); returning as-is.")
        return chunks[:top_n] if top_n is not None else chunks

    if not chunks:
        return chunks

    api_key = require_nvidia_credentials(settings)

    passages = [chunk.text for chunk in chunks]
    _log.debug("Calling reranker %s with %d passages.", model, len(passages))

    rankings = _call_rerank_api(query, passages, model, api_key)

    # Build index → logit mapping from the rankings list.
    score_map: dict[int, float] = {}
    for entry in rankings:
        idx = entry.get("index")
        logit = entry.get("logit", entry.get("score", 0.0))
        if isinstance(idx, int):
            score_map[idx] = float(logit)

    # Attach reranker scores to chunks and sort best-first.
    scored: list[tuple[float, RetrievedChunk]] = []
    for i, chunk in enumerate(chunks):
        new_score = score_map.get(i, chunk.score)
        # Replace the similarity score with the reranker logit.
        reranked_chunk = RetrievedChunk(
            chunk_id=chunk.chunk_id,
            file_path=chunk.file_path,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            score=new_score,
            text=chunk.text,
        )
        scored.append((new_score, reranked_chunk))

    scored.sort(key=lambda t: t[0], reverse=True)
    reranked = [chunk for _, chunk in scored]

    if top_n is not None:
        reranked = reranked[:top_n]

    _log.debug("Reranking complete. Returning %d chunk(s).", len(reranked))
    return reranked
