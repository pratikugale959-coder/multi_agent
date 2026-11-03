"""Central configuration for the Multi-Agent Code Helper.

Why this module exists
----------------------
Every other module needs credentials and paths. If each module called
``os.getenv`` directly we would have ten places to change and no single place to
validate. Instead:

* ``load_environment()`` loads ``.env`` exactly once.
* :class:`Settings` is a typed, immutable snapshot of the configuration.
* :func:`get_settings` returns a cached instance so the whole programme shares it.
* :func:`get_chat_model` returns **one** LiteLLM chat-model object shared by all
  five agents (the assignment requires a single shared chat-model config).
* Embedding configuration stays separate from chat configuration, because the
  embedding model must never be swapped for the chat model by accident.

Unit tests can import this module without any API keys: nothing here performs a
network call. Keys are only *required* by the helpers that are about to talk to
an external service.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR: Path = Path(__file__).resolve().parents[1]
"""The ``code_helper`` package directory (this module lives in ``core/``)."""

PROJECT_ROOT: Path = PACKAGE_DIR.parent
"""The directory that contains ``code_helper/`` and ``target_project/``."""

DEFAULT_TARGET_REPO: Path = PROJECT_ROOT / "target_project"

# --------------------------------------------------------------------------- #
# Model identifiers (verified against the installed NVIDIA NIM / LiteLLM API)
# --------------------------------------------------------------------------- #
# LiteLLM routes NVIDIA NIM models through the ``nvidia_nim`` provider and reads
# the key from ``NVIDIA_NIM_API_KEY``. These defaults are deliberately
# overridable: NVIDIA retires hosted models without notice, so the exact chat
# model is configuration, not a hard-coded assumption. ``openai/gpt-oss-20b``
# was the fastest tool-capable text model available when this default was set.
CHAT_MODEL_ID: str = "openai/gpt-oss-20b"
EMBED_MODEL_ID: str = "nvidia/nemotron-3-embed-1b"
LITELLM_CHAT_MODEL: str = f"nvidia_nim/{CHAT_MODEL_ID}"
LITELLM_EMBED_MODEL: str = f"nvidia_nim/{EMBED_MODEL_ID}"

DEFAULT_NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
DEFAULT_PINECONE_INDEX: str = "code-helper"
DEFAULT_PINECONE_CLOUD: str = "aws"
DEFAULT_PINECONE_REGION: str = "us-east-1"
DEFAULT_EMBEDDING_DIMENSION: int = 2048
DEFAULT_CHUNK_SIZE: int = 100
DEFAULT_CHUNK_OVERLAP: int = 20
DEFAULT_TOP_K: int = 5
DEFAULT_MAX_LLM_CALLS: int = 10
DEFAULT_MAX_TOOL_CALLS: int = 3
DEFAULT_MAX_OUTPUT_TOKENS: int = 800

TIMEOUT_FALLBACK_TEXT = (
    "The model took too long to respond. Please retry, or ask a narrower "
    "question (for a file explanation, try a smaller file)."
)
DEFAULT_CHAT_TIMEOUT_SECONDS: int = 60
DEFAULT_REASONING_EFFORT: str = "low"

#: Models tried when the primary chat endpoint fails (timeout, connection
#: error or provider error). Both were verified with live NVIDIA NIM tool calls.
DEFAULT_FALLBACK_CHAT_MODEL_IDS: tuple[str, ...] = (
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3.5-bolt-e4b",
)
#: Fallback generation is allowed longer than the primary probe: the primary
#: is abandoned after ``NVIDIA_TIMEOUT_SECONDS`` so the user does not wait for a
#: dead endpoint, while a working fallback gets room to finish a real answer.
DEFAULT_FALLBACK_CHAT_TIMEOUT_SECONDS: int = 120
#: After the primary fails, send later calls straight to the fallback for this
#: long. Without a breaker every step of a multi-agent run would pay the primary
#: timeout again (routing, tool turn, final answer).
DEFAULT_PRIMARY_COOLDOWN_SECONDS: int = 300

#: Required credentials. Listed here so validation and error messages agree.
REQUIRED_NVIDIA_VARS = ("NVIDIA_API_KEY",)
REQUIRED_PINECONE_VARS = ("PINECONE_API_KEY",)

_ENV_LOADED = False


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


# --------------------------------------------------------------------------- #
# Environment loading
# --------------------------------------------------------------------------- #
def load_environment(force: bool = False) -> None:
    """Load ``.env`` files into ``os.environ`` (idempotent).

    Search order (first match wins, earlier files do not override later ones
    unless they use ``override=True``):

    1. ``PROJECT_ROOT/.env``
    2. ``PROJECT_ROOT/.env.local`` (overrides ``.env``)

    ``python-dotenv`` never overwrites variables that already exist in the real
    environment unless ``override=True``, which is what we want: a key exported
    in the shell should win over the file.
    """
    global _ENV_LOADED
    if _ENV_LOADED and not force:
        return

    env_file = PROJECT_ROOT / ".env"
    env_local = PROJECT_ROOT / ".env.local"

    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    if env_local.exists():
        load_dotenv(dotenv_path=env_local, override=True)

    # Fall back to a plain search from the current working directory so that
    # ``adk web`` launched from elsewhere still finds the file.
    load_dotenv()

    # LiteLLM's NVIDIA NIM provider looks for NVIDIA_NIM_API_KEY. Keep the two
    # names in sync, but prefer the documented NVIDIA_API_KEY so a placeholder
    # left in NVIDIA_NIM_API_KEY cannot override a real key.
    primary = os.getenv("NVIDIA_API_KEY", "").strip()
    nim = os.getenv("NVIDIA_NIM_API_KEY", "").strip()
    if primary:
        os.environ["NVIDIA_NIM_API_KEY"] = primary
    elif nim:
        os.environ["NVIDIA_API_KEY"] = nim

    _ENV_LOADED = True


def _as_int(name: str, default: int) -> int:
    """Read an integer environment variable, raising a clear error if malformed."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _as_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Read a comma-separated list, or disable it with ``none``/``off``."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() in {"none", "off", "disabled"}:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


# --------------------------------------------------------------------------- #
# Typed settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    nvidia_api_key: str
    nvidia_base_url: str
    chat_model_id: str
    embed_model_id: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_namespace: str
    target_repo_path: Path
    embedding_dimension: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    chat_timeout_seconds: int = DEFAULT_CHAT_TIMEOUT_SECONDS
    #: Models tried in order when the primary chat model times out or errors.
    fallback_chat_model_ids: tuple[str, ...] = DEFAULT_FALLBACK_CHAT_MODEL_IDS
    #: Per-attempt wall-clock bound for each fallback model.
    fallback_chat_timeout_seconds: int = DEFAULT_FALLBACK_CHAT_TIMEOUT_SECONDS
    #: How long the circuit breaker skips the primary after a failure.
    primary_cooldown_seconds: int = DEFAULT_PRIMARY_COOLDOWN_SECONDS
    #: Reasoning budget for models that support it (gpt-oss via NVIDIA NIM).
    #: ``"low"`` keeps answers concise and interactive; empty disables the hint.
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    #: Optional cross-encoder reranker. Empty means "use vector order as-is".
    rerank_model_id: str = ""

    # -- convenience -------------------------------------------------------- #
    @property
    def has_nvidia_credentials(self) -> bool:
        """True when an NVIDIA API key is present."""
        return bool(self.nvidia_api_key)

    @property
    def has_pinecone_credentials(self) -> bool:
        """True when a Pinecone API key is present."""
        return bool(self.pinecone_api_key)

    @property
    def litellm_chat_model(self) -> str:
        """Chat model id with the LiteLLM provider prefix, e.g. ``nvidia_nim/...``."""
        return f"nvidia_nim/{self.chat_model_id}"

    @property
    def litellm_embed_model(self) -> str:
        """Embedding model id with the LiteLLM provider prefix."""
        return f"nvidia_nim/{self.embed_model_id}"

    @property
    def litellm_fallback_chat_models(self) -> tuple[str, ...]:
        """Fallback chat model ids with the LiteLLM provider prefix."""
        return tuple(f"nvidia_nim/{model_id}" for model_id in self.fallback_chat_model_ids)


def _build_settings() -> Settings:
    """Construct :class:`Settings` from the environment (no validation of secrets)."""
    load_environment()

    target_raw = os.getenv("TARGET_REPO_PATH", "").strip()
    target = Path(target_raw).expanduser().resolve() if target_raw else DEFAULT_TARGET_REPO

    nvidia_key = (
        os.getenv("NVIDIA_API_KEY", "").strip() or os.getenv("NVIDIA_NIM_API_KEY", "").strip()
    )

    return Settings(
        nvidia_api_key=nvidia_key,
        nvidia_base_url=os.getenv("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL).strip(),
        chat_model_id=os.getenv("NVIDIA_CHAT_MODEL", CHAT_MODEL_ID).strip(),
        embed_model_id=os.getenv("NVIDIA_EMBED_MODEL", EMBED_MODEL_ID).strip(),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", "").strip(),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", DEFAULT_PINECONE_INDEX).strip(),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", DEFAULT_PINECONE_CLOUD).strip(),
        pinecone_region=os.getenv("PINECONE_REGION", DEFAULT_PINECONE_REGION).strip(),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "").strip(),
        target_repo_path=target,
        embedding_dimension=_as_int("EMBEDDING_DIMENSION", DEFAULT_EMBEDDING_DIMENSION),
        chunk_size=_as_int("CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
        chunk_overlap=_as_int("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
        top_k=_as_int("TOP_K", DEFAULT_TOP_K),
        max_llm_calls=_as_int("MAX_LLM_CALLS", DEFAULT_MAX_LLM_CALLS),
        max_tool_calls=_as_int("MAX_TOOL_CALLS", DEFAULT_MAX_TOOL_CALLS),
        max_output_tokens=_as_int("NVIDIA_MAX_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS),
        chat_timeout_seconds=_as_int("NVIDIA_TIMEOUT_SECONDS", DEFAULT_CHAT_TIMEOUT_SECONDS),
        fallback_chat_model_ids=_as_csv(
            "NVIDIA_FALLBACK_CHAT_MODELS", DEFAULT_FALLBACK_CHAT_MODEL_IDS
        ),
        fallback_chat_timeout_seconds=_as_int(
            "NVIDIA_FALLBACK_TIMEOUT_SECONDS", DEFAULT_FALLBACK_CHAT_TIMEOUT_SECONDS
        ),
        primary_cooldown_seconds=_as_int(
            "NVIDIA_PRIMARY_COOLDOWN_SECONDS", DEFAULT_PRIMARY_COOLDOWN_SECONDS
        ),
        reasoning_effort=os.getenv("NVIDIA_REASONING_EFFORT", DEFAULT_REASONING_EFFORT).strip(),
        rerank_model_id=os.getenv("NVIDIA_RERANK_MODEL", "").strip(),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached :class:`Settings` instance."""
    return _build_settings()


def reset_settings_cache() -> None:
    """Clear cached settings. Used by tests that monkey-patch the environment."""
    get_settings.cache_clear()
    global _ENV_LOADED
    _ENV_LOADED = False


# --------------------------------------------------------------------------- #
# Credential guards
# --------------------------------------------------------------------------- #
def require_nvidia_credentials(settings: Settings | None = None) -> str:
    """Return the NVIDIA API key or raise :class:`ConfigError`.

    Called immediately before an NVIDIA NIM network request so that a missing
    key produces a helpful message instead of an obscure SDK error.
    """
    settings = settings or get_settings()
    if not settings.nvidia_api_key:
        raise ConfigError(
            "NVIDIA API key is missing. Set NVIDIA_API_KEY (or NVIDIA_NIM_API_KEY) "
            "in your .env file or shell environment before calling NVIDIA NIM."
        )
    return settings.nvidia_api_key


def require_pinecone_credentials(settings: Settings | None = None) -> str:
    """Return the Pinecone API key or raise :class:`ConfigError`."""
    settings = settings or get_settings()
    if not settings.pinecone_api_key:
        raise ConfigError(
            "Pinecone API key is missing. Set PINECONE_API_KEY in your .env file "
            "or shell environment before using the vector store."
        )
    return settings.pinecone_api_key


def get_target_repo_path(settings: Settings | None = None) -> Path:
    """Return the absolute path of the project under analysis."""
    return (settings or get_settings()).target_repo_path


# --------------------------------------------------------------------------- #
# Shared chat model
# --------------------------------------------------------------------------- #
def _drop_thought_parts(response: Any) -> bool:
    """Remove ``thought=True`` parts from an LLM response, in place.

    Returns ``True`` when the response consisted only of thoughts and should
    therefore not be yielded at all. The Dev UI renders thought parts as
    visible ``THOUGHT`` blocks; the user asked to see only the final answer, so
    reasoning is removed at the shared model layer before ADK can aggregate it
    into an event.

    Thinking signatures are intentionally discarded with the parts: the model
    configured for this project (``openai/gpt-oss-20b`` through NVIDIA NIM)
    does not require them, and keeping them would mean keeping the thought text.
    """
    content = getattr(response, "content", None)
    parts = getattr(content, "parts", None)
    if not parts:
        return False

    visible = [part for part in parts if not getattr(part, "thought", False)]
    if len(visible) == len(parts):
        return False
    if not visible:
        return True
    content.parts = visible
    return False


def _litellm_model_kwargs(
    model_id: str,
    settings: Settings,
    *,
    timeout_seconds: int,
    include_reasoning_effort: bool,
) -> dict[str, object]:
    """Build LiteLlm kwargs for one model.

    The key and base URL are passed explicitly so behaviour is deterministic even
    if the environment is mutated later. ``num_retries=0`` prevents LiteLLM's
    hidden retries from multiplying the wall-clock bound.
    """
    kwargs: dict[str, object] = {"model": f"nvidia_nim/{model_id}"}
    if settings.nvidia_api_key:
        kwargs["api_key"] = settings.nvidia_api_key
    if settings.nvidia_base_url:
        kwargs["api_base"] = settings.nvidia_base_url
    kwargs["max_tokens"] = settings.max_output_tokens
    kwargs["timeout"] = timeout_seconds
    kwargs["num_retries"] = 0
    # gpt-oss exposes a reasoning budget. "low" keeps simple tool-routing turns
    # short. Other providers may reject the hint, so it is only sent to the
    # primary model (and only when configured).
    if include_reasoning_effort and settings.reasoning_effort:
        kwargs["extra_body"] = {"reasoning_effort": settings.reasoning_effort}
    return kwargs


@lru_cache(maxsize=1)
def get_chat_model():
    """Return the single LiteLLM chat model shared by all five agents.

    Importing :mod:`google.adk.models.lite_llm` is deferred so that unit tests
    which only exercise the RAG pipeline do not pay the import cost, and so this
    module stays importable in a minimal environment.

    The returned object is cached: every agent receives the exact same instance,
    satisfying the "share one chat-model configuration" requirement.

    The model is wrapped with two reliability guards because the hosted NVIDIA
    NIM endpoint intermittently stops responding:

    * Each attempt has a hard wall-clock bound applied with ``asyncio.timeout``.
      LiteLLM's own ``timeout`` is not reliable (a 60 s setting was observed to
      run for 183 s because of hidden retries).
    * If the primary times out or errors, the same request is retried on the
      configured fallback models. A circuit breaker then skips the primary for
      ``NVIDIA_PRIMARY_COOLDOWN_SECONDS`` so one dead endpoint does not add a
      minute to every step of a multi-agent run.

    The UI only shows the final answer, so the wrapper always asks for a complete
    (non-streaming) completion. NIM's streaming tool-call parser intermittently
    rejects gpt-oss deltas; a complete JSON body is parsed reliably.
    """
    import asyncio
    from collections.abc import AsyncGenerator

    from google.adk.models.lite_llm import LiteLlm
    from pydantic import PrivateAttr

    class ReliableLiteLlm(LiteLlm):
        """LiteLlm with a hard per-attempt bound and automatic failover."""

        _timeout_seconds: float = PrivateAttr(default=DEFAULT_CHAT_TIMEOUT_SECONDS)
        _fallback_timeout_seconds: float = PrivateAttr(
            default=DEFAULT_FALLBACK_CHAT_TIMEOUT_SECONDS
        )
        _primary_cooldown_seconds: float = PrivateAttr(default=DEFAULT_PRIMARY_COOLDOWN_SECONDS)
        _primary_unavailable_until: float = PrivateAttr(default=0.0)
        _fallback_models: list[Any] = PrivateAttr(default_factory=list)

        def __init__(
            self,
            *,
            timeout_seconds: float = DEFAULT_CHAT_TIMEOUT_SECONDS,
            fallback_timeout_seconds: float = DEFAULT_FALLBACK_CHAT_TIMEOUT_SECONDS,
            primary_cooldown_seconds: float = DEFAULT_PRIMARY_COOLDOWN_SECONDS,
            fallback_models: list[Any] | None = None,
            **kwargs: Any,
        ):
            super().__init__(**kwargs)
            self._timeout_seconds = float(timeout_seconds)
            self._fallback_timeout_seconds = float(fallback_timeout_seconds)
            self._primary_cooldown_seconds = float(primary_cooldown_seconds)
            self._primary_unavailable_until = 0.0
            self._fallback_models = list(fallback_models or [])

        def reset_health(self) -> None:
            """Clear the circuit breaker (used by tests and after outages)."""
            self._primary_unavailable_until = 0.0

        def _primary_is_cooling_down(self) -> bool:
            return bool(self._fallback_models) and (
                time.monotonic() < self._primary_unavailable_until
            )

        def _mark_primary_failed(self) -> None:
            if self._primary_cooldown_seconds > 0:
                self._primary_unavailable_until = time.monotonic() + self._primary_cooldown_seconds

        async def _collect(
            self, source: AsyncGenerator[Any, None], timeout_seconds: float
        ) -> list[Any]:
            """Drain one generation attempt under a single wall-clock deadline."""
            items: list[Any] = []
            try:
                async with asyncio.timeout(timeout_seconds):
                    async for item in source:
                        if _drop_thought_parts(item):
                            continue
                        items.append(item)
            finally:
                try:
                    await source.aclose()
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass
            return items

        async def generate_content_async(
            self, llm_request: Any, stream: bool = False
        ) -> AsyncGenerator[Any, None]:
            attempts: list[tuple[str, Any]] = []
            if not self._primary_is_cooling_down():
                attempts.append(("primary", None))
            attempts.extend(("fallback", model) for model in self._fallback_models)
            if not attempts:  # no fallback configured, or breaker with no fallback
                attempts.append(("primary", None))

            last_error: Exception | None = None
            for kind, fallback_model in attempts:
                if kind == "primary":
                    timeout_seconds = self._timeout_seconds
                    request = llm_request
                    source = super().generate_content_async(request, stream=False)
                else:
                    timeout_seconds = self._fallback_timeout_seconds
                    # LiteLlm prefers ``llm_request.model`` over its own model, so
                    # point the copy at the fallback. The original request is left
                    # untouched for a later retry.
                    request = llm_request.model_copy(deep=False)
                    request.model = fallback_model.model
                    source = fallback_model.generate_content_async(request, stream=False)

                try:
                    items = await self._collect(source, timeout_seconds)
                except TimeoutError:
                    last_error = TimeoutError(
                        f"{kind} model did not respond within {timeout_seconds:.0f}s"
                    )
                    if kind == "primary":
                        self._mark_primary_failed()
                        _log.warning(
                            "Primary chat model timed out after %.0fs; trying fallback.",
                            timeout_seconds,
                        )
                    else:
                        _log.warning(
                            "Fallback chat model %s timed out after %.0fs.",
                            getattr(fallback_model, "model", "?"),
                            timeout_seconds,
                        )
                    continue
                except Exception as exc:  # noqa: BLE001 - fail over on any provider error
                    last_error = exc
                    if kind == "primary":
                        self._mark_primary_failed()
                        _log.warning(
                            "Primary chat model failed (%s); trying fallback.",
                            type(exc).__name__,
                        )
                    else:
                        _log.warning(
                            "Fallback chat model %s failed (%s).",
                            getattr(fallback_model, "model", "?"),
                            type(exc).__name__,
                        )
                    continue

                if not items:
                    # A thought-only or empty response is not an answer; give the
                    # next model a chance instead of showing a blank message.
                    if kind == "primary":
                        self._mark_primary_failed()
                    continue

                if kind == "primary":
                    self._primary_unavailable_until = 0.0
                for item in items:
                    yield item
                return

            _log.error("All chat models failed; last error: %s", last_error)
            # A bounded failure still ends with a readable answer instead of a
            # traceback and an empty UI.
            from google.adk.models.llm_response import LlmResponse
            from google.genai import types

            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=TIMEOUT_FALLBACK_TEXT)],
                )
            )

    settings = get_settings()
    kwargs = _litellm_model_kwargs(
        settings.chat_model_id,
        settings,
        timeout_seconds=settings.chat_timeout_seconds,
        include_reasoning_effort=True,
    )

    fallback_models: list[Any] = []
    seen = {settings.chat_model_id}
    for model_id in settings.fallback_chat_model_ids:
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        fallback_models.append(
            LiteLlm(
                **_litellm_model_kwargs(
                    model_id,
                    settings,
                    timeout_seconds=settings.fallback_chat_timeout_seconds,
                    include_reasoning_effort=False,
                )
            )
        )

    return ReliableLiteLlm(
        timeout_seconds=settings.chat_timeout_seconds,
        fallback_timeout_seconds=settings.fallback_chat_timeout_seconds,
        primary_cooldown_seconds=settings.primary_cooldown_seconds,
        fallback_models=fallback_models,
        **kwargs,
    )


def describe_configuration() -> dict[str, object]:
    """Return a redacted view of the configuration (safe to print/log).

    Never includes secret values -- only booleans for presence.
    """
    settings = get_settings()
    return {
        "project_root": str(PROJECT_ROOT),
        "target_repo_path": str(settings.target_repo_path),
        "chat_model": settings.chat_model_id,
        "embed_model": settings.embed_model_id,
        "rerank_model": settings.rerank_model_id or None,
        "litellm_chat_model": settings.litellm_chat_model,
        "litellm_embed_model": settings.litellm_embed_model,
        "nvidia_base_url": settings.nvidia_base_url,
        "nvidia_key_present": settings.has_nvidia_credentials,
        "pinecone_index": settings.pinecone_index_name,
        "pinecone_cloud": settings.pinecone_cloud,
        "pinecone_region": settings.pinecone_region,
        "pinecone_key_present": settings.has_pinecone_credentials,
        "embedding_dimension": settings.embedding_dimension,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "max_llm_calls": settings.max_llm_calls,
        "max_tool_calls": settings.max_tool_calls,
        "max_output_tokens": settings.max_output_tokens,
        "chat_timeout_seconds": settings.chat_timeout_seconds,
        "fallback_chat_models": list(settings.fallback_chat_model_ids),
        "fallback_chat_timeout_seconds": settings.fallback_chat_timeout_seconds,
        "primary_cooldown_seconds": settings.primary_cooldown_seconds,
        "reasoning_effort": settings.reasoning_effort or None,
    }
