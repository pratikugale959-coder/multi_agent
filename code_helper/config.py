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

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR: Path = Path(__file__).resolve().parent
"""The ``code_helper`` package directory."""

PROJECT_ROOT: Path = PACKAGE_DIR.parent
"""The directory that contains ``code_helper/`` and ``target_project/``."""

DEFAULT_TARGET_REPO: Path = PROJECT_ROOT / "target_project"

# --------------------------------------------------------------------------- #
# Model identifiers (verified against the installed NVIDIA NIM / LiteLLM API)
# --------------------------------------------------------------------------- #
# LiteLLM routes NVIDIA NIM models through the ``nvidia_nim`` provider and reads
# the key from ``NVIDIA_NIM_API_KEY``. The assignment fixes these values.
CHAT_MODEL_ID: str = "meta/llama-3.2-11b-vision-instruct"
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
    # names in sync so either convention works.
    if not os.getenv("NVIDIA_NIM_API_KEY") and os.getenv("NVIDIA_API_KEY"):
        os.environ["NVIDIA_NIM_API_KEY"] = os.environ["NVIDIA_API_KEY"]

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


def _build_settings() -> Settings:
    """Construct :class:`Settings` from the environment (no validation of secrets)."""
    load_environment()

    target_raw = os.getenv("TARGET_REPO_PATH", "").strip()
    target = Path(target_raw).expanduser().resolve() if target_raw else DEFAULT_TARGET_REPO

    nvidia_key = (
        os.getenv("NVIDIA_NIM_API_KEY", "").strip() or os.getenv("NVIDIA_API_KEY", "").strip()
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
@lru_cache(maxsize=1)
def get_chat_model():
    """Return the single LiteLLM chat model shared by all five agents.

    Importing :mod:`google.adk.models.lite_llm` is deferred so that unit tests
    which only exercise the RAG pipeline do not pay the import cost, and so this
    module stays importable in a minimal environment.

    The returned object is cached: every agent receives the exact same instance,
    satisfying the "share one chat-model configuration" requirement.
    """
    from google.adk.models.lite_llm import LiteLlm

    settings = get_settings()
    # Pass the key/base explicitly. This keeps behaviour deterministic even if
    # the environment is mutated later, and avoids relying on implicit lookup.
    kwargs: dict[str, str] = {"model": settings.litellm_chat_model}
    if settings.nvidia_api_key:
        kwargs["api_key"] = settings.nvidia_api_key
    if settings.nvidia_base_url:
        kwargs["api_base"] = settings.nvidia_base_url
    return LiteLlm(**kwargs)


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
    }
