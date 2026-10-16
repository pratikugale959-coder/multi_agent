"""config.py -- Centralized configuration for the Naive RAG pipeline.

Loads NVIDIA NIM credentials and Pinecone settings from the environment, exposes
them as a typed dataclass, and reports whether mock or local fallbacks are needed.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_CHAT_MODEL = "meta/llama-3.2-11b-vision-instruct"
DEFAULT_EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


@dataclass
class RAGConfig:
    """Typed settings container for the whole pipeline."""

    nvidia_api_key: str
    nvidia_base_url: str
    chat_model: str
    embed_model: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    use_mock_ai: bool
    use_local_vector_db: bool


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_rag_config(force_mock_ai: bool = False, force_local_db: bool = False) -> RAGConfig:
    """Build a validated RAGConfig from the environment.

    Falls back to mock AI / local vector store when credentials are missing so a
    classroom demo can still run offline.
    """
    nvidia_key = (
        os.getenv("NVIDIA_API_KEY", "").strip() or os.getenv("NVIDIA_NIM_API_KEY", "").strip()
    )
    pinecone_key = os.getenv("PINECONE_API_KEY", "").strip()

    return RAGConfig(
        nvidia_api_key=nvidia_key,
        nvidia_base_url=os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL).strip(),
        chat_model=os.getenv("NVIDIA_CHAT_MODEL", DEFAULT_CHAT_MODEL).strip(),
        embed_model=os.getenv("NVIDIA_EMBED_MODEL", DEFAULT_EMBED_MODEL).strip(),
        pinecone_api_key=pinecone_key,
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "naive-rag-demo").strip(),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip(),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").strip(),
        chunk_size=_env_int("CHUNK_SIZE", 350),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 50),
        top_k=_env_int("TOP_K", 3),
        use_mock_ai=force_mock_ai or len(nvidia_key) == 0,
        use_local_vector_db=force_local_db or len(pinecone_key) == 0,
    )


def get_document_path(filename: str = "astratech_employee_handbook.txt") -> Path:
    """Return the absolute path of a document inside the local data directory."""
    return Path(__file__).resolve().parent / "data" / filename
