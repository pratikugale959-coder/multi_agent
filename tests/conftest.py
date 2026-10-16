"""Shared test fixtures.

Every test here is deterministic and offline. External services (LiteLLM,
Pinecone) are replaced by lightweight fakes, so the suite runs without API keys
or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code_helper.config import Settings  # noqa: E402


def make_settings(target_repo: Path) -> Settings:
    """Build a :class:`Settings` object pointing at *target_repo* with fake keys."""
    return Settings(
        nvidia_api_key="test-nvidia-key",
        nvidia_base_url="https://example.invalid/v1",
        chat_model_id="meta/llama-3.2-11b-vision-instruct",
        embed_model_id="nvidia/nemotron-3-embed-1b",
        pinecone_api_key="test-pinecone-key",
        pinecone_index_name="code-helper",
        pinecone_cloud="aws",
        pinecone_region="us-east-1",
        pinecone_namespace="test-ns",
        target_repo_path=Path(target_repo).resolve(),
        embedding_dimension=2048,
        chunk_size=100,
        chunk_overlap=20,
        top_k=5,
    )


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """Create a small project tree with python files and files that must be ignored."""
    root = tmp_path / "proj"
    root.mkdir()

    (root / "alpha.py").write_text(
        "import os\n\n\ndef hello():\n    return 'hi'\n", encoding="utf-8"
    )
    (root / "beta.py").write_text(
        "from alpha import hello\n\n\nclass Widget:\n    pass\n", encoding="utf-8"
    )
    # A file long enough to require two chunks (size 100, overlap 20 -> step 80).
    (root / "big.py").write_text(
        "\n".join(f"line_{i} = {i}" for i in range(1, 172)), encoding="utf-8"
    )

    # Things that must never be indexed.
    (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (root / "broken.py").write_text("def oops(:\n    pass\n", encoding="utf-8")
    (root / "binary.py").write_bytes(b"\x00\x01\x02binary")
    (root / "notes.txt").write_text("not python", encoding="utf-8")

    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "hidden.py").write_text("x = 1\n", encoding="utf-8")

    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "cached.py").write_text("y = 2\n", encoding="utf-8")

    venv = root / ".venv"
    venv.mkdir()
    (venv / "lib.py").write_text("z = 3\n", encoding="utf-8")

    (root / "README.md").write_text(
        "# Demo Project\n\nThis project demonstrates a tiny pipeline.\n\n"
        "## Usage\n\n```bash\npip install -r requirements.txt\npython main.py\n```\n",
        encoding="utf-8",
    )
    (root / "requirements.txt").write_text("requests>=2\n", encoding="utf-8")
    return root


@pytest.fixture
def settings(sample_project: Path) -> Settings:
    """Settings pointing at the sample project with dummy credentials."""
    return make_settings(sample_project)


class FakeEmbedder:
    """In-memory embedder that returns fixed-width vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.document_batches: list[list[str]] = []
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Query text must be a non-empty string")
        self.queries.append(text)
        return [0.1] * self.dimension

    def embed_documents(self, texts, batch_size: int = 32) -> list[list[float]]:
        texts = list(texts)
        self.document_batches.append(texts)
        return [[0.2] * self.dimension for _ in texts]


class FakeVectorStore:
    """In-memory vector store capturing upserts and returning canned matches."""

    def __init__(self, matches: list[dict] | None = None, dimension: int = 4) -> None:
        self.matches = matches or []
        self.dimension = dimension
        self.upserts: list[tuple[list, list]] = []
        self.ensure_called = False

    def ensure_index(self) -> dict:
        self.ensure_called = True
        return {
            "name": "code-helper",
            "dimension": self.dimension,
            "metric": "cosine",
            "ready": True,
        }

    def upsert_chunks(self, chunks, vectors, batch_size: int = 100) -> int:
        chunks = list(chunks)
        vectors = list(vectors)
        assert len(chunks) == len(vectors), "chunks and vectors must align"
        self.upserts.append((chunks, vectors))
        return len(chunks)

    def query(self, vector, top_k: int = 5, filter=None, include_metadata: bool = True):
        return self.matches[:top_k]

    def stats(self) -> dict:
        return {"total_vector_count": len(self.matches), "dimension": self.dimension}


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_store() -> FakeVectorStore:
    return FakeVectorStore()


#: Tool modules that resolve settings through a module-level ``get_settings``.
TOOL_MODULES = (
    "code_helper.tools.search_tools",
    "code_helper.tools.explain_tools",
    "code_helper.tools.architecture_tools",
    "code_helper.tools.onboarding_tools",
)


@pytest.fixture
def patch_tool_settings(monkeypatch: pytest.MonkeyPatch):
    """Return a function that points every tool module at given :class:`Settings`."""
    import importlib

    modules = [importlib.import_module(name) for name in TOOL_MODULES]

    def _apply(new_settings: Settings) -> Settings:
        for module in modules:
            monkeypatch.setattr(module, "get_settings", lambda: new_settings)
        return new_settings

    return _apply
