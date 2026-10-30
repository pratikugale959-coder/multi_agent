"""Orchestration for the custom RAG pipeline.

This module wires the five stages together:

===  ==========================================  ==========================
#    Stage                                       Module
===  ==========================================  ==========================
1    Read source files                           :mod:`code_helper.rag.ingestion`
2    Split files into chunks                     :mod:`code_helper.rag.chunking`
3    Generate embeddings (NVIDIA NIM)            :mod:`code_helper.rag.embeddings`
4    Store vectors + metadata (Pinecone)         :mod:`code_helper.rag.vector_store`
5    Embed query and retrieve matches            :mod:`code_helper.rag.retrieval`
===  ==========================================  ==========================

Each stage lives in its own module; this file only coordinates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from code_helper.core.config import Settings, get_settings
from code_helper.rag.chunking import Chunk, chunk_source_files
from code_helper.rag.embeddings import EmbeddingClient
from code_helper.rag.ingestion import discover_source_files, read_source_files
from code_helper.rag.retrieval import RetrievedChunk, format_results, retrieve
from code_helper.rag.vector_store import PineconeVectorStore

DEFAULT_EMBED_BATCH = 32


@dataclass
class IndexReport:
    """Summary of one indexing run."""

    root: str
    files_discovered: int
    files_skipped: int
    chunks_created: int
    chunks_indexed: int
    chunks_pruned: int
    embedding_dimension: int
    index_name: str
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view of the report."""
        return {
            "root": self.root,
            "files_discovered": self.files_discovered,
            "files_skipped": self.files_skipped,
            "chunks_created": self.chunks_created,
            "chunks_indexed": self.chunks_indexed,
            "chunks_pruned": self.chunks_pruned,
            "embedding_dimension": self.embedding_dimension,
            "index_name": self.index_name,
            "errors": self.errors,
        }


def build_chunks(
    root: str | Path,
    settings: Settings | None = None,
) -> tuple[list[Chunk], int, int, list[str]]:
    """Stages 1-2: discover files and chunk them.

    Discovery happens exactly once: the same path list is counted and read.

    Returns:
        ``(chunks, files_discovered, files_skipped, errors)``.
    """
    settings = settings or get_settings()
    paths = discover_source_files(root)
    sources, skipped = read_source_files(paths, root)
    chunks = chunk_source_files(
        sources,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    errors: list[str] = []
    if skipped:
        errors.append(f"{skipped} file(s) skipped (binary, undecodable or too large)")
    return chunks, len(paths), skipped, errors


def index_repository(
    root: str | Path | None = None,
    settings: Settings | None = None,
    embedder: EmbeddingClient | None = None,
    store: PineconeVectorStore | None = None,
    embed_batch_size: int = DEFAULT_EMBED_BATCH,
    progress: bool = False,
    prune: bool = False,
) -> IndexReport:
    """Run stages 1-4 for every indexable file under *root*.

    Args:
        root: Directory to index (defaults to ``TARGET_REPO_PATH``).
        settings: Optional configuration override.
        embedder: Optional embedder override (tests inject a fake).
        store: Optional vector-store override (tests inject a fake).
        embed_batch_size: Documents per embedding request.
        progress: Print progress lines.
        prune: Delete vectors in the namespace that are not part of this run.
            Explicit only: normal indexing never destroys data.

    Returns:
        An :class:`IndexReport` describing what happened.
    """
    settings = settings or get_settings()
    root_path = Path(root).resolve() if root else settings.target_repo_path

    def log(message: str) -> None:
        if progress:
            print(message)

    log(f"[1/4] Reading source files under {root_path}")
    chunks, discovered, skipped, errors = build_chunks(root_path, settings)
    log(f"      {len(chunks)} chunk(s) created from readable files")

    report = IndexReport(
        root=str(root_path),
        files_discovered=discovered,
        files_skipped=skipped,
        chunks_created=len(chunks),
        chunks_indexed=0,
        chunks_pruned=0,
        embedding_dimension=settings.embedding_dimension,
        index_name=settings.pinecone_index_name,
        errors=errors,
    )

    if not chunks:
        report.errors.append("No indexable chunks were produced")
        return report

    log(f"[2/4] Chunking complete (size={settings.chunk_size}, overlap={settings.chunk_overlap})")

    embedder = embedder or EmbeddingClient(settings)
    texts = [chunk.embedding_text for chunk in chunks]
    log(f"[3/4] Embedding {len(texts)} chunk(s) with {settings.litellm_embed_model}")
    vectors = embedder.embed_documents(texts, batch_size=embed_batch_size)

    store = store or PineconeVectorStore(settings)
    description = store.ensure_index()
    log(f"[4/4] Upserting into index '{settings.pinecone_index_name}'")

    indexed = store.upsert_chunks(chunks, vectors)
    report.chunks_indexed = indexed
    if description.get("dimension"):
        report.embedding_dimension = int(description["dimension"])

    if prune:
        current_ids = {chunk.chunk_id for chunk in chunks}
        try:
            stale = sorted(set(store.list_vector_ids()) - current_ids)
            report.chunks_pruned = store.delete_ids(stale)
        except AttributeError:
            report.errors.append("The configured vector store does not support pruning")
        log(f"[prune] Removed {report.chunks_pruned} stale vector(s)")

    return report


def answer_question(
    query: str,
    top_k: int | None = None,
    settings: Settings | None = None,
    embedder: EmbeddingClient | None = None,
    store: PineconeVectorStore | None = None,
) -> dict[str, object]:
    """Retrieve evidence and return a grounded, formatted answer payload."""
    results: list[RetrievedChunk] = retrieve(
        query,
        top_k=top_k,
        settings=settings,
        embedder=embedder,
        store=store,
    )
    return {
        "query": query,
        "has_evidence": bool(results),
        "evidence": format_results(results),
        "source_files": sorted({result.file_path for result in results}),
        "results": [
            {
                "file": result.file_path,
                "line_start": result.line_start,
                "line_end": result.line_end,
                "score": round(result.score, 6),
                "text": result.text,
            }
            for result in results
        ],
    }
