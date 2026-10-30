"""Custom RAG pipeline: ingestion, chunking, embeddings, storage, retrieval.

The stages are importable individually, but most callers only need the small
public API re-exported here.
"""

from code_helper.rag.chunking import Chunk, ChunkingError, chunk_code, chunk_source_files
from code_helper.rag.ingestion import (
    SourceFile,
    discover_source_files,
    load_source_files,
    read_source_file,
)
from code_helper.rag.pipeline import IndexReport, answer_question, build_chunks, index_repository
from code_helper.rag.retrieval import RetrievedChunk, format_results, retrieve
from code_helper.rag.vector_store import PineconeVectorStore, VectorStoreError

__all__ = [
    "Chunk",
    "ChunkingError",
    "IndexReport",
    "PineconeVectorStore",
    "RetrievedChunk",
    "SourceFile",
    "VectorStoreError",
    "answer_question",
    "build_chunks",
    "chunk_code",
    "chunk_source_files",
    "discover_source_files",
    "format_results",
    "index_repository",
    "load_source_files",
    "read_source_file",
    "retrieve",
]
