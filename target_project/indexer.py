"""indexer.py -- Offline indexing pipeline (stages 1-4).

Reads a source document, splits it into overlapping chunks, embeds each chunk via
NVIDIA NIM, and stores the vectors in Pinecone (or the local fallback).
"""

from pathlib import Path

from config import RAGConfig, get_rag_config
from vector_store import upsert_to_vector_db


def load_document(file_path: str) -> str:
    """Read and clean the raw text of a source document."""
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    with open(path_obj, encoding="utf-8") as handle:
        text = handle.read()

    cleaned = text.strip()
    print(f"[OK] Loaded {len(cleaned)} characters from {path_obj.name}.")
    return cleaned


def chunk_document(text: str, chunk_size: int = 350, chunk_overlap: int = 50) -> list:
    """Split text into overlapping character windows.

    Overlap preserves context that would otherwise be cut at a chunk boundary.
    """
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    step = chunk_size - chunk_overlap
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        window = text[start : start + chunk_size].strip()
        if window:
            chunks.append(
                {
                    "id": f"doc_chunk_{index}",
                    "text": window,
                    "start_char": start,
                    "end_char": start + len(window),
                }
            )
            index += 1
        start += step

    print(f"[OK] Created {len(chunks)} chunk(s).")
    return chunks


def embed_chunks(chunks: list, config: RAGConfig) -> list:
    """Embed every chunk with the NVIDIA NIM embedding model."""
    if config.use_mock_ai:
        raise RuntimeError("NVIDIA credentials are required to embed chunks")

    from openai import OpenAI

    client = OpenAI(base_url=config.nvidia_base_url, api_key=config.nvidia_api_key)
    texts = [chunk["text"] for chunk in chunks]
    response = client.embeddings.create(input=texts, model=config.embed_model)

    embedded = []
    for index, item in enumerate(response.data):
        embedded.append(
            {
                "id": chunks[index]["id"],
                "text": chunks[index]["text"],
                "embedding": item.embedding,
            }
        )
    print(f"[OK] Embedded {len(embedded)} chunk(s).")
    return embedded


def run_indexing_pipeline(file_path: str, config: RAGConfig = None) -> list:
    """Run the full indexing pipeline: ingest, chunk, embed, store."""
    if config is None:
        config = get_rag_config()

    document = load_document(file_path)
    chunks = chunk_document(document, config.chunk_size, config.chunk_overlap)
    embedded_chunks = embed_chunks(chunks, config)
    upsert_to_vector_db(embedded_chunks, config)
    return embedded_chunks
