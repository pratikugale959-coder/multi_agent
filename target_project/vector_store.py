"""vector_store.py -- Vector database management with a local fallback.

Wraps Pinecone Serverless for storage and similarity search. When no Pinecone key
is configured the module transparently falls back to a local JSON store so demos
never fail. Both branches expose the same interface: upsert_to_vector_db() and
query_vector_db().
"""

import json
import math
import os
from pathlib import Path

from config import RAGConfig
from pinecone import Pinecone, ServerlessSpec

LOCAL_DB_FILE = Path(__file__).resolve().parent / "data" / "local_vector_db.json"


def calculate_cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Returns 0.0 when the vectors differ in length or either norm is zero.
    """
    if len(vec_a) != len(vec_b):
        return 0.0

    dot_product = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0

    for index in range(len(vec_a)):
        a_val = vec_a[index]
        b_val = vec_b[index]
        dot_product += a_val * b_val
        norm_a_sq += a_val * a_val
        norm_b_sq += b_val * b_val

    if norm_a_sq <= 0.0 or norm_b_sq <= 0.0:
        return 0.0

    return dot_product / (math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq))


def get_pinecone_client(api_key: str) -> Pinecone:
    """Instantiate and return the Pinecone client."""
    return Pinecone(api_key=api_key)


def ensure_pinecone_index_exists(
    client: Pinecone,
    index_name: str,
    dimension: int = 2048,
    cloud: str = "aws",
    region: str = "us-east-1",
) -> None:
    """Create the serverless index if it does not already exist."""
    index_names = [item.name for item in client.list_indexes()]

    if index_name not in index_names:
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"[OK] Pinecone index '{index_name}' created (dimension={dimension}).")
    else:
        print(f"[OK] Using existing Pinecone index '{index_name}'.")


def upsert_to_vector_db(chunks_with_embeddings: list, config: RAGConfig) -> int:
    """Store embeddings plus metadata into Pinecone, or the local JSON fallback."""
    count = len(chunks_with_embeddings)
    if count == 0:
        return 0

    if not config.use_local_vector_db:
        try:
            client = get_pinecone_client(config.pinecone_api_key)
            dimension = len(chunks_with_embeddings[0]["embedding"])
            ensure_pinecone_index_exists(
                client=client,
                index_name=config.pinecone_index_name,
                dimension=dimension,
                cloud=config.pinecone_cloud,
                region=config.pinecone_region,
            )
            index = client.Index(config.pinecone_index_name)

            vectors_payload = []
            for item in chunks_with_embeddings:
                vectors_payload.append(
                    {
                        "id": item["id"],
                        "values": item["embedding"],
                        "metadata": {"text": item["text"], "chunk_id": item["id"]},
                    }
                )

            index.upsert(vectors=vectors_payload)
            print(f"[OK] Upserted {count} vectors to Pinecone.")
            return count
        except Exception as error:
            print(f"[!] Pinecone upsert failed ({error}). Using local fallback.")

    LOCAL_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for item in chunks_with_embeddings:
        records.append(
            {
                "id": item["id"],
                "embedding": item["embedding"],
                "metadata": {"text": item["text"], "chunk_id": item["id"]},
            }
        )

    with open(LOCAL_DB_FILE, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    print(f"[OK] Saved {count} vectors to the local store {LOCAL_DB_FILE.name}.")
    return count


def query_vector_db(query_vector: list, top_k: int, config: RAGConfig) -> list:
    """Return the top-k most similar chunks for a query vector."""
    results = []

    if not config.use_local_vector_db:
        try:
            client = get_pinecone_client(config.pinecone_api_key)
            index = client.Index(config.pinecone_index_name)
            response = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

            for match in response.matches:
                metadata = match.metadata or {}
                results.append(
                    {
                        "id": match.id,
                        "score": float(match.score),
                        "text": metadata.get("text", ""),
                    }
                )
            return results
        except Exception as error:
            print(f"[!] Pinecone query failed ({error}). Using local fallback.")

    if not LOCAL_DB_FILE.exists():
        print(f"[!] Local vector store {LOCAL_DB_FILE} not found. Run the indexer first.")
        return []

    with open(LOCAL_DB_FILE, encoding="utf-8") as handle:
        stored_records = json.load(handle)

    scored_items = []
    for record in stored_records:
        similarity = calculate_cosine_similarity(query_vector, record["embedding"])
        scored_items.append(
            {"id": record["id"], "score": similarity, "text": record["metadata"]["text"]}
        )

    scored_items.sort(key=lambda item: item["score"], reverse=True)
    return scored_items[:top_k]


def clear_local_store() -> None:
    """Delete the local JSON fallback file if it exists."""
    if LOCAL_DB_FILE.exists():
        os.remove(LOCAL_DB_FILE)
        print("[OK] Local vector store cleared.")
