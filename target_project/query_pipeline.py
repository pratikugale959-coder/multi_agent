"""query_pipeline.py -- Online query pipeline (stages 5-8).

Embeds a user question, retrieves the top-k chunks, injects them into a grounded
prompt, and asks the LLM for an answer that cites only the retrieved context.
"""

from config import RAGConfig, get_rag_config
from vector_store import query_vector_db

GROUNDING_INSTRUCTION = (
    "Answer the user's question using ONLY the reference context below. "
    "If the answer is not in the context, say you do not have enough information. "
    "Never speculate."
)


def embed_query(query: str, config: RAGConfig) -> list:
    """Embed the user query with the same model used for documents."""
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("Query cannot be empty")

    if config.use_mock_ai:
        raise RuntimeError("NVIDIA credentials are required to embed a query")

    from openai import OpenAI

    client = OpenAI(base_url=config.nvidia_base_url, api_key=config.nvidia_api_key)
    response = client.embeddings.create(input=[cleaned], model=config.embed_model)
    return response.data[0].embedding


def retrieve_top_k(query_vector: list, config: RAGConfig) -> list:
    """Retrieve the top-k chunks most similar to the query vector."""
    matches = query_vector_db(query_vector, config.top_k, config)
    if not matches:
        print("[!] No matching chunks. Did you run the indexer first?")
    return matches


def build_prompt_with_context(query: str, retrieved_chunks: list) -> str:
    """Assemble the augmented prompt from retrieved context and the question."""
    blocks = []
    for number, chunk in enumerate(retrieved_chunks, start=1):
        blocks.append(f"--- CONTEXT {number} (id={chunk['id']}, score={chunk['score']:.4f}) ---")
        blocks.append(chunk["text"])

    context = "\n".join(blocks)
    return (
        f"{GROUNDING_INSTRUCTION}\n\n"
        f"=== BEGIN CONTEXT ===\n{context}\n=== END CONTEXT ===\n\n"
        f"QUESTION: {query}\n\nGROUNDED ANSWER:"
    )


def generate_answer(prompt: str, config: RAGConfig) -> str:
    """Send the augmented prompt to the chat model and return its answer."""
    if config.use_mock_ai:
        raise RuntimeError("NVIDIA credentials are required to generate an answer")

    from openai import OpenAI

    client = OpenAI(base_url=config.nvidia_base_url, api_key=config.nvidia_api_key)
    response = client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise assistant grounded in provided context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=300,
    )
    return (response.choices[0].message.content or "").strip()


def run_query_pipeline(query: str, config: RAGConfig = None) -> dict:
    """Run the full query pipeline and return the answer with its sources."""
    if config is None:
        config = get_rag_config()

    query_vector = embed_query(query, config)
    retrieved = retrieve_top_k(query_vector, config)
    prompt = build_prompt_with_context(query, retrieved)
    answer = generate_answer(prompt, config)

    return {
        "query": query,
        "retrieved_chunks": retrieved,
        "augmented_prompt": prompt,
        "answer": answer,
    }
