# Naive RAG Demo

A minimal, self-contained Retrieval-Augmented Generation pipeline used as the
target project for the Multi-Agent Code Helper. It ingests a text document,
chunks it, embeds the chunks with NVIDIA NIM, stores the vectors in Pinecone
(with a local JSON fallback), and answers questions grounded in the retrieved
passages.

## Installation

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials, then export them.

## Usage

Run the full demo (index the sample document, then answer the canonical question):

```bash
python main.py
```

Run only the indexing pipeline:

```bash
python main.py --index
```

Ask a single custom question:

```bash
python main.py --query "How much is the remote home office setup stipend?"
```

## Files

- `config.py` -- typed configuration loaded from the environment.
- `indexer.py` -- stages 1-4: ingest, chunk, embed, store.
- `vector_store.py` -- Pinecone access with a local fallback.
- `query_pipeline.py` -- stages 5-8: embed query, retrieve, augment, generate.
- `main.py` -- command-line runner.
