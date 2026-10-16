"""Integration check: index the target project into Pinecone and search it.

Run from the directory containing ``code_helper``::

    python scripts/index_and_search.py
    python scripts/index_and_search.py --query "where are imports parsed?"

This exercises the real pipeline end to end (stages 1-5) against live NVIDIA NIM
embeddings and Pinecone. It creates the index if missing but never deletes it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_helper.config import get_settings  # noqa: E402
from code_helper.rag_pipeline import index_repository  # noqa: E402
from code_helper.retrieval import retrieve  # noqa: E402
from code_helper.vector_store import PineconeVectorStore  # noqa: E402

DEFAULT_QUERIES = [
    "Where is the upsert to the vector database implemented?",
    "How are imports extracted with the ast module?",
    "Where are environment variables loaded?",
]


def main() -> int:
    """Index the project, print statistics, then answer sample queries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="", help="A single custom question.")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    print(f"Target project: {settings.target_repo_path}")

    report = index_repository(settings=settings, progress=args.progress)
    print(json.dumps(report.as_dict(), indent=2))

    store = PineconeVectorStore(settings)
    print("Index stats:", json.dumps(store.stats(), indent=2))

    queries = [args.query] if args.query.strip() else DEFAULT_QUERIES
    for query in queries:
        print("\n" + "=" * 70)
        print("QUERY:", query)
        results = retrieve(query, settings=settings)
        if not results:
            print("  (no matches)")
            continue
        for rank, result in enumerate(results, start=1):
            print(
                f"  [{rank}] {result.file_path} lines {result.line_range} score={result.score:.4f}"
            )
    print("\nINDEX + SEARCH CHECK COMPLETE")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
