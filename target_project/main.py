"""main.py -- Command-line runner for the Naive RAG demo.

Usage:
    python main.py                 # run the full demo (index + query)
    python main.py --index         # indexing pipeline only
    python main.py --query "..."   # query pipeline only
    python main.py --compare       # raw LLM vs RAG side by side
"""

import argparse
from pathlib import Path

from config import get_rag_config
from indexer import run_indexing_pipeline
from query_pipeline import run_query_pipeline

CANONICAL_QUESTION = "How much is the one-time remote home office setup stipend?"


def print_banner(title: str) -> None:
    """Print a visible terminal header."""
    bar = "=" * 70
    print(f"\n{bar}\n  {title.upper()}\n{bar}\n")


def run_demo(config) -> None:
    """Index the sample document, then answer the canonical question with RAG."""
    print_banner("Naive RAG demo")
    document = Path(__file__).parent / "data" / "astratech_employee_handbook.txt"
    if not document.exists():
        print(f"[!] Document not found: {document}")
        return

    run_indexing_pipeline(str(document), config)
    result = run_query_pipeline(CANONICAL_QUESTION, config)
    print(f"\nQuestion: {CANONICAL_QUESTION}")
    print(f"Answer: {result['answer']}")


def main() -> None:
    """Parse arguments and dispatch to the chosen routine."""
    parser = argparse.ArgumentParser(description="Naive RAG demo (NVIDIA NIM + Pinecone)")
    parser.add_argument("--index", action="store_true", help="Run indexing only")
    parser.add_argument("--query", type=str, default="", help="Run a single query")
    parser.add_argument("--compare", action="store_true", help="Compare raw LLM vs RAG")
    parser.add_argument("--mock", action="store_true", help="Force offline mode")
    args = parser.parse_args()

    config = get_rag_config(force_mock_ai=args.mock)

    if args.index:
        print_banner("Indexing only")
        document = Path(__file__).parent / "data" / "astratech_employee_handbook.txt"
        run_indexing_pipeline(str(document), config)
        return

    if args.query.strip():
        result = run_query_pipeline(args.query.strip(), config)
        print(result["answer"])
        return

    run_demo(config)


if __name__ == "__main__":
    main()
