"""Command-line entry point for indexing the target project.

Usage (run from the directory that contains ``code_helper``)::

    python -m code_helper.index_codebase
    python -m code_helper.index_codebase --root target_project --progress
    python -m code_helper.index_codebase --stats

This is the script a developer runs **once** before asking the agents questions,
so that Pinecone contains the code that the agents will retrieve.
"""

from __future__ import annotations

import argparse
import json
import sys

from code_helper.config import ConfigError, get_settings
from code_helper.rag_pipeline import index_repository
from code_helper.vector_store import PineconeVectorStore, VectorStoreError


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m code_helper.index_codebase",
        description="Index a Python project into Pinecone for the Multi-Agent Code Helper.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Directory to index (defaults to TARGET_REPO_PATH from .env).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print per-stage progress.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Only print current index statistics; do not index anything.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    settings = get_settings()

    if args.stats:
        try:
            stats = PineconeVectorStore(settings).stats()
        except (VectorStoreError, ConfigError) as exc:
            print(f"Could not read index statistics: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(stats, indent=2))
        return 0

    try:
        report = index_repository(
            root=args.root,
            settings=settings,
            progress=args.progress,
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except VectorStoreError as exc:
        print(f"Vector store error: {exc}", file=sys.stderr)
        return 3

    print(json.dumps(report.as_dict(), indent=2))
    if report.chunks_indexed == 0:
        print("Nothing was indexed. Check the target path and your credentials.", file=sys.stderr)
        return 1
    print(f"Indexed {report.chunks_indexed} chunk(s) into '{report.index_name}'.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
