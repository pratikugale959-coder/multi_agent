"""Tests for the deterministic "where is ...?" symbol locator."""

from __future__ import annotations

from code_helper.agents.fast_path import deterministic_answer
from code_helper.core.config import Settings
from code_helper.tools.symbol_locator import locate_symbol


def _write(settings: Settings, name: str, text: str) -> None:
    (settings.target_repo_path / name).write_text(text, encoding="utf-8")


def test_locates_an_obvious_operation(settings: Settings) -> None:
    _write(
        settings,
        "loader.py",
        "def load_document(path):\n"
        '    """Read and clean a source document."""\n'
        "    return path\n"
        "\n"
        "def run_pipeline():\n"
        '    """Load the document and process it."""\n'
        '    return load_document("x")\n',
    )
    answer = locate_symbol("Where is the document loaded?", settings)
    assert answer is not None
    assert "`loader.py`" in answer
    assert "`load_document()`" in answer
    assert "lines 1-3" in answer


def test_locates_all_definitions_in_one_file(settings: Settings) -> None:
    _write(
        settings,
        "store.py",
        "def upsert_to_vector_db(items):\n"
        '    """Store in Pinecone or the local JSON fallback."""\n'
        "    return len(items)\n"
        "\n"
        "def query_vector_db(query):\n"
        '    """Query Pinecone or the local JSON fallback."""\n'
        "    return query\n",
    )
    answer = locate_symbol("Where is the local JSON fallback used?", settings)
    assert answer is not None
    assert "`store.py`" in answer
    assert "`upsert_to_vector_db()`" in answer
    assert "`query_vector_db()`" in answer


def test_where_are_prefers_the_operation_named_by_the_question(settings: Settings) -> None:
    """Regression: "embedded" must locate ``embed_chunks``, not a caller."""
    _write(
        settings,
        "indexer.py",
        "def chunk_document(text):\n"
        '    """Split a document into chunks."""\n'
        "    return [text]\n"
        "\n"
        "def embed_chunks(chunks):\n"
        '    """Embed every chunk with the embedding model."""\n'
        "    return chunks\n"
        "\n"
        "def run_indexing_pipeline(document):\n"
        '    """Chunk, embed and store a document."""\n'
        "    return embed_chunks(chunk_document(document))\n",
    )
    answer = locate_symbol("Where are document chunks embedded?", settings)
    assert answer is not None
    assert "`embed_chunks()`" in answer
    assert "`run_indexing_pipeline()`" not in answer


def test_single_term_must_match_a_definition_name(settings: Settings) -> None:
    _write(
        settings,
        "store.py",
        'def upsert_to_vector_db(items):\n    """Store the vectors."""\n    return items\n',
    )
    answer = locate_symbol("Where is the upsert?", settings)
    assert answer is not None
    assert "`upsert_to_vector_db()`" in answer


def test_absent_topic_returns_none(settings: Settings) -> None:
    assert locate_symbol("Where is the Kubernetes deployment configured?", settings) is None


def test_tie_across_files_returns_none(settings: Settings) -> None:
    body = 'def handle():\n    """Handle the document load."""\n    return 1\n'
    _write(settings, "first.py", body)
    _write(settings, "second.py", body)
    assert locate_symbol("Where is the document loaded?", settings) is None


def test_fast_path_answers_obvious_location_questions(settings: Settings) -> None:
    _write(
        settings,
        "loader.py",
        'def load_document(path):\n    """Read and clean a source document."""\n    return path\n',
    )
    answer = deterministic_answer("Where is the document loaded?", settings)
    assert answer is not None
    assert "`loader.py`" in answer
    assert "`load_document()`" in answer


def test_fast_path_answers_where_are_questions(settings: Settings) -> None:
    """\"Where are ...?\" reaches the locator too, not just \"Where is ...?\"."""
    _write(
        settings,
        "indexer.py",
        "def embed_chunks(chunks):\n"
        '    """Embed every chunk with the embedding model."""\n'
        "    return chunks\n",
    )
    answer = deterministic_answer("Where are document chunks embedded?", settings)
    assert answer is not None
    assert "`indexer.py`" in answer
    assert "`embed_chunks()`" in answer
