"""Tests for the five custom ADK tools (deterministic, no network).

The tools expose only user-facing arguments, because Google ADK turns every
parameter into something the language model must fill in. Configuration is
resolved through each module's ``get_settings`` seam, which the
``patch_tool_settings`` fixture replaces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from code_helper import tools as tools_pkg
from code_helper.config import ConfigError, Settings
from code_helper.tools import search_tools
from code_helper.tools.architecture_tools import (
    collect_definitions,
    inspect_imports,
    list_project_files,
)
from code_helper.tools.explain_tools import read_project_file
from code_helper.tools.onboarding_tools import get_project_overview
from code_helper.tools.search_tools import rag_search


# --------------------------------------------------------------------------- #
# rag_search
# --------------------------------------------------------------------------- #
def test_rag_search_rejects_empty_query() -> None:
    assert "non-empty query" in rag_search("")


def test_rag_search_rejects_bad_top_k() -> None:
    assert "top_k" in rag_search("where is x", top_k=0)


def test_rag_search_accepts_string_top_k(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: LLMs may send numeric arguments as strings."""
    captured: dict = {}

    def fake_retrieve(query, top_k=None, **kwargs):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(search_tools, "retrieve", fake_retrieve)
    rag_search("where is x", top_k="5")
    assert captured["top_k"] == 5


def test_rag_search_rejects_non_numeric_top_k() -> None:
    assert "integer" in rag_search("where is x", top_k="many")


def test_rag_search_reports_no_evidence(
    settings: Settings, patch_tool_settings, monkeypatch
) -> None:
    patch_tool_settings(settings)
    monkeypatch.setattr(search_tools, "retrieve", lambda *a, **k: [])
    assert "No relevant evidence" in rag_search("where is x")


def test_rag_search_formats_matches(settings: Settings, patch_tool_settings, monkeypatch) -> None:
    from code_helper.retrieval import RetrievedChunk

    patch_tool_settings(settings)
    monkeypatch.setattr(
        search_tools,
        "retrieve",
        lambda *a, **k: [RetrievedChunk("id", "vector_store.py", 40, 95, 0.9, "def upsert(): ...")],
    )
    result = rag_search("where is upsert")
    assert "vector_store.py" in result
    assert "40-95" in result


def test_rag_search_rejects_unrelated_topic(
    settings: Settings, patch_tool_settings, monkeypatch
) -> None:
    """A topic absent from the project must not return unrelated chunks.

    Regression: "Where is the Kubernetes deployment configured?" used to return
    main.py/config.py and invite the model to speculate.
    """
    from code_helper.retrieval import RetrievedChunk

    patch_tool_settings(settings)
    monkeypatch.setattr(
        search_tools,
        "retrieve",
        lambda *a, **k: [
            RetrievedChunk("a", "main.py", 1, 66, 0.72, "import argparse\n"),
            RetrievedChunk("b", "config.py", 1, 78, 0.71, "import os\n"),
        ],
    )
    result = rag_search("Where is the Kubernetes deployment configured?")
    assert "No relevant evidence" in result
    assert "do NOT speculate" in result


def test_rag_search_keeps_lexically_supported_results(
    settings: Settings, patch_tool_settings, monkeypatch
) -> None:
    from code_helper.retrieval import RetrievedChunk

    patch_tool_settings(settings)
    monkeypatch.setattr(
        search_tools,
        "retrieve",
        lambda *a, **k: [
            RetrievedChunk("id", "vector_store.py", 40, 95, 0.9, "def cosine_similarity(a, b): ...")
        ],
    )
    result = rag_search("where is cosine similarity calculated")
    assert "vector_store.py" in result
    assert "No relevant evidence" not in result


def test_rag_search_exact_filename_reads_the_named_file(
    settings: Settings, patch_tool_settings, monkeypatch
) -> None:
    """A named file is read directly, never left to top-k similarity."""
    patch_tool_settings(settings)

    def must_not_run(*args, **kwargs):
        raise AssertionError("semantic retrieval must be skipped for a named file")

    monkeypatch.setattr(search_tools, "retrieve", must_not_run)
    result = rag_search("where is alpha.py")
    assert "Exact file match" in result
    assert "alpha.py" in result
    assert "def hello" in result


def test_rag_search_reports_named_file_not_found(
    settings: Settings, patch_tool_settings, monkeypatch
) -> None:
    patch_tool_settings(settings)

    def must_not_run(*args, **kwargs):
        raise AssertionError("semantic retrieval must be skipped for a named file")

    monkeypatch.setattr(search_tools, "retrieve", must_not_run)
    result = rag_search("where is main.py")
    assert "No readable file named" in result
    assert "Do NOT infer" in result


def test_rag_search_named_file_cannot_escape_project(
    settings: Settings, patch_tool_settings
) -> None:
    patch_tool_settings(settings)
    result = rag_search("where is ../../secret.py")
    assert "No readable file named" in result


def test_rag_search_handles_config_error(
    settings: Settings, patch_tool_settings, monkeypatch
) -> None:
    patch_tool_settings(settings)

    def boom(*a, **k):
        raise ConfigError("no key")

    monkeypatch.setattr(search_tools, "retrieve", boom)
    assert "configuration problem" in rag_search("q")


# --------------------------------------------------------------------------- #
# read_project_file
# --------------------------------------------------------------------------- #
def test_read_project_file_reads_inside_root(settings: Settings, patch_tool_settings) -> None:
    patch_tool_settings(settings)
    assert "def hello" in read_project_file("alpha.py")


def test_read_project_file_blocks_traversal(settings: Settings, patch_tool_settings) -> None:
    patch_tool_settings(settings)
    assert "Error" in read_project_file("../../secret.py")


def test_read_project_file_blocks_absolute_outside(
    settings: Settings, patch_tool_settings, tmp_path: Path
) -> None:
    patch_tool_settings(settings)
    outside = tmp_path / "outside.py"
    outside.write_text("secret = 1\n", encoding="utf-8")
    assert "Error" in read_project_file(str(outside))


def test_read_project_file_rejects_empty() -> None:
    assert "non-empty" in read_project_file("")


# --------------------------------------------------------------------------- #
# list_project_files
# --------------------------------------------------------------------------- #
def test_list_project_files_excludes_caches_and_secrets(
    settings: Settings, patch_tool_settings
) -> None:
    patch_tool_settings(settings)
    listing = list_project_files()
    assert "alpha.py" in listing
    assert ".env" not in listing
    assert "__pycache__" not in listing
    assert ".venv" not in listing


def test_list_project_files_missing_root(tmp_path: Path, patch_tool_settings) -> None:
    from tests.conftest import make_settings

    patch_tool_settings(make_settings(tmp_path / "missing"))
    assert "does not exist" in list_project_files()


# --------------------------------------------------------------------------- #
# inspect_imports
# --------------------------------------------------------------------------- #
def test_inspect_imports_reports_imports_and_definitions(
    settings: Settings, patch_tool_settings
) -> None:
    patch_tool_settings(settings)
    report = inspect_imports("beta.py")
    assert "from alpha import hello" in report
    assert "class Widget" in report


def test_inspect_imports_handles_syntax_error(settings: Settings, patch_tool_settings) -> None:
    patch_tool_settings(settings)
    assert "invalid Python syntax" in inspect_imports("broken.py")


def test_inspect_imports_blocks_traversal(settings: Settings, patch_tool_settings) -> None:
    patch_tool_settings(settings)
    assert "Error" in inspect_imports("../../../etc/passwd")


def test_collect_definitions_handles_invalid_source() -> None:
    assert collect_definitions("def oops(:") == {"classes": [], "functions": []}


def test_collect_definitions_extracts_names() -> None:
    source = "class A:\n    pass\n\ndef f():\n    pass\n"
    result = collect_definitions(source)
    assert result["classes"] == ["A"]
    assert result["functions"] == ["f"]


# --------------------------------------------------------------------------- #
# get_project_overview
# --------------------------------------------------------------------------- #
def test_overview_separates_documented_and_inferred(
    settings: Settings, patch_tool_settings
) -> None:
    patch_tool_settings(settings)
    overview = get_project_overview()
    assert "[documented] python main.py" in overview
    assert "Purpose (from README.md)" in overview


def test_overview_reports_missing_readme(tmp_path: Path, patch_tool_settings) -> None:
    from tests.conftest import make_settings

    root = tmp_path / "bare"
    root.mkdir()
    (root / "only.py").write_text("x = 1\n", encoding="utf-8")
    patch_tool_settings(make_settings(root))
    assert "no README.md found" in get_project_overview()


# --------------------------------------------------------------------------- #
# Public signatures must not leak internal arguments to the LLM
# --------------------------------------------------------------------------- #
def test_required_tool_signatures_are_minimal() -> None:
    import inspect as inspect_module

    assert list(inspect_module.signature(rag_search).parameters) == ["query", "top_k"]
    assert list(inspect_module.signature(read_project_file).parameters) == ["file_path"]
    assert list(inspect_module.signature(list_project_files).parameters) == []
    assert list(inspect_module.signature(inspect_imports).parameters) == ["file_path"]
    assert list(inspect_module.signature(get_project_overview).parameters) == []


# --------------------------------------------------------------------------- #
# Package exports / AI-free requirement
# --------------------------------------------------------------------------- #
def test_all_five_required_tools_are_exported() -> None:
    for name in (
        "rag_search",
        "read_project_file",
        "list_project_files",
        "inspect_imports",
        "get_project_overview",
    ):
        assert hasattr(tools_pkg, name), f"missing tool: {name}"


def test_deterministic_tools_have_no_llm_dependency() -> None:
    """The architecture/onboarding/explain tools must not depend on an LLM."""
    import inspect as inspect_module

    for func in (list_project_files, inspect_imports, read_project_file, get_project_overview):
        module = inspect_module.getmodule(func)
        source = inspect_module.getsource(module)
        assert "litellm" not in source, f"{module.__name__} unexpectedly depends on litellm"
        assert "google.adk" not in source, f"{module.__name__} unexpectedly depends on ADK"
