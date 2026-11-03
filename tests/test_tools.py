"""Tests for the five custom ADK tools (deterministic, no network).

The tools expose only user-facing arguments, because Google ADK turns every
parameter into something the language model must fill in. Configuration is
resolved through each module's ``get_settings`` seam, which the
``patch_tool_settings`` fixture replaces.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from google.genai import types

from code_helper import tools as tools_pkg
from code_helper.agents.callbacks import clean_answer
from code_helper.agents.fast_path import deterministic_answer, fast_path_callback
from code_helper.core.config import ConfigError, Settings
from code_helper.tools import search_tools
from code_helper.tools.architecture_tools import (
    collect_definitions,
    inspect_imports,
    list_project_files,
)
from code_helper.tools.explain_tools import read_project_file, summarize_project_file
from code_helper.tools.onboarding_tools import ENTRY_POINT_NAMES, get_project_overview
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
    from code_helper.rag.retrieval import RetrievedChunk

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
    from code_helper.rag.retrieval import RetrievedChunk

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
    from code_helper.rag.retrieval import RetrievedChunk

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


def test_rag_search_points_at_the_file_covering_every_query_term(
    settings: Settings, patch_tool_settings, monkeypatch
) -> None:
    """Regression: a higher-ranked storage helper must not hide the defining file."""
    from code_helper.rag.retrieval import RetrievedChunk

    patch_tool_settings(settings)
    monkeypatch.setattr(
        search_tools,
        "retrieve",
        lambda *a, **k: [
            RetrievedChunk(
                "rank1",
                "vector_store.py",
                81,
                170,
                0.70,
                "for item in chunks_with_embeddings:\n    index.upsert(vectors=payload)",
            ),
            RetrievedChunk(
                "rank2",
                "indexer.py",
                1,
                91,
                0.69,
                "def embed_chunks(chunks):\n    # Embedded document chunks are returned here.",
            ),
        ],
    )
    result = rag_search("document chunks embedded")
    assert "Best query-term match" in result
    hint = result.split("Best query-term match", 1)[1]
    assert "indexer.py" in hint
    assert "vector_store.py" not in hint.split("Prefer this file", 1)[0]
    # The full-term match is also listed first, so a model that cites the top
    # result lands on the defining file rather than the storage helper.
    assert result.index("indexer.py (lines") < result.index("vector_store.py (lines")


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


def test_rag_search_caps_named_file_output(settings: Settings, patch_tool_settings) -> None:
    """A mixed question about a large file must not flood the prompt."""
    patch_tool_settings(settings)
    long_file = settings.target_repo_path / "long.py"
    long_file.write_text(
        "\n".join(f"value_{i} = {i}" for i in range(1, 701)),
        encoding="utf-8",
    )
    result = rag_search("where is long.py used")
    assert "Showing the first 4 of" in result
    assert len(result) < 12_000


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


def test_read_project_file_caps_very_large_files(settings: Settings, patch_tool_settings) -> None:
    patch_tool_settings(settings)
    huge = settings.target_repo_path / "huge.py"
    huge.write_text("x = 1\n" * 12_000, encoding="utf-8")
    result = read_project_file("huge.py")
    assert "Truncated after" in result
    assert len(result) < 41_000


def test_summarize_project_file_returns_compact_outline(
    settings: Settings, patch_tool_settings
) -> None:
    patch_tool_settings(settings)
    outline = summarize_project_file("alpha.py")
    assert "File: alpha.py" in outline
    assert "import os" in outline
    assert "hello()" in outline
    assert len(outline) < 1000


def test_summarize_project_file_blocks_traversal(settings: Settings, patch_tool_settings) -> None:
    patch_tool_settings(settings)
    assert "Error" in summarize_project_file("../../secret.py")


def test_fast_path_locates_named_file(settings: Settings) -> None:
    answer = deterministic_answer("Where is alpha.py?", settings)
    assert answer is not None
    assert "alpha.py" in answer
    assert "Path:" in answer


def test_fast_path_summarizes_named_file(settings: Settings) -> None:
    answer = deterministic_answer("What does alpha.py do?", settings)
    assert answer is not None
    assert "Definitions:" in answer
    assert "hello()" in answer


def test_fast_path_skips_mixed_or_unrelated_queries(settings: Settings) -> None:
    assert deterministic_answer("Where is the Kubernetes deployment configured?", settings) is None
    assert deterministic_answer("Explain the design of the deployment process", settings) is None


def test_fast_path_answers_import_questions_without_a_model(settings: Settings) -> None:
    outgoing = deterministic_answer("What files does beta.py import?", settings)
    assert outgoing is not None
    assert "`alpha.py`" in outgoing
    assert "from alpha import hello" in outgoing

    incoming = deterministic_answer("Where is alpha.py used?", settings)
    assert incoming is not None
    assert "`beta.py`" in incoming


def test_fast_path_answers_two_file_relationship(settings: Settings) -> None:
    answer = deterministic_answer("How does beta.py connect to alpha.py?", settings)
    assert answer is not None
    assert "`beta.py` -> `alpha.py`" in answer
    assert "from alpha import hello" in answer


def test_fast_path_callback_returns_immediate_content(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("code_helper.agents.fast_path.get_settings", lambda: settings)

    class Context:
        user_content = types.Content(role="user", parts=[types.Part(text="Where is alpha.py?")])

    result = fast_path_callback(Context())
    assert result is not None
    assert "alpha.py" in (result.parts[0].text or "")


def test_fast_path_summarizes_project_without_a_model_call(settings: Settings) -> None:
    answer = deterministic_answer("Tell me something about this codebase in 5 lines", settings)
    assert answer is not None
    assert "This project demonstrates a tiny pipeline." in answer
    assert "Python files" in answer
    assert "pip install -r requirements.txt" in answer
    # The short overview stays concise instead of dumping the full onboarding report.
    assert len(answer.splitlines()) <= 6


def test_fast_path_answers_project_overview_and_run_questions(settings: Settings) -> None:
    for question in (
        "What does this project do?",
        "How do I run this project?",
        "Where do I start?",
    ):
        answer = deterministic_answer(question, settings)
        assert answer is not None, question
        assert "alpha.py" in answer


def test_fast_path_explains_the_helpers_own_scope(settings: Settings) -> None:
    answer = deterministic_answer("Explain how the coordinator agent works", settings)
    assert answer is not None
    assert "code_helper/agents/coordinator.py" in answer
    assert "outside the indexed scope" in answer
    assert settings.target_repo_path.name in answer
    assert "TARGET_REPO_PATH" in answer


def test_fast_path_does_not_treat_file_questions_as_scope_questions(settings: Settings) -> None:
    """A real file in the target project must still use the outline path."""
    assert deterministic_answer("What does beta.py do?", settings) is not None
    # A helper file that is not in the target still gets the scope explanation.
    assert "indexed scope" in (deterministic_answer("Where is coordinator.py?", settings) or "")


def test_fast_path_answers_overview_before_the_coordinator_model(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("code_helper.agents.fast_path.get_settings", lambda: settings)

    class Context:
        user_content = types.Content(
            role="user", parts=[types.Part(text="What does this project do?")]
        )

    result = fast_path_callback(Context())
    assert result is not None
    assert "This project demonstrates a tiny pipeline." in (result.parts[0].text or "")


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


def test_onboarding_excludes_package_init_from_entry_points() -> None:
    """``__init__.py`` is package plumbing, not an application entry point."""
    assert "__init__.py" not in ENTRY_POINT_NAMES


# --------------------------------------------------------------------------- #
# Response hygiene
# --------------------------------------------------------------------------- #
def test_clean_answer_strips_clear_planning_prefix() -> None:
    text = "We need to check the tool result.\n\nThe function lives in alpha.py."
    assert clean_answer(text) == "The function lives in alpha.py."


def test_clean_answer_keeps_legitimate_answer_opening() -> None:
    """Phrases like "We found..." can be a real answer, so they are preserved."""
    text = "We found the function in alpha.py."
    assert clean_answer(text) == text


@pytest.mark.parametrize(
    "planning",
    [
        "User wants overview. Use onboarding_guide.",
        "User needs to find the right worker.",
        "User is asking about the coordinator.",
        "User requested a short summary.",
        "The user wants an overview.",
        "The user needs the coordinator explained.",
    ],
)
def test_clean_answer_strips_extended_planning_prefixes(planning: str) -> None:
    assert clean_answer(f"{planning}\nThe real answer is here.") == "The real answer is here."


def test_clean_answer_removes_a_planning_only_response() -> None:
    """Private routing notes must never become the visible answer."""
    assert clean_answer("User wants overview. Use onboarding_guide.") == ""


# --------------------------------------------------------------------------- #
# Public signatures must not leak internal arguments to the LLM
# --------------------------------------------------------------------------- #
def test_required_tool_signatures_are_minimal() -> None:
    import inspect as inspect_module

    assert list(inspect_module.signature(rag_search).parameters) == ["query", "top_k"]
    assert list(inspect_module.signature(read_project_file).parameters) == ["file_path"]
    assert list(inspect_module.signature(summarize_project_file).parameters) == ["file_path"]
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

    for func in (
        list_project_files,
        inspect_imports,
        read_project_file,
        summarize_project_file,
        get_project_overview,
    ):
        module = inspect_module.getmodule(func)
        source = inspect_module.getsource(module)
        assert "litellm" not in source, f"{module.__name__} unexpectedly depends on litellm"
        assert "google.adk" not in source, f"{module.__name__} unexpectedly depends on ADK"
