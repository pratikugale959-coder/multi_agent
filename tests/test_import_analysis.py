"""Tests for deterministic import-graph answers (no model, no network)."""

from __future__ import annotations

from code_helper.core.config import Settings
from code_helper.tools.import_analysis import (
    build_import_graph,
    render_importers,
    render_imports,
    render_relationship,
)


def test_graph_resolves_project_imports(settings: Settings) -> None:
    graph = build_import_graph(settings)
    assert "alpha.py" in graph.files
    beta_edges = graph.edges["beta.py"]
    alpha_edges = [edge for edge in beta_edges if edge.target == "alpha.py"]
    assert alpha_edges, beta_edges
    assert alpha_edges[0].internal is True
    assert alpha_edges[0].statement == "from alpha import hello"


def test_graph_classifies_stdlib_and_missing_files(settings: Settings) -> None:
    graph = build_import_graph(settings)
    alpha_edges = graph.edges["alpha.py"]
    assert any(edge.target == "os" and edge.stdlib for edge in alpha_edges)
    # A syntactically broken file is skipped, not allowed to abort the graph.
    assert "broken.py" in graph.files


def test_render_imports_lists_internal_and_stdlib(settings: Settings) -> None:
    answer = render_imports("beta.py", settings)
    assert answer is not None
    assert "`alpha.py`" in answer
    assert "`from alpha import hello`" in answer
    assert "Standard library" in answer


def test_render_importers_finds_reverse_dependency(settings: Settings) -> None:
    answer = render_importers("alpha.py", settings)
    assert answer is not None
    assert "`beta.py`" in answer
    assert "from alpha import hello" in answer


def test_render_importers_reports_no_importers(settings: Settings) -> None:
    answer = render_importers("beta.py", settings)
    assert answer is not None
    assert "No Python file" in answer


def test_render_relationship_finds_direct_and_indirect_paths(settings: Settings) -> None:
    direct = render_relationship("beta.py", "alpha.py", settings)
    assert direct is not None
    assert "`beta.py` -> `alpha.py`" in direct
    assert "from alpha import hello" in direct

    # gamma imports beta, which imports alpha: a two-hop path.
    (settings.target_repo_path / "gamma.py").write_text(
        "from beta import Widget\n", encoding="utf-8"
    )
    indirect = render_relationship("gamma.py", "alpha.py", settings)
    assert indirect is not None
    assert "`gamma.py` -> `beta.py` -> `alpha.py`" in indirect


def test_render_relationship_reports_reverse_direction(settings: Settings) -> None:
    answer = render_relationship("alpha.py", "beta.py", settings)
    assert answer is not None
    assert "is imported by" in answer


def test_render_relationship_reports_missing_connection(settings: Settings) -> None:
    answer = render_relationship("alpha.py", "big.py", settings)
    assert answer is not None
    assert "No import path" in answer


def test_render_helpers_return_none_for_unknown_file(settings: Settings) -> None:
    assert render_imports("missing.py", settings) is None
    assert render_importers("missing.py", settings) is None
    assert render_relationship("missing.py", "alpha.py", settings) is None
