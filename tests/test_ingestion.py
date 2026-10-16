"""Tests for stage 1 -- file discovery and safe reading."""

from __future__ import annotations

from pathlib import Path

import pytest

from code_helper.ingestion import (
    IngestionError,
    discover_source_files,
    is_excluded,
    load_source_files,
    read_source_file,
)


def test_discovers_only_python_files(sample_project: Path) -> None:
    names = {p.name for p in discover_source_files(sample_project)}
    # binary.py is a .py file, so it is discovered here; it is dropped later when
    # it cannot be decoded (see test_excludes_secret_and_binary_files).
    assert names == {"alpha.py", "beta.py", "big.py", "broken.py", "binary.py"}


def test_excludes_git_venv_and_cache_dirs(sample_project: Path) -> None:
    found = {p.as_posix() for p in discover_source_files(sample_project)}
    assert not any(".git/" in f for f in found)
    assert not any("__pycache__" in f for f in found)
    assert not any(".venv/" in f for f in found)


def test_excludes_secret_and_binary_files(sample_project: Path) -> None:
    names = {p.name for p in discover_source_files(sample_project)}
    assert ".env" not in names  # secrets are never discovered
    # binary.py is discovered by extension but load_source_files drops it because
    # it cannot be decoded safely.
    loaded = {s.relative_path for s in load_source_files(sample_project)}
    assert "binary.py" not in loaded
    assert "alpha.py" in loaded


def test_is_excluded_recognises_secret_names(tmp_path: Path) -> None:
    assert is_excluded(Path(".env"))
    assert is_excluded(Path("nested/credentials.json"))
    assert is_excluded(Path("keys/server.pem"))
    assert is_excluded(Path(".git/config"))
    assert not is_excluded(Path("src/app.py"))


def test_read_source_file_returns_relative_path_and_text(sample_project: Path) -> None:
    source = read_source_file("alpha.py", sample_project)
    assert source.relative_path == "alpha.py"
    assert "def hello" in source.text
    assert source.line_count == 5


def test_read_source_file_rejects_path_traversal(sample_project: Path) -> None:
    with pytest.raises(IngestionError):
        read_source_file("../../outside.py", sample_project)


def test_read_source_file_rejects_absolute_path_outside_root(sample_project: Path) -> None:
    outside = sample_project.parent / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(IngestionError):
        read_source_file(outside, sample_project)


def test_read_source_file_rejects_missing_file(sample_project: Path) -> None:
    with pytest.raises(IngestionError):
        read_source_file("does_not_exist.py", sample_project)


def test_read_source_file_rejects_non_utf8(tmp_path: Path) -> None:
    root = tmp_path
    (root / "latin.py").write_bytes(b"# -*- coding: latin-1 -*-\n'\xe9'\n")
    with pytest.raises(IngestionError):
        read_source_file("latin.py", root)


def test_discover_raises_for_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        discover_source_files(tmp_path / "nope")


def test_load_source_files_skips_unreadable_but_keeps_rest(sample_project: Path) -> None:
    loaded = {s.relative_path for s in load_source_files(sample_project)}
    # broken.py has invalid syntax but is still readable text, so it is loaded.
    assert {"alpha.py", "beta.py", "big.py", "broken.py"} <= loaded
