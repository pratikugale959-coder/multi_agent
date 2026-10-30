"""Stage 1 of the RAG pipeline -- file ingestion.

Responsibility
--------------
Walk the target project and return the source files worth indexing, then read
their original text. Nothing here touches the network or an LLM, which makes it
fully unit-testable.

Safety rules enforced here
--------------------------
* Only files *inside* the configured root are considered.
* ``.git``, virtual environments, caches and build output are skipped.
* Secret files (``.env``, ``*.key``, ``*.pem`` ...) are never read.
* Binary files (detected by a NUL byte) are skipped.
* Oversized files are skipped to avoid exploding the index.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

#: Directory names that are never descended into.
EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
        ".adk",
        "node_modules",
        "build",
        "dist",
        "site-packages",
        ".idea",
        ".vscode",
    }
)

#: File name patterns that are always skipped (secrets and noise).
EXCLUDED_FILE_NAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".env.local",
        ".env.example",
        "credentials",
        "credentials.json",
        "secrets.json",
    }
)

#: File extensions that are always skipped (secrets).
EXCLUDED_SUFFIXES: frozenset[str] = frozenset({".pem", ".key", ".pfx", ".p12", ".crt", ".sqlite3"})

#: Extensions indexed by default.
DEFAULT_EXTENSIONS: tuple[str, ...] = (".py",)

#: Files larger than this are skipped (bytes).
MAX_FILE_BYTES: int = 1_000_000


class IngestionError(RuntimeError):
    """Raised when the ingestion root is invalid."""


@dataclass(frozen=True)
class SourceFile:
    """A readable source file and its original text."""

    relative_path: str
    absolute_path: Path
    text: str

    @property
    def line_count(self) -> int:
        """Number of lines in the original text (empty text -> 0)."""
        if self.text == "":
            return 0
        return len(self.text.splitlines())


def _is_within(path: Path, root: Path) -> bool:
    """Return True if *path* is inside *root* (resolved)."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def is_excluded(path: Path) -> bool:
    """Return True if *path* should never be indexed.

    Checks excluded directory names anywhere in the path, excluded file names,
    and excluded suffixes.
    """
    if path.name in EXCLUDED_FILE_NAMES:
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    # Any excluded directory component disqualifies the file.
    return any(part in EXCLUDED_DIRS for part in path.parts)


def is_probably_binary(path: Path, sniff_bytes: int = 2048) -> bool:
    """Heuristically detect a binary file by looking for a NUL byte."""
    try:
        with path.open("rb") as handle:
            chunk = handle.read(sniff_bytes)
    except OSError:
        return True
    return b"\x00" in chunk


def discover_source_files(
    root: str | Path,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
) -> list[Path]:
    """Return sorted absolute paths of indexable files under *root*.

    Args:
        root: Directory to walk. Must exist and be a directory.
        extensions: Allowed suffixes, e.g. ``(".py",)``.

    Raises:
        IngestionError: If *root* does not exist or is not a directory.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise IngestionError(f"Target directory does not exist: {root_path}")
    if not root_path.is_dir():
        raise IngestionError(f"Target path is not a directory: {root_path}")

    normalised = {ext.lower() for ext in extensions}
    found: list[Path] = []

    for candidate in root_path.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in normalised:
            continue
        if is_excluded(candidate.relative_to(root_path)):
            continue
        if not _is_within(candidate, root_path):
            continue
        found.append(candidate)

    return sorted(found)


def read_source_file(path: str | Path, root: str | Path) -> SourceFile:
    """Read one file, returning a :class:`SourceFile`.

    Args:
        path: File to read.
        root: Root the file must live inside (path-traversal guard).

    Raises:
        IngestionError: If the file escapes *root*, is excluded, is binary, is
            too large, is missing, or cannot be decoded.
    """
    root_path = Path(root).expanduser().resolve()
    file_path = Path(path).expanduser()
    if not file_path.is_absolute():
        file_path = (root_path / file_path).resolve()
    else:
        file_path = file_path.resolve()

    if not _is_within(file_path, root_path):
        raise IngestionError(f"Refusing to read a file outside the project root: {file_path}")
    if is_excluded(file_path.relative_to(root_path)):
        raise IngestionError(f"Refusing to read an excluded file: {file_path.name}")
    if not file_path.exists() or not file_path.is_file():
        raise IngestionError(f"File not found: {file_path}")
    if file_path.stat().st_size > MAX_FILE_BYTES:
        raise IngestionError(f"File is too large to index (> {MAX_FILE_BYTES} bytes): {file_path}")
    if is_probably_binary(file_path):
        raise IngestionError(f"Refusing to index a binary file: {file_path.name}")

    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise IngestionError(f"File is not valid UTF-8: {file_path}") from exc
    except OSError as exc:
        raise IngestionError(f"Could not read file: {file_path} ({exc})") from exc

    return SourceFile(
        relative_path=file_path.relative_to(root_path).as_posix(),
        absolute_path=file_path,
        text=text,
    )


def load_source_files(
    root: str | Path,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
) -> list[SourceFile]:
    """Discover and read every indexable file under *root*.

    Files that raise :class:`IngestionError` (binary, undecodable, oversized) are
    skipped rather than aborting the whole run, so a single bad file cannot stop
    indexing.
    """
    root_path = Path(root).expanduser().resolve()
    sources: list[SourceFile] = []
    for path in discover_source_files(root_path, extensions=extensions):
        try:
            sources.append(read_source_file(path, root_path))
        except IngestionError:
            continue
    return sources


def iter_relative_paths(files: Iterable[SourceFile]) -> list[str]:
    """Convenience helper returning the relative path of each source file."""
    return [f.relative_path for f in files]
