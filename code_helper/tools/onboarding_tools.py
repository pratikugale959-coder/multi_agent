"""Deterministic tool for the **Onboarding Guide** agent.

``get_project_overview`` summarises the target project from **evidence only**:
the file listing, the README (if present), ``__main__`` guards and the
requirements file. It clearly separates *documented* commands (found in the
README) from *inferred* commands (guessed from entry points), and never invents
functionality.
"""

from __future__ import annotations

import re
from pathlib import Path

from code_helper.config import get_settings
from code_helper.ingestion import discover_source_files
from code_helper.paths import project_root

ENTRY_POINT_NAMES = ("main.py", "__main__.py", "app.py", "cli.py", "run.py", "__init__.py")

COMMAND_PATTERN = re.compile(
    r"^\s*(?:[$>]\s*)?((?:python|py|pip|pip3|npx|adk|uvicorn|docker)\b[^\n`]*)$",
    re.IGNORECASE,
)

FENCE_PATTERN = re.compile(r"```")


def _read_if_exists(root: Path, name: str) -> str | None:
    """Return the text of ``root/name`` when present, else ``None``."""
    path = root / name
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _documented_commands(readme_text: str | None) -> list[str]:
    """Extract shell commands that appear in the README (documented evidence)."""
    if not readme_text:
        return []
    commands: list[str] = []
    in_fence = False
    for raw in readme_text.splitlines():
        line = raw.rstrip()
        if FENCE_PATTERN.match(line.strip()):
            in_fence = not in_fence
            continue
        if not in_fence:
            continue
        match = COMMAND_PATTERN.match(line)
        if match:
            command = match.group(1).strip()
            if command and command not in commands:
                commands.append(command)
    return commands


def _entry_points(root: Path, files: list[Path]) -> tuple[list[str], list[str]]:
    """Return (named_entry_points, files_with_main_guard)."""
    named: list[str] = []
    guarded: list[str] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.name in ENTRY_POINT_NAMES:
            named.append(relative)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "__name__" in text and "__main__" in text:
            guarded.append(relative)
    return named, guarded


def _suggest_reading_order(files: list[Path], root: Path, entry_points: list[str]) -> list[str]:
    """Suggest a beginner reading order: entry point, config, then the rest."""
    ordered: list[str] = []

    def add(relative: str) -> None:
        if relative not in ordered:
            ordered.append(relative)

    for entry in entry_points:
        add(entry)

    priorities = ("config.py", "settings.py", "constants.py")
    for path in files:
        if path.name in priorities:
            add(path.relative_to(root).as_posix())

    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.name in ("README.md", "requirements.txt"):
            continue
        add(relative)
    return ordered


def get_project_overview() -> str:
    """Summarise the target project: purpose, structure, entry points and setup.

    Use this for onboarding questions such as "what does this project do?",
    "where do I start?" and "how do I run it?".

    Returns:
        A structured overview built only from observable evidence.
    """
    root = project_root(get_settings())
    if not root.exists():
        return f"Error: target project root does not exist: {root}"

    files = discover_source_files(root)
    readme = _read_if_exists(root, "README.md")
    requirements = _read_if_exists(root, "requirements.txt")

    lines: list[str] = [f"Project root: {root}"]

    # -- purpose ----------------------------------------------------------- #
    if readme:
        first_paragraph = []
        for raw in readme.splitlines():
            stripped = raw.strip()
            if stripped.startswith("#"):
                continue
            if stripped == "" and first_paragraph:
                break
            if stripped:
                first_paragraph.append(stripped)
        purpose = " ".join(first_paragraph) or "(README has no introductory text)"
        lines.append(f"Purpose (from README.md): {purpose}")
    else:
        lines.append(
            "Purpose: no README.md found, so the purpose cannot be confirmed from documentation."
        )

    # -- structure --------------------------------------------------------- #
    lines.append(f"Python files ({len(files)}):")
    if files:
        for path in files:
            lines.append(f"  - {path.relative_to(root).as_posix()}")
    else:
        lines.append("  - (none)")

    if requirements:
        packages = [
            line.strip()
            for line in requirements.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        lines.append(f"Dependencies (from requirements.txt): {', '.join(packages) or '(empty)'}")
    else:
        lines.append("Dependencies: no requirements.txt found (none documented).")

    # -- entry points ------------------------------------------------------ #
    named, guarded = _entry_points(root, files)
    lines.append("Entry points (evidence-based):")
    if named:
        lines.append(f"  - Conventional names: {', '.join(named)}")
    if guarded:
        lines.append(f"  - Files with an 'if __name__ == \"__main__\"' guard: {', '.join(guarded)}")
    if not named and not guarded:
        lines.append("  - None detected.")

    # -- run instructions -------------------------------------------------- #
    documented = _documented_commands(readme)
    lines.append("Run instructions:")
    if documented:
        for command in documented:
            lines.append(f"  - [documented] {command}")
    else:
        lines.append("  - No run commands found in the README (nothing documented).")
    inferred: list[str] = []
    for guarded_file in guarded:
        inferred.append(f"python {guarded_file}")
    for command in inferred:
        lines.append(f"  - [inferred] {command}  (inferred from an __main__ guard, not documented)")
    if not documented and not inferred:
        lines.append("  - No run instructions could be determined from evidence.")

    # -- reading order ----------------------------------------------------- #
    order = _suggest_reading_order(files, root, named + guarded)
    if order:
        lines.append("Suggested reading order for a new developer:")
        for position, relative in enumerate(order, start=1):
            lines.append(f"  {position}. {relative}")

    lines.append(
        "Note: this overview is derived only from the files listed above; it does not "
        "infer functionality that is not visible in the source."
    )
    return "\n".join(lines)
