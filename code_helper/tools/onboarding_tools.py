"""Deterministic tool for the **Onboarding Guide** agent.

``get_project_overview`` summarises the target project from **evidence only**:
the file listing, the README (if present), ``__main__`` guards and the
requirements file. It clearly separates *documented* commands (found in the
README) from *inferred* commands (guessed from entry points), and never invents
functionality.

The same evidence is exposed as :class:`ProjectOverview` by
:func:`collect_project_overview`, so the coordinator's deterministic fast path
can render a short summary without a model call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from code_helper.core.config import Settings, get_settings
from code_helper.core.paths import project_root
from code_helper.rag.ingestion import discover_source_files

ENTRY_POINT_NAMES = ("main.py", "__main__.py", "app.py", "cli.py", "run.py")

COMMAND_PATTERN = re.compile(
    r"^\s*(?:[$>]\s*)?((?:python|py|pip|pip3|npx|adk|uvicorn|docker)\b[^\n`]*)$",
    re.IGNORECASE,
)

FENCE_PATTERN = re.compile(r"```")


@dataclass(frozen=True)
class ProjectOverview:
    """Verified facts about the target project, with no model involvement."""

    root: Path
    purpose: str
    readme_found: bool
    files: tuple[str, ...]
    dependencies: tuple[str, ...]
    requirements_found: bool
    named_entry_points: tuple[str, ...]
    guarded_entry_points: tuple[str, ...]
    documented_commands: tuple[str, ...]
    inferred_commands: tuple[str, ...]
    reading_order: tuple[str, ...]


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


def _purpose_from_readme(readme: str) -> str:
    """Return the README introduction (everything before the first heading/blank)."""
    first_paragraph: list[str] = []
    for raw in readme.splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        if stripped == "" and first_paragraph:
            break
        if stripped:
            first_paragraph.append(stripped)
    return " ".join(first_paragraph) or "(README has no introductory text)"


def collect_project_overview(settings: Settings | None = None) -> ProjectOverview:
    """Collect verified project facts without rendering them.

    Raises:
        FileNotFoundError: If the configured project root does not exist.
    """
    settings = settings or get_settings()
    root = project_root(settings)
    if not root.exists():
        raise FileNotFoundError(f"target project root does not exist: {root}")

    files = discover_source_files(root)
    readme = _read_if_exists(root, "README.md")
    requirements = _read_if_exists(root, "requirements.txt")

    dependencies: tuple[str, ...] = ()
    if requirements:
        dependencies = tuple(
            line.strip()
            for line in requirements.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    named, guarded = _entry_points(root, files)
    documented = _documented_commands(readme)
    inferred = [f"python {guarded_file}" for guarded_file in guarded]

    return ProjectOverview(
        root=root,
        purpose=_purpose_from_readme(readme) if readme else "",
        readme_found=readme is not None,
        files=tuple(path.relative_to(root).as_posix() for path in files),
        dependencies=dependencies,
        requirements_found=requirements is not None,
        named_entry_points=tuple(named),
        guarded_entry_points=tuple(guarded),
        documented_commands=tuple(documented),
        inferred_commands=tuple(inferred),
        reading_order=tuple(_suggest_reading_order(files, root, named + guarded)),
    )


def render_project_overview(overview: ProjectOverview) -> str:
    """Render the full, evidence-based onboarding overview."""
    lines: list[str] = [f"Project root: {overview.root}"]

    # -- purpose ----------------------------------------------------------- #
    if overview.readme_found:
        lines.append(f"Purpose (from README.md): {overview.purpose}")
    else:
        lines.append(
            "Purpose: no README.md found, so the purpose cannot be confirmed from documentation."
        )

    # -- structure --------------------------------------------------------- #
    lines.append(f"Python files ({len(overview.files)}):")
    if overview.files:
        for relative in overview.files:
            lines.append(f"  - {relative}")
    else:
        lines.append("  - (none)")

    if overview.requirements_found:
        packages = ", ".join(overview.dependencies) or "(empty)"
        lines.append(f"Dependencies (from requirements.txt): {packages}")
    else:
        lines.append("Dependencies: no requirements.txt found (none documented).")

    # -- entry points ------------------------------------------------------ #
    lines.append("Entry points (evidence-based):")
    if overview.named_entry_points:
        lines.append(f"  - Conventional names: {', '.join(overview.named_entry_points)}")
    if overview.guarded_entry_points:
        lines.append(
            "  - Files with an 'if __name__ == \"__main__\"' guard: "
            f"{', '.join(overview.guarded_entry_points)}"
        )
    if not overview.named_entry_points and not overview.guarded_entry_points:
        lines.append("  - None detected.")

    # -- run instructions -------------------------------------------------- #
    lines.append("Run instructions:")
    if overview.documented_commands:
        for command in overview.documented_commands:
            lines.append(f"  - [documented] {command}")
    else:
        lines.append("  - No run commands found in the README (nothing documented).")
    for command in overview.inferred_commands:
        lines.append(f"  - [inferred] {command}  (inferred from an __main__ guard, not documented)")
    if not overview.documented_commands and not overview.inferred_commands:
        lines.append("  - No run instructions could be determined from evidence.")

    # -- reading order ----------------------------------------------------- #
    if overview.reading_order:
        lines.append("Suggested reading order for a new developer:")
        for position, relative in enumerate(overview.reading_order, start=1):
            lines.append(f"  {position}. {relative}")

    lines.append(
        "Note: this overview is derived only from the files listed above; it does not "
        "infer functionality that is not visible in the source."
    )
    return "\n".join(lines)


def get_project_overview() -> str:
    """Summarise the target project: purpose, structure, entry points and setup.

    Use this for onboarding questions such as "what does this project do?",
    "where do I start?" and "how do I run it?".

    Returns:
        A structured overview built only from observable evidence.
    """
    try:
        overview = collect_project_overview()
    except FileNotFoundError as exc:
        return f"Error: {exc}"
    return render_project_overview(overview)
