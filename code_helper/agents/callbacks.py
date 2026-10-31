"""Response hygiene callbacks for the worker agents.

``openai/gpt-oss-20b`` is the only reliable tool-capable model currently served
by the configured NVIDIA NIM endpoint. With tool results in the conversation it
sometimes prefixes an otherwise correct answer with a line of private planning
(``"We need to..."``, ``"User asks..."``). The instruction already forbids this,
but a deterministic cleanup makes the user-visible answer stable.

The callback only touches **final text** responses. Tool-call responses are left
alone. A response made up entirely of planning text is reduced to an empty part
instead of being shown to the user.
"""

from __future__ import annotations

from typing import Any

#: Leading lines that are model planning, not part of the answer. Matching is
#: case-insensitive and only applies at the start of the response.
_REASONING_PREFIXES: tuple[str, ...] = (
    "user asks",
    "user asks for",
    "user wants",
    "user needs",
    "user requested",
    "user requests",
    "user is asking",
    "user is requesting",
    "user is looking",
    "the user asks",
    "the user asked",
    "the user wants",
    "the user needs",
    "the user requested",
    "the user is asking",
    "the user is requesting",
    "according to routing",
    "routing rules",
    "we need",
    "we should",
    "we must",
    "we cannot",
    "need to",
    "let's",
    "let me",
    "i need",
    "i should",
    "i will",
    "answer with",
    "answer in",
    "respond with",
    "reply with",
    "state the",
    "give the",
)


def _clean_answer(text: str) -> str:
    """Return the answer with leading planning lines removed (may be empty)."""
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        candidate = lines[index].strip().lower()
        if not candidate:
            index += 1
            continue
        if any(candidate.startswith(prefix) for prefix in _REASONING_PREFIXES):
            index += 1
            continue
        break

    return "\n".join(line.rstrip() for line in lines[index:]).strip()


def _strip_leading_reasoning(text: str) -> str:
    """Remove leading planning lines while preserving the actual answer.

    If *every* line was planning, the result is empty rather than the original
    text: the user must never see private routing notes such as
    ``"User wants overview. Use onboarding_guide."``.
    """
    return _clean_answer(text)


def clean_answer(text: str) -> str:
    """Public helper: strip planning lines, returning ``""`` if none remain.

    Callers that accumulate events (such as the live demo script) use this to
    ignore a later planning-only event instead of overwriting a good answer.
    """
    return _clean_answer(text)


def strip_reasoning(callback_context: Any, llm_response: Any) -> Any:
    """ADK ``after_model_callback`` that removes thoughts and planning lines.

    The shared model wrapper already drops ``thought=True`` parts before they
    reach ADK, but this callback keeps the guarantee for any model path that
    bypasses that wrapper. Returns the modified response when something
    changed, otherwise ``None`` so ADK keeps the original response untouched.
    """
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None)
    if not parts:
        return None

    # Thoughts are private reasoning, never part of the answer. Removing them
    # here as well means a response with no visible text cannot leak them.
    visible_parts = [part for part in parts if not getattr(part, "thought", False)]
    removed_thoughts = len(visible_parts) != len(parts)
    if removed_thoughts:
        content.parts = visible_parts

    # A pure tool-call turn has no text and is left untouched. Mixed responses
    # (a transfer call plus a short text part) still need their text sanitised.
    text_parts = [part for part in visible_parts if getattr(part, "text", None)]
    if not text_parts:
        return llm_response if removed_thoughts else None

    # ADK may split one answer across several text parts (for example a short
    # planning line followed by the answer). Sanitize the combined text so a
    # part that is only planning does not survive on its own.
    combined = ""
    for index, part in enumerate(text_parts):
        if index and not combined.endswith(("\n", " ")):
            combined += "\n"
        combined += part.text
    cleaned = _strip_leading_reasoning(combined)
    if cleaned == combined:
        return llm_response if removed_thoughts else None

    text_parts[0].text = cleaned
    for part in text_parts[1:]:
        part.text = ""
    return llm_response
