"""Integration check: run the multi-agent system and show routing.

Run from the directory containing ``code_helper``::

    python scripts/run_agent_demo.py
    python scripts/run_agent_demo.py "What does query_pipeline.py do?"

This drives the real coordinator through ADK's runner, printing which worker was
selected for each question and the answer it returned. It makes live LLM calls.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.genai import types  # noqa: E402

from code_helper.agent import root_agent  # noqa: E402

APP_NAME = "code_helper"

SAMPLE_QUESTIONS = [
    "Where is the upsert to the Pinecone vector database implemented?",
    "What does the file query_pipeline.py do?",
    "How do the modules in this project import and depend on each other?",
]


def _text_of(event: object) -> str:
    """Concatenate any text parts in an ADK event."""
    content = getattr(event, "content", None)
    if content is None or getattr(content, "parts", None) is None:
        return ""
    chunks = []
    for part in content.parts:
        text = getattr(part, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


async def ask(runner, question: str) -> None:
    """Send one question and print the routing trace plus the final answer."""

    session = await runner.session_service.create_session(app_name=APP_NAME, user_id="user")
    message = types.Content(role="user", parts=[types.Part(text=question)])

    print("=" * 78)
    print("QUESTION:", question)
    print("-" * 78)

    final_answer = ""
    authors: list[str] = []
    async for event in runner.run_async(user_id="user", session_id=session.id, new_message=message):
        author = getattr(event, "author", "?")
        if author not in authors:
            authors.append(author)
        text = _text_of(event)
        if text:
            final_answer = text
            print(f"[{author}] {text[:500]}")

    print("-" * 78)
    print("ROUTING:", " -> ".join(authors))
    print("ANSWER (final):", final_answer[:1200] if final_answer else "(no text)")
    print()


async def main() -> int:
    """Run the sample questions (or a single provided one)."""
    from google.adk.runners import InMemoryRunner

    questions = sys.argv[1:] or SAMPLE_QUESTIONS
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    for question in questions:
        await ask(runner, question)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main()))
