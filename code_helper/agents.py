"""The four specialised worker agents.

Each worker is a Google ADK ``Agent`` with a focused description and instruction
(so the coordinator can route accurately) and **at least one tool**:

=====================  ==========================================================
Worker                 Tool(s)
=====================  ==========================================================
``code_searcher``      ``rag_search`` (custom RAG retrieval)
``code_explainer``     Filesystem MCP toolset + ``read_project_file``
``architecture_mapper`` ``list_project_files``, ``inspect_imports``
``onboarding_guide``   ``get_project_overview``
=====================  ==========================================================

All four share the *same* chat model instance returned by
:func:`code_helper.config.get_chat_model`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from code_helper.config import get_chat_model, get_settings
from code_helper.mcp_filesystem import filesystem_toolset_or_none
from code_helper.tools.architecture_tools import inspect_imports, list_project_files
from code_helper.tools.explain_tools import read_project_file
from code_helper.tools.onboarding_tools import get_project_overview
from code_helper.tools.search_tools import rag_search

logger = logging.getLogger(__name__)

#: Safety net: a worker may not make more than this many model calls per request.
MAX_MODEL_CALLS_PER_WORKER = 5

#: State tracking for tool calls and model calls per invocation to prevent loops.
#: Entries are removed by :func:`reset_invocation_state` when the run finishes so
#: a long-lived ``adk web`` process does not leak one entry per request.
_invocation_tool_counts: dict[str, dict[str, int]] = {}
_invocation_seen_calls: dict[str, set[tuple[str, str, str]]] = {}
_invocation_model_counts: dict[str, dict[str, int]] = {}


def get_max_tool_calls_for_agent(agent_name: str) -> int:
    """Return specific maximum tool calls allowed per worker agent."""
    settings = get_settings()
    # Onboarding and Search workers need only 1 tool call
    if agent_name in ("onboarding_guide", "code_searcher"):
        return 1
    # Explainer reads at most 2 files
    if agent_name == "code_explainer":
        return 2
    # Architecture mapper may list files then inspect imports
    # (up to max_tool_calls, default 3).
    return settings.max_tool_calls


def reset_invocation_state(callback_context: CallbackContext) -> None:
    """Drop the per-invocation counters once a worker has answered.

    Registered as each worker's ``after_agent_callback``. It must *not* live on
    the coordinator: ADK runs the coordinator and the chosen worker as separate
    steps, so the coordinator's ``after_agent_callback`` fires before the worker
    starts and would wipe the counters too early.

    Without this cleanup the three module-level maps would grow by one entry for
    every request handled by a long-running server.
    """
    inv_id = callback_context.invocation_id or "default"
    _invocation_tool_counts.pop(inv_id, None)
    _invocation_seen_calls.pop(inv_id, None)
    _invocation_model_counts.pop(inv_id, None)


def enforce_tool_call_limit(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> dict[str, Any] | None:
    """Intercept tool calls before execution to enforce call limits and prevent infinite loops."""
    agent_name = tool_context.agent_name or "worker"
    inv_id = tool_context.invocation_id or "default"
    tool_name = getattr(tool, "name", str(tool))

    agent_counts = _invocation_tool_counts.setdefault(inv_id, {})
    seen_calls = _invocation_seen_calls.setdefault(inv_id, set())

    current_count = agent_counts.get(agent_name, 0)
    max_allowed = get_max_tool_calls_for_agent(agent_name)

    # 1. Reject duplicate tool call with identical arguments in the same turn
    call_sig = (agent_name, tool_name, json.dumps(args, sort_keys=True, default=str))
    if call_sig in seen_calls:
        logger.warning(
            "Prevented duplicate tool call for %s: %s with args %s",
            agent_name,
            tool_name,
            args,
        )
        return {
            "result": (
                f"You have already called {tool_name} with these arguments in this turn. "
                "Do NOT call this tool again. Immediately provide your final response to the user "
                "in plain text using the previous results."
            )
        }

    # 2. Reject if agent has reached its allowed tool call budget
    if current_count >= max_allowed:
        logger.warning(
            "Tool call limit reached for %s (%d >= %d). Rejecting call to %s.",
            agent_name,
            current_count,
            max_allowed,
            tool_name,
        )
        return {
            "result": (
                f"Tool call limit reached for {agent_name} (maximum {max_allowed} calls). "
                "Do NOT call any more tools. Immediately synthesize and return your final answer "
                "to the user based on the information gathered so far."
            )
        }

    seen_calls.add(call_sig)
    agent_counts[agent_name] = current_count + 1
    return None


def enforce_model_call_limit(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Intercept model calls to enforce limits and force a final answer.

    Once a worker has spent its tool budget the tools are removed from the next
    request, so the model can only produce prose instead of looping. A hard cap
    of :data:`MAX_MODEL_CALLS_PER_WORKER` model calls is a final safety net.
    """
    agent_name = callback_context.agent_name or "worker"
    inv_id = callback_context.invocation_id or "default"

    model_counts = _invocation_model_counts.setdefault(inv_id, {})
    current_calls = model_counts.get(agent_name, 0) + 1
    model_counts[agent_name] = current_calls

    tool_counts = _invocation_tool_counts.get(inv_id, {}).get(agent_name, 0)
    max_tool_calls = get_max_tool_calls_for_agent(agent_name)

    # If the worker has executed its tool(s) and reached its tool limit,
    # disable tools for this model call so it cannot loop and is forced to answer in text.
    if tool_counts >= max_tool_calls:
        if llm_request.config:
            llm_request.config.tools = None
        llm_request.tools_dict = {}

    # Hard cap on LLM calls per worker to guarantee no runaway loop.
    if current_calls > MAX_MODEL_CALLS_PER_WORKER:
        logger.warning(
            "Agent model call limit reached for %s (%d calls). Halting loop.",
            agent_name,
            current_calls,
        )
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            f"[{agent_name}] I have finished analysing the "
                            "codebase with the available tools."
                        )
                    )
                ],
            )
        )

    return None


#: Workers must not hand off to each other or back to the coordinator. Only the
#: coordinator chooses a worker; a worker then answers with its own tools. This
#: prevents confusing multi-hop transfers such as
#: ``coordinator -> onboarding_guide -> code_explainer``.
WORKER_LIMITS: dict[str, Any] = {
    "disallow_transfer_to_parent": True,
    "disallow_transfer_to_peers": True,
    "before_tool_callback": enforce_tool_call_limit,
    "before_model_callback": enforce_model_call_limit,
    "after_agent_callback": reset_invocation_state,
}

CODE_SEARCHER_INSTRUCTION = """You are the Code Searcher for a Python codebase.

Your single job: find WHERE something is implemented.

Method:
1. Call the `rag_search` tool once with a focused natural-language query. If the
   user names a file (e.g. `main.py`), include that filename in the query:
   `rag_search` then performs an exact file lookup and returns that file.
2. After `rag_search` returns, DO NOT call it again.
3. Base every claim strictly on the returned blocks. If the tool reports an
   "Exact file match", describe that file and its contents directly.
4. If the requested file, function or symbol does not appear in ANY returned
   block, answer exactly: "Not found in the indexed code." Do not infer a file's
   existence or contents from imports in other files.
5. If `rag_search` says "No relevant evidence found", answer exactly:
   "Not found in the indexed code." Do not call another tool and do not use
   general knowledge about the topic (for example Kubernetes or Docker).
6. Never write "it is possible", "suggests", "might be" or "likely" about where
   something is implemented. Report only what the tool returned.
7. Always cite the source filename and line range (e.g. `vector_store.py` lines
   40-95). Quote only the relevant lines; do not paste whole files.
8. Never invent a filename, function or line number.

CRITICAL RULE: Call `rag_search` at most once per query. Immediately provide your
final textual answer once search results are returned. You do not explain designs
or architecture in depth; that is another agent's job.
"""

CODE_EXPLAINER_INSTRUCTION = """You are the Code Explainer for a Python codebase.

Your single job: explain what a COMPLETE source file does.

Method:
1. Read the target file in full. Prefer the Filesystem MCP tools (`read_text_file` /
   `read_file`, `list_directory`) restricted to the project root. If MCP is
   unavailable, use the `read_project_file` tool.
2. If the user did not name a file and no file is obvious from the question, ask
   which file they mean. Never read an arbitrary file just to produce an answer.
3. Read the file at most once. Once the file content is retrieved, DO NOT call any more tools.
4. Explain, in order: the module purpose, its imports and what they provide,
   each class and function, the control flow, and how the file depends on others.
5. Clearly separate what you OBSERVED in the file from anything you INFER.
6. Never invent code and never claim to have read a file you did not read. If you
   were only given a partial file, say which parts you saw.

CRITICAL RULE: Do not make repetitive tool calls. Provide your full explanation
directly in text once the file is read.
"""

ARCHITECTURE_MAPPER_INSTRUCTION = """You are the Architecture Mapper for a Python codebase.

Your single job: describe how the project is put together.

Method:
1. If the question is about a technology or component that this project does not
   contain (for example Kubernetes, Docker, a database or an external service),
   do not inspect any file; answer exactly "Not found in the indexed code."
2. Use `list_project_files` first to enumerate the modules.
3. Only call `inspect_imports` for a file the user named, or for a file returned
   by `list_project_files` that is relevant to the structural question. Never
   pick a file arbitrarily (for example, do not inspect `main.py` just because
   the question mentions configuration).
4. Use `inspect_imports` on the modules that matter to extract real imports and
   definitions (parsed with Python's `ast` module -- trust it over intuition).
5. If no module relates to the question, stop and answer
   "Not found in the indexed code."
6. Once the files and imports are retrieved, stop calling tools.
7. Describe the module dependency direction and identify likely entry points
   from evidence (a `__main__` guard, a conventional `main.py`, etc.).
8. Report ONLY the files and imports that the tools returned. Do not add a module
   or an import that no tool reported, and do not present a configuration module
   as an entry point unless it has an execution guard.
9. Prefer this deterministic analysis over guessing. State when evidence is thin.

CRITICAL RULE: Call tools at most 2-3 times to gather structural facts, then
immediately synthesize and output your final architectural description.
"""

ONBOARDING_GUIDE_INSTRUCTION = """You are the Onboarding Guide for a Python codebase.

Your single job: help a new developer get started.

Method:
1. Call `get_project_overview` once to obtain the verified structure, entry points and
   whatever run instructions are documented.
2. After `get_project_overview` returns, DO NOT call it or any other tool again.
3. Immediately explain the project purpose, the important files, how to install and run it,
   and a sensible reading order for a beginner.
4. Explicitly distinguish DOCUMENTED commands from INFERRED ones.
5. Do not invent functionality or commands that the overview did not report.

CRITICAL RULE: Never call `get_project_overview` more than once. You must provide
a final textual answer immediately after receiving the overview.
"""


def make_code_searcher() -> Agent:
    """Build the Code Searcher worker (RAG retrieval)."""
    return Agent(
        name="code_searcher",
        model=get_chat_model(),
        description=(
            "Finds WHERE a function, feature, variable or operation is implemented in "
            "the codebase using semantic (RAG) search over indexed source code. Use for "
            "questions like 'where is X implemented?' or 'which file handles Y?', "
            "including 'where is X configured?' questions. It reports 'not found' when "
            "the topic is absent."
        ),
        instruction=CODE_SEARCHER_INSTRUCTION,
        tools=[rag_search],
        **WORKER_LIMITS,
    )


def make_code_explainer() -> Agent:
    """Build the Code Explainer worker (full-file reading + explanation)."""
    tools: list[object] = [read_project_file]
    toolset = filesystem_toolset_or_none()
    if toolset is not None:
        tools.insert(0, toolset)
    return Agent(
        name="code_explainer",
        model=get_chat_model(),
        description=(
            "Explains what an entire source file does: its imports, functions, classes "
            "and control flow. Use when the user names a whole file or asks 'what does "
            "<file> do?'."
        ),
        instruction=CODE_EXPLAINER_INSTRUCTION,
        tools=tools,
        **WORKER_LIMITS,
    )


def make_architecture_mapper() -> Agent:
    """Build the Architecture Mapper worker (deterministic static analysis)."""
    return Agent(
        name="architecture_mapper",
        model=get_chat_model(),
        description=(
            "Maps module structure, dependencies and entry points across the whole "
            "project using static analysis. Use only for questions about this "
            "project's own modules, imports, layout or how its files connect -- not "
            "for external technologies such as Kubernetes, Docker or cloud services."
        ),
        instruction=ARCHITECTURE_MAPPER_INSTRUCTION,
        tools=[list_project_files, inspect_imports],
        **WORKER_LIMITS,
    )


def make_onboarding_guide() -> Agent:
    """Build the Onboarding Guide worker (project overview + setup)."""
    return Agent(
        name="onboarding_guide",
        model=get_chat_model(),
        description=(
            "Helps a new developer get started: project purpose, important files, "
            "installation and run steps, and a beginner reading order. Use for 'what "
            "does this project do?', 'how do I run it?' and 'where do I start?'."
        ),
        instruction=ONBOARDING_GUIDE_INSTRUCTION,
        tools=[get_project_overview],
        **WORKER_LIMITS,
    )


def build_worker_agents() -> list[Agent]:
    """Build and return the four worker agents in routing order."""
    return [
        make_code_searcher(),
        make_code_explainer(),
        make_architecture_mapper(),
        make_onboarding_guide(),
    ]
