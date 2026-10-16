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

from google.adk.agents import Agent

from code_helper.config import get_chat_model
from code_helper.mcp_filesystem import filesystem_toolset_or_none
from code_helper.tools.architecture_tools import inspect_imports, list_project_files
from code_helper.tools.explain_tools import read_project_file
from code_helper.tools.onboarding_tools import get_project_overview
from code_helper.tools.search_tools import rag_search

#: Workers must not hand off to each other or back to the coordinator. Only the
#: coordinator chooses a worker; a worker then answers with its own tools. This
#: prevents confusing multi-hop transfers such as
#: ``coordinator -> onboarding_guide -> code_explainer``.
WORKER_LIMITS: dict[str, bool] = {
    "disallow_transfer_to_parent": True,
    "disallow_transfer_to_peers": True,
}

CODE_SEARCHER_INSTRUCTION = """You are the Code Searcher for a Python codebase.

Your single job: find WHERE something is implemented.

Method:
1. Always call the `rag_search` tool with a focused natural-language query.
2. Base every claim strictly on the returned chunks.
3. Always cite the source filename and the line range (e.g. `vector_store.py`
   lines 40-95) and, when useful, the similarity score.
4. Quote only the relevant lines; do not paste whole files.
5. If `rag_search` returns no evidence, say so plainly. Never invent a filename,
   function or line number.

You do not explain designs or architecture in depth; that is another agent's job.
"""

CODE_EXPLAINER_INSTRUCTION = """You are the Code Explainer for a Python codebase.

Your single job: explain what a COMPLETE source file does.

Method:
1. Read the file in full. Prefer the Filesystem MCP tools (`read_text_file` /
   `read_file`, `list_directory`) restricted to the project root. If MCP is
   unavailable, use the `read_project_file` tool.
2. Explain, in order: the module purpose, its imports and what they provide,
   each class and function, the control flow, and how the file depends on others.
3. Clearly separate what you OBSERVED in the file from anything you INFER.
4. Never invent code and never claim to have read a file you did not read. If you
   were only given a partial file, say which parts you saw.
"""

ARCHITECTURE_MAPPER_INSTRUCTION = """You are the Architecture Mapper for a Python codebase.

Your single job: describe how the project is put together.

Method:
1. Use `list_project_files` to enumerate the modules.
2. Use `inspect_imports` on the modules that matter to extract real imports and
   definitions (parsed with Python's `ast` module -- trust it over intuition).
3. Describe the module dependency direction and identify likely entry points
   from evidence (a `__main__` guard, a conventional `main.py`, etc.).
4. Report ONLY the files and imports that the tools returned. Do not add a module
   or an import that no tool reported, and do not present a configuration module
   as an entry point unless it has an execution guard.
5. Prefer this deterministic analysis over guessing. State when evidence is thin.
"""

ONBOARDING_GUIDE_INSTRUCTION = """You are the Onboarding Guide for a Python codebase.

Your single job: help a new developer get started.

Method:
1. Call `get_project_overview` to obtain the verified structure, entry points and
   whatever run instructions are documented.
2. Explain the project purpose, the important files, how to install and run it,
   and a sensible reading order for a beginner.
3. Explicitly distinguish DOCUMENTED commands from INFERRED ones.
4. Do not invent functionality or commands that the overview did not report.
"""


def make_code_searcher() -> Agent:
    """Build the Code Searcher worker (RAG retrieval)."""
    return Agent(
        name="code_searcher",
        model=get_chat_model(),
        description=(
            "Finds WHERE a function, feature, variable or operation is implemented in "
            "the codebase using semantic (RAG) search over indexed source code. Use for "
            "questions like 'where is X implemented?' or 'which file handles Y?'."
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
            "project using static analysis. Use for questions about imports, module "
            "relationships, project layout or how files connect."
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
