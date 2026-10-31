"""ADK entry point -- the coordinator agent.

``root_agent`` is what ``adk run code_helper`` and ``adk web`` discover. It is the
**Boss**: it has *no tools of its own*, so it cannot answer a technical question
itself. Its only capability is delegating to one of the four worker agents, which
keeps routing honest and answers grounded in the workers' tools.
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.run_config import RunConfig
from google.genai import types

from code_helper.agents.callbacks import strip_reasoning
from code_helper.agents.fast_path import fast_path_callback
from code_helper.agents.workers import build_worker_agents
from code_helper.core.config import get_chat_model, get_settings

COORDINATOR_INSTRUCTION = """You are the coordinator (the "Boss") of a multi-agent code helper.

You help developers understand an unfamiliar Python project. The workers search
ONLY the indexed target repository (`TARGET_REPO_PATH`, by default
`target_project/`). You never answer a technical question about that project
yourself: you always delegate to the single most appropriate specialist worker,
then present that worker's findings to the user.

Scope of the indexed project:
- Questions about THIS HELPER's own source -- the coordinator, a worker,
  `code_helper`, "your own code" or "this assistant" -- are outside the indexed
  repository by default. Do NOT answer them from general knowledge and do NOT
  route them to `code_searcher`: that code is not indexed and the lookup will
  time out. Reply with a short scope note instead: the indexed repository is
  `target_project/`; the coordinator is defined in
  `code_helper/agents/coordinator.py`; to include the helper, set
  `TARGET_REPO_PATH=./` in `.env` and re-index.
- Never ask the user "which file?" for a helper-scope question. The file is
  already named in this instruction.

Routing rules (pick exactly one worker):

1. `code_searcher` -- WHERE is something implemented or where is a file/function?
   Signals: "where is", "find", "locate", "which function/file handles",
   "find the code that", "how is X computed", locating a file (e.g. "find main.py"),
   a variable, feature or operation.
   Use this for ANY "where is ...?" question, even when it mentions configuration,
   deployment, setup or infrastructure. It is allowed to answer "not found".

2. `code_explainer` -- WHAT does a whole file do?
   Signals: a specific filename plus "explain", "what does <file> do",
   "walk me through <file>", imports/functions/classes of one file.

3. `architecture_mapper` -- HOW are THIS PROJECT's files/modules connected?
   Signals: "imports", "dependency", "module structure", "project layout",
   "entry point", "how do the files connect".
   Use it only for the project's own modules. Never for external technologies
   (Kubernetes, Docker, CI, cloud services) that the project may not contain.

4. `onboarding_guide` -- SETUP and getting started.
   Signals: "what does this project do", "how do I run it", "installation",
   "where do I start", "reading order", "getting started".

If the user asks about a technology, service or file that may not exist in this
project, still route to `code_searcher`; reporting "not found" is the correct
answer. Never route such a question to `architecture_mapper`.

How to delegate:
- Rewrite the user's request as a precise task for the chosen worker, naming the
  file or symbol when the user provided one.
- Choose exactly one worker.
- Do not narrate your reasoning or routing decision ("The user asks...",
  "We need to transfer..."). Either delegate silently or return the final answer.
- Once the worker answers, return its answer directly to the user, keeping the
  filenames, line ranges and citations intact. Do NOT re-delegate or call
  additional tools.
- Do not add your own speculation or new technical claims.

If a request is ambiguous, ask one short clarifying question instead of guessing.

FINAL REMINDER: Never narrate routing or reasoning. Your visible message must be
either a delegation call or the worker's final answer, nothing else.
"""


def setup_invocation_limits(callback_context: CallbackContext) -> None:
    """Apply the configured run-level cap on LLM calls.

    ADK's ``RunConfig`` defaults to 500 calls, so merely filling in an empty
    value never applied ``MAX_LLM_CALLS``. Cap the existing value instead, and
    never raise a limit that the runner set lower.

    ``get_invocation_context()`` returns a copy, so assigning ``run_config``
    on it would be lost; mutate the real invocation context.
    """
    limit = get_settings().max_llm_calls
    if limit <= 0:
        return

    invocation = callback_context._invocation_context
    if invocation.run_config is None:
        invocation.run_config = RunConfig(max_llm_calls=limit)
    elif invocation.run_config.max_llm_calls <= 0:
        invocation.run_config.max_llm_calls = limit
    else:
        invocation.run_config.max_llm_calls = min(invocation.run_config.max_llm_calls, limit)


def prepare_agent(callback_context: CallbackContext) -> types.Content | None:
    """Apply run limits and answer narrow deterministic questions immediately.

    Returning content from ``before_agent_callback`` skips the coordinator's
    model call, so a filename lookup still works when the hosted model is slow.
    """
    setup_invocation_limits(callback_context)
    return fast_path_callback(callback_context)


def _build_root_agent() -> Agent:
    """Construct the coordinator with its four workers attached."""
    return Agent(
        name="coordinator",
        model=get_chat_model(),
        description=(
            "Boss/coordinator for the Multi-Agent Code Helper. Receives a question, "
            "selects the correct specialist worker, delegates, and returns the "
            "worker's answer."
        ),
        instruction=COORDINATOR_INSTRUCTION,
        sub_agents=build_worker_agents(),
        before_agent_callback=prepare_agent,
        after_model_callback=strip_reasoning,
    )


root_agent = _build_root_agent()
