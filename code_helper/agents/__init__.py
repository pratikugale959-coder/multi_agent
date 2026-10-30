"""The agent graph: one coordinator and four specialist workers.

``root_agent`` is what ADK discovers from the top-level ``code_helper`` package.
Keeping the coordinator and workers in this subpackage makes the distinction
between the two roles explicit.
"""

from code_helper.agents.coordinator import (
    COORDINATOR_INSTRUCTION,
    root_agent,
    setup_invocation_limits,
)
from code_helper.agents.workers import (
    ARCHITECTURE_MAPPER_INSTRUCTION,
    CODE_EXPLAINER_INSTRUCTION,
    CODE_SEARCHER_INSTRUCTION,
    ONBOARDING_GUIDE_INSTRUCTION,
    WORKER_LIMITS,
    build_worker_agents,
    enforce_model_call_limit,
    enforce_tool_call_limit,
    reset_invocation_state,
)

__all__ = [
    "ARCHITECTURE_MAPPER_INSTRUCTION",
    "CODE_EXPLAINER_INSTRUCTION",
    "CODE_SEARCHER_INSTRUCTION",
    "COORDINATOR_INSTRUCTION",
    "ONBOARDING_GUIDE_INSTRUCTION",
    "WORKER_LIMITS",
    "build_worker_agents",
    "enforce_model_call_limit",
    "enforce_tool_call_limit",
    "reset_invocation_state",
    "root_agent",
    "setup_invocation_limits",
]
