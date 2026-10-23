"""Tests for the agent graph: one coordinator, four workers, shared model.

These tests construct real ADK ``Agent`` objects but never run them, so no model
inference or network access is required.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from google.adk.agents.run_config import RunConfig
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

from code_helper.agent import root_agent, setup_invocation_limits
from code_helper.agents import (
    _invocation_model_counts,
    _invocation_seen_calls,
    _invocation_tool_counts,
    build_worker_agents,
    enforce_model_call_limit,
    enforce_tool_call_limit,
    reset_invocation_state,
)
from code_helper.config import get_chat_model, get_settings

WORKER_NAMES = ["code_searcher", "code_explainer", "architecture_mapper", "onboarding_guide"]


def test_coordinator_exists_and_is_named() -> None:
    assert root_agent.name == "coordinator"


def test_coordinator_has_exactly_four_workers() -> None:
    assert len(root_agent.sub_agents) == 4
    assert {a.name for a in root_agent.sub_agents} == set(WORKER_NAMES)


def test_coordinator_has_no_tools_of_its_own() -> None:
    # The Boss must delegate, never answer directly.
    assert not root_agent.tools


@pytest.mark.parametrize("name", WORKER_NAMES)
def test_each_worker_has_at_least_one_tool(name: str) -> None:
    worker = next(a for a in root_agent.sub_agents if a.name == name)
    assert worker.tools, f"{name} has no tools"
    assert len(worker.tools) >= 1


@pytest.mark.parametrize("name", WORKER_NAMES)
def test_each_worker_has_a_description_for_routing(name: str) -> None:
    worker = next(a for a in root_agent.sub_agents if a.name == name)
    assert worker.description and len(worker.description) > 30


def test_all_agents_share_one_chat_model_instance() -> None:
    shared = get_chat_model()
    assert root_agent.model is shared
    for worker in root_agent.sub_agents:
        assert worker.model is shared, f"{worker.name} does not share the chat model"


def test_shared_model_bounds_latency() -> None:
    """A stalled request must not hang forever, and output must be capped."""
    settings = get_settings()
    model = get_chat_model()
    additional = model._additional_args
    assert additional["max_tokens"] == settings.max_output_tokens
    assert additional["timeout"] == settings.chat_timeout_seconds
    assert additional["num_retries"] == 0
    # The hard timeout is enforced at the ADK layer, not only by LiteLLM.
    assert model._timeout_seconds == settings.chat_timeout_seconds


def test_hard_timeout_cancels_a_slow_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider that ignores LiteLLM's timeout must still be bounded."""

    async def never_returns(*args, **kwargs):
        await asyncio.sleep(30)

    model = get_chat_model()
    monkeypatch.setattr(model.llm_client, "acompletion", never_returns)
    original = model._timeout_seconds
    model._timeout_seconds = 0.2
    request = LlmRequest(
        model=model.model,
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
    )

    async def consume() -> None:
        async for _ in model.generate_content_async(request):
            pass

    try:
        with pytest.raises(TimeoutError, match="exceeded"):
            asyncio.run(consume())
    finally:
        model._timeout_seconds = original


def test_code_explainer_has_mcp_toolset_attached() -> None:
    explainer = next(a for a in root_agent.sub_agents if a.name == "code_explainer")
    tool_names = {type(tool).__name__ for tool in explainer.tools}
    # The Filesystem MCP toolset plus the deterministic read_project_file tool.
    assert "McpToolset" in tool_names, f"MCP toolset missing; tools={tool_names}"
    assert any(callable(tool) for tool in explainer.tools)


@pytest.mark.parametrize("name", WORKER_NAMES)
def test_workers_cannot_hand_off_to_each_other(name: str) -> None:
    worker = next(a for a in root_agent.sub_agents if a.name == name)
    assert worker.disallow_transfer_to_parent is True
    assert worker.disallow_transfer_to_peers is True


def test_build_worker_agents_returns_four_distinct_agents() -> None:
    workers = build_worker_agents()
    assert len(workers) == 4
    assert len({id(w) for w in workers}) == 4


def test_worker_instructions_are_specific() -> None:
    for worker in root_agent.sub_agents:
        assert worker.instruction and "you are" in worker.instruction.lower()


def test_mapper_refuses_arbitrary_files_and_absent_technology() -> None:
    """Regression: Kubernetes question must not trigger inspect_imports(main.py)."""
    mapper = next(a for a in root_agent.sub_agents if a.name == "architecture_mapper")
    assert "Never\n   pick a file arbitrarily" in mapper.instruction
    assert "Kubernetes" in mapper.instruction
    assert "Not found in the indexed code" in mapper.instruction


def test_coordinator_routes_where_is_questions_to_the_searcher() -> None:
    instruction = root_agent.instruction
    assert 'ANY "where is ...?" question' in instruction
    assert "never route such a question to `architecture_mapper`" in instruction.lower()


@pytest.mark.parametrize("name", WORKER_NAMES)
def test_workers_have_call_limit_callbacks(name: str) -> None:
    worker = next(a for a in root_agent.sub_agents if a.name == name)
    assert worker.before_tool_callback is not None
    assert worker.before_model_callback is not None


def test_coordinator_has_before_agent_callback() -> None:
    assert root_agent.before_agent_callback is not None


def test_tool_call_limit_enforcement() -> None:
    tool = MagicMock()
    tool.name = "get_project_overview"

    ctx = MagicMock()
    ctx.agent_name = "onboarding_guide"
    ctx.invocation_id = "test-inv-1"

    # First call should succeed (returns None)
    res1 = enforce_tool_call_limit(tool, {}, ctx)
    assert res1 is None

    # Duplicate call in same invocation should be blocked
    res2 = enforce_tool_call_limit(tool, {}, ctx)
    assert res2 is not None
    assert "already called" in res2["result"]


def test_model_call_limit_disables_tools_when_budget_met() -> None:
    ctx = MagicMock()
    ctx.agent_name = "onboarding_guide"
    ctx.invocation_id = "test-inv-model-1"

    # Simulate 1 tool call made (max for onboarding_guide is 1)
    _invocation_tool_counts[ctx.invocation_id] = {ctx.agent_name: 1}

    def dummy_func():
        pass

    tool = FunctionTool(dummy_func)
    req = LlmRequest(
        model="test",
        contents=[],
        config=types.GenerateContentConfig(tools=[dummy_func]),
        tools_dict={"dummy_func": tool},
    )

    enforce_model_call_limit(ctx, req)
    assert req.config.tools is None
    assert req.tools_dict == {}


@pytest.mark.parametrize("name", WORKER_NAMES)
def test_workers_clean_up_invocation_state(name: str) -> None:
    # Cleanup belongs on the worker: the coordinator's after_agent_callback
    # fires before the worker runs, so it would wipe the counters too early.
    worker = next(a for a in root_agent.sub_agents if a.name == name)
    assert worker.after_agent_callback is not None


def test_setup_invocation_limits_caps_adk_default() -> None:
    # ADK's RunConfig defaults to 500 calls; MAX_LLM_CALLS must still apply.
    invocation = MagicMock()
    invocation.run_config = RunConfig(max_llm_calls=500)
    ctx = MagicMock()
    ctx._invocation_context = invocation

    setup_invocation_limits(ctx)

    assert invocation.run_config.max_llm_calls == get_settings().max_llm_calls


def test_reset_invocation_state_clears_all_counters() -> None:
    invocation_id = "test-reset-1"
    _invocation_tool_counts[invocation_id] = {"onboarding_guide": 1}
    _invocation_seen_calls[invocation_id] = {("onboarding_guide", "tool", "{}")}
    _invocation_model_counts[invocation_id] = {"onboarding_guide": 1}

    ctx = MagicMock()
    ctx.invocation_id = invocation_id

    reset_invocation_state(ctx)

    assert invocation_id not in _invocation_tool_counts
    assert invocation_id not in _invocation_seen_calls
    assert invocation_id not in _invocation_model_counts
