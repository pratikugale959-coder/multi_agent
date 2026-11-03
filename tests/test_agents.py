"""Tests for the agent graph: one coordinator, four workers, shared model.

These tests construct real ADK ``Agent`` objects but never run them, so no model
inference or network access is required.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from google.adk.agents.run_config import RunConfig
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
from litellm import ModelResponse

from code_helper.agents.callbacks import strip_reasoning
from code_helper.agents.coordinator import root_agent, setup_invocation_limits
from code_helper.agents.workers import (
    _invocation_model_counts,
    _invocation_seen_calls,
    _invocation_tool_counts,
    build_worker_agents,
    enforce_model_call_limit,
    enforce_tool_call_limit,
    reset_invocation_state,
)
from code_helper.core.config import (
    TIMEOUT_FALLBACK_TEXT,
    _drop_thought_parts,
    get_chat_model,
    get_settings,
)

WORKER_NAMES = ["code_searcher", "code_explainer", "architecture_mapper", "onboarding_guide"]


@pytest.fixture(autouse=True)
def _reset_model_breaker():
    """Keep circuit-breaker state from leaking between tests."""
    model = get_chat_model()
    model.reset_health()
    yield
    model.reset_health()


def _completion(text: str) -> ModelResponse:
    """Build a minimal LiteLLM response understood by ADK's adapter."""
    return ModelResponse(
        id="chatcmpl-test",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        created=0,
        model="test-model",
        object="chat.completion",
    )


def test_coordinator_exists_and_is_named() -> None:
    assert root_agent.name == "coordinator"


def test_agent_module_exposes_the_same_root_agent() -> None:
    """``adk web`` only discovers directories that contain ``agent.py``."""
    from code_helper.agent import root_agent as module_root_agent

    assert module_root_agent is root_agent


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
    if settings.reasoning_effort:
        assert additional["extra_body"] == {"reasoning_effort": settings.reasoning_effort}
    # The hard timeout is enforced at the ADK layer, not only by LiteLLM.
    assert model._timeout_seconds == settings.chat_timeout_seconds
    # Failover configuration is present and bounded too.
    assert model._fallback_timeout_seconds == settings.fallback_chat_timeout_seconds
    assert model._primary_cooldown_seconds == settings.primary_cooldown_seconds
    expected_fallbacks = [
        model_id
        for model_id in settings.fallback_chat_model_ids
        if model_id != settings.chat_model_id
    ]
    assert len(model._fallback_models) == len(expected_fallbacks)


def _collect_text(model, request) -> str:
    async def consume() -> str:
        chunks: list[str] = []
        async for item in model.generate_content_async(request):
            for part in item.content.parts:
                if part.text:
                    chunks.append(part.text)
        return "".join(chunks)

    return asyncio.run(consume())


def test_hard_timeout_cancels_a_slow_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider that ignores LiteLLM's timeout must still be bounded."""

    async def never_returns(*args, **kwargs):
        await asyncio.sleep(30)

    model = get_chat_model()
    monkeypatch.setattr(model.llm_client, "acompletion", never_returns)
    for fallback in model._fallback_models:
        monkeypatch.setattr(fallback.llm_client, "acompletion", never_returns)
    monkeypatch.setattr(model, "_timeout_seconds", 0.2)
    monkeypatch.setattr(model, "_fallback_timeout_seconds", 0.2)
    request = LlmRequest(
        model=model.model,
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
    )

    text = _collect_text(model, request)
    assert text == TIMEOUT_FALLBACK_TEXT


def test_primary_timeout_falls_back_to_a_working_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user gets an answer from the fallback instead of a timeout notice."""
    model = get_chat_model()
    assert model._fallback_models, "no fallback models configured"

    async def never_returns(*args, **kwargs):
        await asyncio.sleep(30)

    async def working_fallback(*args, **kwargs):
        return _completion("answer from the fallback model")

    monkeypatch.setattr(model.llm_client, "acompletion", never_returns)
    for fallback in model._fallback_models:
        monkeypatch.setattr(fallback.llm_client, "acompletion", working_fallback)
    monkeypatch.setattr(model, "_timeout_seconds", 0.2)

    request = LlmRequest(
        model=model.model,
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
    )

    text = _collect_text(model, request)
    assert text == "answer from the fallback model"
    # The breaker remembers the failure so the next step skips the primary.
    assert model._primary_unavailable_until > time.monotonic()


def test_primary_error_falls_back_to_a_working_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = get_chat_model()
    assert model._fallback_models, "no fallback models configured"

    async def provider_error(*args, **kwargs):
        raise RuntimeError("provider is down")

    async def working_fallback(*args, **kwargs):
        return _completion("recovered answer")

    monkeypatch.setattr(model.llm_client, "acompletion", provider_error)
    for fallback in model._fallback_models:
        monkeypatch.setattr(fallback.llm_client, "acompletion", working_fallback)

    request = LlmRequest(
        model=model.model,
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
    )

    assert _collect_text(model, request) == "recovered answer"
    assert model._primary_unavailable_until > time.monotonic()


def test_breaker_skips_the_primary_during_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    model = get_chat_model()
    assert model._fallback_models, "no fallback models configured"
    calls: list[str] = []

    async def primary_must_not_run(*args, **kwargs):
        calls.append("primary")
        raise AssertionError("the primary must be skipped during cooldown")

    async def working_fallback(*args, **kwargs):
        calls.append("fallback")
        return _completion("fallback while cooling down")

    monkeypatch.setattr(model.llm_client, "acompletion", primary_must_not_run)
    for fallback in model._fallback_models:
        monkeypatch.setattr(fallback.llm_client, "acompletion", working_fallback)
    model._primary_unavailable_until = time.monotonic() + 60

    request = LlmRequest(
        model=model.model,
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
    )

    assert _collect_text(model, request) == "fallback while cooling down"
    assert calls == ["fallback"]


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


def test_code_searcher_can_refine_a_weak_search() -> None:
    from code_helper.agents.workers import get_max_tool_calls_for_agent

    assert get_max_tool_calls_for_agent("code_searcher") >= 2
    searcher = next(a for a in root_agent.sub_agents if a.name == "code_searcher")
    assert "at most 3 times" in searcher.instruction
    assert "never with the\n   same query twice" in searcher.instruction


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


def test_drop_thought_parts_keeps_visible_text() -> None:
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="private plan", thought=True),
                types.Part(text="The answer"),
            ],
        )
    )

    dropped = _drop_thought_parts(response)

    assert dropped is False
    assert [part.text for part in response.content.parts] == ["The answer"]


def test_drop_thought_parts_discards_a_thought_only_response() -> None:
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text="private plan", thought=True)],
        )
    )

    assert _drop_thought_parts(response) is True


def test_drop_thought_parts_leaves_normal_responses_untouched() -> None:
    parts = [types.Part(text="The answer")]
    response = LlmResponse(content=types.Content(role="model", parts=parts))

    assert _drop_thought_parts(response) is False
    assert response.content.parts == parts


def test_streaming_model_never_yields_thought_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The user sees the final answer only, even while reasoning streams."""
    requested_streams: list[bool] = []

    async def fake_stream(self, llm_request, stream=False):
        requested_streams.append(stream)
        yield LlmResponse(
            partial=True,
            content=types.Content(
                role="model", parts=[types.Part(text="let me think", thought=True)]
            ),
        )
        yield LlmResponse(
            partial=True,
            content=types.Content(role="model", parts=[types.Part(text="answer")]),
        )
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="final reasoning", thought=True),
                    types.Part(text="answer"),
                ],
            )
        )

    monkeypatch.setattr(LiteLlm, "generate_content_async", fake_stream)
    model = get_chat_model()
    request = LlmRequest(
        model=model.model,
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
    )

    async def collect() -> list[str]:
        texts: list[str] = []
        async for item in model.generate_content_async(request, stream=True):
            texts.extend(part.text for part in item.content.parts if part.text)
        return texts

    assert asyncio.run(collect()) == ["answer", "answer"]
    # NIM's streaming tool-call parser is unreliable; the wrapper asks for the
    # complete response even when ADK requests a stream.
    assert requested_streams == [False]


def test_strip_reasoning_callback_removes_thought_parts() -> None:
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="private plan", thought=True),
                types.Part(text="The answer"),
            ],
        )
    )

    cleaned = strip_reasoning(None, response)

    assert cleaned is response
    assert [part.text for part in cleaned.content.parts] == ["The answer"]


def test_strip_reasoning_callback_discards_thought_only_response() -> None:
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text="private plan", thought=True)],
        )
    )

    cleaned = strip_reasoning(None, response)

    assert cleaned is response
    assert cleaned.content.parts == []
