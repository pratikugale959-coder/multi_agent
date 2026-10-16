"""Tests for the agent graph: one coordinator, four workers, shared model.

These tests construct real ADK ``Agent`` objects but never run them, so no model
inference or network access is required.
"""

from __future__ import annotations

import pytest

from code_helper.agent import root_agent
from code_helper.agents import build_worker_agents
from code_helper.config import get_chat_model

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
