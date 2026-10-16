"""Tests for the Filesystem MCP wiring (no subprocess is spawned)."""

from __future__ import annotations

from pathlib import Path

from code_helper.config import Settings
from code_helper.mcp_filesystem import (
    READ_ONLY_MCP_TOOLS,
    build_filesystem_toolset,
    describe_mcp_configuration,
    filesystem_server_params,
    safe_subprocess_env,
)


def test_safe_env_strips_secrets() -> None:
    base = {
        "PATH": "/usr/bin",
        "NVIDIA_API_KEY": "secret",
        "PINECONE_API_KEY": "secret",
        "MY_PASSWORD": "secret",
        "GITHUB_TOKEN": "secret",
        "SystemRoot": "C:\\Windows",
        "SOME_RANDOM_VAR": "kept-out-because-not-allowlisted",
    }
    env = safe_subprocess_env(base)
    assert env["PATH"] == "/usr/bin"
    assert env["SystemRoot"] == "C:\\Windows"
    assert "NVIDIA_API_KEY" not in env
    assert "PINECONE_API_KEY" not in env
    assert "MY_PASSWORD" not in env
    assert "GITHUB_TOKEN" not in env
    assert "SOME_RANDOM_VAR" not in env


def test_read_only_filter_excludes_write_tools() -> None:
    forbidden = {"write_file", "edit_file", "create_directory", "move_file", "delete_file"}
    assert forbidden.isdisjoint(set(READ_ONLY_MCP_TOOLS))


def test_server_params_target_the_project_root(settings: Settings) -> None:
    params = filesystem_server_params(settings.target_repo_path)
    assert params.command == "npx"
    assert "-y" in params.args
    assert "@modelcontextprotocol/server-filesystem" in params.args
    assert str(settings.target_repo_path) in params.args


def test_toolset_builds_and_exposes_get_tools() -> None:
    toolset = build_filesystem_toolset()
    assert toolset is not None
    assert hasattr(toolset, "get_tools")


def test_describe_mcp_configuration_is_restricted() -> None:
    info = describe_mcp_configuration()
    root = Path(str(info["allowed_root"]))
    assert root.name == "target_project"
    assert info["command"] == "npx"
    # Secrets must not appear in the sanitised environment key list.
    assert not any("KEY" in key.upper() for key in info["env_keys_passed"])
