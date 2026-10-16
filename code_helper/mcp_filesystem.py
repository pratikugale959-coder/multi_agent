"""Filesystem MCP integration for the Code Explainer agent.

We connect Google ADK's supported MCP client to the official Filesystem MCP
server (``@modelcontextprotocol/server-filesystem``) launched through ``npx``.

Security decisions
------------------
* The server is given **exactly one** allowed directory: the target project root.
  It never sees the drive root, the ``code_helper`` package or the ``.env`` file.
* A ``tool_filter`` exposes only read-only operations. Write/edit/move/create
  tools are never surfaced to the agent.
* The subprocess receives a **sanitised environment**: variables whose names look
  like secrets (``*KEY*``, ``*TOKEN*``, ``*SECRET*``, ``*PASSWORD*``) are removed
  so the child process cannot read the API keys even transitively.
* MCP server instantiation is guarded so the agent still imports if ``npx`` is
  unavailable; in that case the Code Explainer falls back to the deterministic
  ``read_project_file`` tool.
"""

from __future__ import annotations

import os
from pathlib import Path

from code_helper.config import get_settings
from code_helper.paths import project_root

#: Only these environment variables are passed to the npx subprocess.
_SAFE_ENV_KEYS = (
    "PATH",
    "PATHEXT",
    "SystemRoot",
    "SYSTEMROOT",
    "windir",
    "COMSPEC",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMDATA",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "OS",
    "LANG",
    "NODE_PATH",
)

#: Names that always indicate a secret and must never reach the child process.
_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")

#: MCP filesystem tools we deliberately expose (read-only subset). The names are
#: verified against the server by ``scripts/mcp_list_tools.py``.
READ_ONLY_MCP_TOOLS: tuple[str, ...] = (
    "read_file",
    "read_text_file",
    "read_media_file",
    "read_multiple_files",
    "list_directory",
    "list_directory_with_sizes",
    "directory_tree",
    "search_files",
    "get_file_info",
    "list_allowed_directories",
)


def safe_subprocess_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of the environment with secret-looking variables removed.

    Only allow-listed keys survive, and any key containing a secret hint is
    dropped regardless. This is unit-testable without spawning a process.
    """
    env = dict(base_env if base_env is not None else os.environ)
    safe: dict[str, str] = {}
    for key, value in env.items():
        if any(hint in key.upper() for hint in _SECRET_HINTS):
            continue
        if key in _SAFE_ENV_KEYS:
            safe[key] = value
    return safe


def filesystem_server_params(root: Path | None = None):
    """Build the ``StdioServerParameters`` for the Filesystem MCP server."""
    from mcp import StdioServerParameters

    allowed = Path(root or project_root()).resolve()
    return StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(allowed)],
        env=safe_subprocess_env(),
    )


def build_filesystem_toolset(root: Path | None = None):
    """Create the read-only Filesystem MCP toolset via Google ADK.

    Returns:
        A configured ``McpToolset``, or ``None`` if MCP classes are unavailable.
    """
    from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams

    connection = StdioConnectionParams(server_params=filesystem_server_params(root), timeout=30.0)
    return McpToolset(
        connection_params=connection,
        tool_filter=list(READ_ONLY_MCP_TOOLS),
        # Read-only tools never need confirmation prompts.
        require_confirmation=False,
    )


def filesystem_toolset_or_none(root: Path | None = None, verbose: bool = True):
    """Best-effort wrapper used by the agent factory.

    If the MCP client libraries are missing or the toolset cannot be constructed,
    we log a warning and return ``None`` so the Code Explainer still works with
    its deterministic ``read_project_file`` tool.
    """
    if os.getenv("DISABLE_MCP_FILESYSTEM", "").strip() in {"1", "true", "True"}:
        if verbose:
            print("[mcp] Filesystem MCP disabled via DISABLE_MCP_FILESYSTEM.")
        return None
    try:
        return build_filesystem_toolset(root)
    except Exception as exc:  # noqa: BLE001 - never let MCP break agent import
        if verbose:
            print(
                f"[mcp] Could not build the Filesystem MCP toolset ({exc}). "
                "The Code Explainer will use read_project_file instead."
            )
        return None


def describe_mcp_configuration() -> dict[str, object]:
    """Return a redacted description of the MCP setup (safe to print)."""
    settings = get_settings()
    root = project_root(settings)
    return {
        "server": "@modelcontextprotocol/server-filesystem",
        "command": "npx",
        "allowed_root": str(root),
        "read_only_tools": list(READ_ONLY_MCP_TOOLS),
        "env_keys_passed": sorted(safe_subprocess_env().keys()),
        "disabled": os.getenv("DISABLE_MCP_FILESYSTEM", "").strip() in {"1", "true", "True"},
    }
