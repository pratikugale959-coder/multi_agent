"""Integration check: inspect the real Filesystem MCP server.

Run from the directory containing ``code_helper``::

    python scripts/mcp_list_tools.py

It launches the official Filesystem MCP server through ``npx`` (requires Node.js
and network access on first run), lists the tools the server actually exposes,
checks that our read-only allow-list matches, and reads one file end to end.

This confirms the tool names we filter on in ``code_helper/mcp_filesystem.py``
are real, rather than guessed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_helper.config import get_settings  # noqa: E402
from code_helper.mcp_filesystem import READ_ONLY_MCP_TOOLS, filesystem_server_params  # noqa: E402


async def inspect_server() -> int:
    """Connect to the MCP server, list tools, and read a file."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = filesystem_server_params()
    print("Launching:", params.command, " ".join(params.args))

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            listed = await session.list_tools()
            names = sorted(tool.name for tool in listed.tools)
            print(f"\nServer exposes {len(names)} tools:")
            for name in names:
                marker = " (allowed)" if name in READ_ONLY_MCP_TOOLS else ""
                print(f"  - {name}{marker}")

            missing = [t for t in READ_ONLY_MCP_TOOLS if t not in names]
            if missing:
                print(f"\nWARNING: allow-listed tools not offered by the server: {missing}")
            else:
                print("\nAll allow-listed read-only tools exist on the server.")

            allowed_dirs = await session.call_tool("list_allowed_directories", {})
            text = "".join(getattr(part, "text", "") for part in allowed_dirs.content)
            print("Allowed directories:", text.strip())

            target = get_settings().target_repo_path
            reading = await session.call_tool("read_text_file", {"path": str(target / "config.py")})
            body = "".join(getattr(part, "text", "") for part in reading.content)
            print(f"\nRead {len(body)} characters from target_project/config.py via MCP.")
            print("First line:", body.splitlines()[0] if body else "(empty)")
    return 0 if not missing else 1


def main() -> int:
    """Entry point."""
    return asyncio.run(inspect_server())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
