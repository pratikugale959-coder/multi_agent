"""Web entry point for the Multi-Agent Code Helper.

Run it with::

    python -m code_helper.web

This serves the same ADK Dev UI as ``adk web`` but adds two reliability fixes
that the stock command does not provide on Windows:

* **SSE keep-alive.** ADK's ``/run_sse`` endpoint writes nothing while a model
  call is in flight. A slow NVIDIA NIM turn can leave the stream silent for
  more than a minute, which browsers and proxies may treat as a dead
  connection and surface as ``TypeError: Failed to fetch``. The middleware
  below writes SSE comment chunks on a timer, so the connection stays warm
  without adding any events to the conversation.
* **UTF-8 console output.** LiteLLM debug logs contain emoji; the default
  Windows code page cannot encode them, which produced noisy
  ``UnicodeEncodeError`` tracebacks.

The chat itself never shows agent thoughts: the shared model wrapper in
:mod:`code_helper.core.config` strips ``thought=True`` parts before ADK can
render them.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from code_helper.core.config import PROJECT_ROOT

#: SSE comment written while the agent is busy. Event consumers ignore lines
#: that do not start with ``data:``, so this never changes the chat transcript.
SSE_KEEP_ALIVE: bytes = b": keep-alive\n\n"

#: Seconds of silence after which a keep-alive comment is sent. Chosen well
#: below common 30 s proxy/browser idle limits.
DEFAULT_HEARTBEAT_SECONDS: float = 10.0


class SseHeartbeatMiddleware:
    """ASGI middleware that keeps ``/run_sse`` responses alive.

    The wrapped application's ``send`` calls are queued and forwarded by a
    single writer task, which also owns the heartbeat timer. Serialising sends
    matters because the ASGI spec does not allow concurrent ``send`` calls.
    """

    def __init__(self, app: Any, interval: float = DEFAULT_HEARTBEAT_SECONDS) -> None:
        if interval <= 0:
            raise ValueError("heartbeat interval must be greater than zero")
        self.app = app
        self.interval = float(interval)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not str(scope.get("path", "")).endswith("/run_sse"):
            await self.app(scope, receive, send)
            return

        messages: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def queue_send(message: dict[str, Any]) -> None:
            await messages.put(message)

        async def pump() -> None:
            started = False
            finished = False
            while True:
                try:
                    message = await asyncio.wait_for(messages.get(), timeout=self.interval)
                except TimeoutError:
                    if started and not finished:
                        await send(
                            {
                                "type": "http.response.body",
                                "body": SSE_KEEP_ALIVE,
                                "more_body": True,
                            }
                        )
                    continue
                if message is None:
                    return
                if message["type"] == "http.response.start":
                    started = True
                await send(message)
                if message["type"] == "http.response.body" and not message.get("more_body", False):
                    finished = True

        async with asyncio.TaskGroup() as group:
            app_task = group.create_task(self.app(scope, receive, queue_send))
            # The sentinel is queued only after every real message, so the
            # writer drains the stream before it stops.
            app_task.add_done_callback(lambda _task: messages.put_nowait(None))
            group.create_task(pump())


def create_app(
    *,
    agents_dir: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    session_service_uri: str = "memory://",
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
) -> Any:
    """Build the ADK web application with the keep-alive middleware attached.

    Sessions default to ``memory://`` so the local demo does not depend on the
    per-agent SQLite file under ``.adk/``. Pass ``--session-service-uri
    sqlite:///...`` (or any URI supported by ADK) to persist chat history
    between server restarts.
    """
    from google.adk.cli.fast_api import get_fast_api_app

    app = get_fast_api_app(
        agents_dir=str(agents_dir or PROJECT_ROOT),
        web=True,
        session_service_uri=session_service_uri,
        host=host,
        bind_host=host,
        port=port,
        auto_create_session=True,
    )
    if heartbeat_seconds <= 0:
        return app
    return SseHeartbeatMiddleware(app, interval=heartbeat_seconds)


def _configure_utf8_stdio() -> None:
    """Make Windows console streams tolerate emoji in dependency logs."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    """Parse command-line options and serve the ADK web UI."""
    parser = argparse.ArgumentParser(
        prog="python -m code_helper.web",
        description="Serve the Multi-Agent Code Helper web UI with SSE keep-alive.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument(
        "--agents-dir",
        default=str(PROJECT_ROOT),
        help="Directory containing the agent package (default: the project root).",
    )
    parser.add_argument(
        "--session-service-uri",
        default="memory://",
        help=(
            "ADK session backend. Defaults to in-memory; use sqlite:///path/to/sessions.db "
            "to persist history across restarts."
        ),
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=DEFAULT_HEARTBEAT_SECONDS,
        help=(
            "Seconds of SSE silence before a keep-alive comment is written "
            f"(default: {DEFAULT_HEARTBEAT_SECONDS:g}; 0 disables it)."
        ),
    )
    args = parser.parse_args(argv)

    _configure_utf8_stdio()

    import uvicorn

    app = create_app(
        agents_dir=args.agents_dir,
        host=args.host,
        port=args.port,
        session_service_uri=args.session_service_uri,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    display_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(f"ADK web UI: http://{display_host}:{args.port}/dev-ui/", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
