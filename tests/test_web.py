"""Tests for the web launcher's SSE keep-alive middleware.

The middleware is pure ASGI, so it can be exercised without starting a server,
opening a socket, or loading the ADK web assets.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from code_helper.web import SSE_KEEP_ALIVE, SseHeartbeatMiddleware


def _scope(path: str) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }


async def _receive() -> dict[str, Any]:
    return {"type": "http.disconnect"}


def test_heartbeat_is_written_while_the_app_is_silent() -> None:
    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await asyncio.sleep(0.3)
        await send({"type": "http.response.body", "body": b"data: {}\n\n", "more_body": False})

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def run() -> None:
        middleware = SseHeartbeatMiddleware(app, interval=0.05)
        await middleware(_scope("/run_sse"), _receive, send)

    asyncio.run(run())

    assert sent[0]["type"] == "http.response.start"
    assert sent[-1] == {
        "type": "http.response.body",
        "body": b"data: {}\n\n",
        "more_body": False,
    }
    keep_alives = [
        message
        for message in sent
        if message["type"] == "http.response.body" and message.get("body") == SSE_KEEP_ALIVE
    ]
    assert keep_alives, f"no keep-alive written; messages={sent}"
    assert all(message.get("more_body") is True for message in keep_alives)

    final_index = sent.index(sent[-1])
    assert all(sent.index(message) < final_index for message in keep_alives)


def test_heartbeat_is_not_added_to_other_routes() -> None:
    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b"[]", "more_body": False})

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def run() -> None:
        middleware = SseHeartbeatMiddleware(app, interval=0.05)
        await middleware(_scope("/list-apps"), _receive, send)

    asyncio.run(run())

    assert [message["type"] for message in sent] == [
        "http.response.start",
        "http.response.body",
    ]


def test_heartbeat_interval_must_be_positive() -> None:
    with pytest.raises(ValueError):
        SseHeartbeatMiddleware(lambda scope, receive, send: None, interval=0)
