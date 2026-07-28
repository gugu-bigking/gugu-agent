"""Tests for the typed stream errors and chat list methods on AgentClient."""

import json
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from client import (
    AgentClient,
    StreamClientError,
    StreamInterruptedError,
    StreamNetworkError,
    StreamServerError,
    StreamTimeoutError,
)


@pytest.fixture
def client() -> AgentClient:
    c = AgentClient.__new__(AgentClient)  # bypass __init__ (no /info fetch)
    c.base_url = "http://0.0.0.0"
    c.auth_secret = None
    c.timeout = 5.0
    c.info = None
    c.agent = "gugu-agent"
    return c


@asynccontextmanager
async def _fake_stream_cm(aiter_lines, *, raise_for_status: Any = None):
    """Build an async context manager that mimics httpx's stream response."""

    class _Resp:
        def __init__(self) -> None:
            self._lines = aiter_lines
            self.raise_for_status_called = False

        async def aiter_lines(self):
            async for line in self._lines:
                yield line

        def raise_for_status(self) -> None:
            self.raise_for_status_called = True
            if raise_for_status is not None:
                raise raise_for_status

    resp = _Resp()
    yield resp


def _http_error(status_code: int, body: str = "boom") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://0.0.0.0/stream")
    response = httpx.Response(status_code, text=body, request=request)
    return httpx.HTTPStatusError("err", request=request, response=response)


@pytest.mark.asyncio
async def test_astream_5xx_becomes_stream_server_error(client):
    async def _lines():
        return
        yield  # pragma: no cover — make this an async generator

    cm = _fake_stream_cm(_lines(), raise_for_status=_http_error(503, "upstream down"))
    with patch("httpx.AsyncClient.stream", return_value=cm):
        with pytest.raises(StreamServerError) as exc_info:
            async for _ in client.astream("hi"):
                pass
        assert "503" in str(exc_info.value)


@pytest.mark.asyncio
async def test_astream_4xx_becomes_stream_client_error(client):
    async def _lines():
        return
        yield  # pragma: no cover

    cm = _fake_stream_cm(_lines(), raise_for_status=_http_error(400, "bad model"))
    with patch("httpx.AsyncClient.stream", return_value=cm):
        with pytest.raises(StreamClientError) as exc_info:
            async for _ in client.astream("hi"):
                pass
        assert "400" in str(exc_info.value)


@pytest.mark.asyncio
async def test_astream_timeout_becomes_stream_timeout_error(client):
    with patch("httpx.AsyncClient.stream", side_effect=httpx.ConnectTimeout("read timeout")):
        with pytest.raises(StreamTimeoutError):
            async for _ in client.astream("hi"):
                pass


@pytest.mark.asyncio
async def test_astream_connect_error_becomes_stream_network_error(client):
    with patch("httpx.AsyncClient.stream", side_effect=httpx.ConnectError("connection refused")):
        with pytest.raises(StreamNetworkError):
            async for _ in client.astream("hi"):
                pass


@pytest.mark.asyncio
async def test_astream_remote_protocol_error_becomes_stream_interrupted(client):
    async def _failing():
        raise httpx.RemoteProtocolError("eof")
        yield  # pragma: no cover

    cm = _fake_stream_cm(_failing())
    with patch("httpx.AsyncClient.stream", return_value=cm):
        with pytest.raises((StreamInterruptedError, StreamNetworkError)):
            async for _ in client.astream("hi"):
                pass


@pytest.mark.asyncio
async def test_astream_no_done_marker_raises_stream_interrupted(client):
    async def _one_line():
        yield f"data: {json.dumps({'type': 'token', 'content': 'hi'})}"
        # No [DONE] line.

    cm = _fake_stream_cm(_one_line())
    with patch("httpx.AsyncClient.stream", return_value=cm):
        with pytest.raises(StreamInterruptedError):
            async for _ in client.astream("hello"):
                pass


@pytest.mark.asyncio
async def test_alist_chats_returns_metadata(client):
    payload = [
        {
            "thread_id": "t1",
            "user_id": "u1",
            "agent": "gugu-agent",
            "title": "Hello",
            "preview": "",
            "created_at": 0.0,
            "updated_at": 1.0,
        }
    ]

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict[str, Any]]:
            return payload

    with patch("httpx.AsyncClient.get", return_value=_Resp()):
        items = await client.alist_chats("u1", agent="gugu-agent")
    assert len(items) == 1
    assert items[0].thread_id == "t1"
    assert items[0].title == "Hello"


@pytest.mark.asyncio
async def test_acreate_chat_posts_payload_and_returns_item(client):
    body = {
        "thread_id": "t-new",
        "user_id": "u1",
        "agent": "gugu-agent",
        "title": "What is BM25?",
        "preview": "What is BM25?",
        "created_at": 0.0,
        "updated_at": 0.0,
    }

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return body

    with patch("httpx.AsyncClient.post", return_value=_Resp()):
        item = await client.acreate_chat(
            user_id="u1", agent="gugu-agent", first_message="What is BM25?"
        )
    assert item.thread_id == "t-new"
    assert item.title == "What is BM25?"


@pytest.mark.asyncio
async def test_aupdate_chat_patches_fields(client):
    body = {
        "thread_id": "t1",
        "user_id": "u1",
        "agent": "gugu-agent",
        "title": "renamed",
        "preview": "",
        "created_at": 0.0,
        "updated_at": 1.0,
    }

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return body

    with patch("httpx.AsyncClient.patch", return_value=_Resp()):
        item = await client.aupdate_chat("t1", title="renamed")
    assert item.title == "renamed"


@pytest.mark.asyncio
async def test_atouch_chat_swallows_404(client):
    class _Resp:
        status_code = 404

        def raise_for_status(self) -> None:
            raise AssertionError("should not be called for 404")

    with patch("httpx.AsyncClient.post", return_value=_Resp()):
        await client.atouch_chat("missing", preview="x")  # no raise


@pytest.mark.asyncio
async def test_atouch_chat_swallows_network_error(client):
    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("offline")):
        await client.atouch_chat("t1")  # no raise
