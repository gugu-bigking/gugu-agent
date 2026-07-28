"""Tests for gugu-agent attachment pre-processing and tool-retry behavior."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents import gugu_agent as gugu_agent_module
from agents.gugu_agent import (
    _is_transient_tool_error,
    _LARGE_IMAGE_B64_BYTES,
    _parse_data_url,
    _resolve_attachments,
    RetryingToolNode,
)


# --- helpers ---------------------------------------------------------------


def _png_b64(n_bytes: int) -> str:
    """Build a base64 string of `n_bytes` decoded length (PNGs start 0x89..)."""
    raw = b"\x89PNG" + b"\x00" * max(0, n_bytes - 4)
    return base64.b64encode(raw).decode("ascii")


def _image_block(b64: str, mime: str = "image/png") -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


def _human_msg(blocks, msg_id: str = "h1") -> HumanMessage:
    return HumanMessage(content=blocks, id=msg_id)


# --- _parse_data_url -------------------------------------------------------


def test_parse_data_url_extracts_png():
    mime, payload = _parse_data_url("data:image/png;base64,abcd")
    assert mime == "image/png"
    assert payload == "abcd"


def test_parse_data_url_non_data_url_returns_none():
    assert _parse_data_url("https://example.com/a.png") == (None, None)


def test_parse_data_url_non_base64_returns_none_payload():
    mime, payload = _parse_data_url("data:image/png;charset=utf-8,hi")
    assert mime == "image/png"
    assert payload is None


# --- _resolve_attachments --------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_strips_large_image_and_uses_description():
    large_b64 = _png_b64(_LARGE_IMAGE_B64_BYTES + 1024)
    state = {"messages": [_human_msg([_image_block(large_b64)])]}

    with patch.object(
        gugu_agent_module, "understand_image_func", return_value="a chart of revenue"
    ) as desc:
        out = await _resolve_attachments(state)  # type: ignore[arg-type]

    desc.assert_called_once()
    new_msg = out["messages"][0]
    assert isinstance(new_msg, HumanMessage)
    assert new_msg.id == "h1"
    block = new_msg.content[0]
    assert block["type"] == "text"
    assert "a chart of revenue" in block["text"]
    assert "Attached image" in block["text"]


@pytest.mark.asyncio
async def test_resolve_keeps_small_image_untouched():
    small_b64 = _png_b64(1024)  # well under threshold
    original = _human_msg([_image_block(small_b64)])
    state = {"messages": [original]}

    with patch.object(gugu_agent_module, "understand_image_func") as desc:
        out = await _resolve_attachments(state)  # type: ignore[arg-type]

    desc.assert_not_called()
    assert out == {}  # no state change → caller keeps original message


@pytest.mark.asyncio
async def test_resolve_keeps_image_when_understand_image_fails():
    large_b64 = _png_b64(_LARGE_IMAGE_B64_BYTES + 1024)
    block = _image_block(large_b64)
    state = {"messages": [_human_msg([block])]}

    with patch.object(
        gugu_agent_module,
        "understand_image_func",
        side_effect=RuntimeError("openai down"),
    ):
        out = await _resolve_attachments(state)  # type: ignore[arg-type]

    assert out == {}  # falls back to leaving original block in place


@pytest.mark.asyncio
async def test_resolve_noop_for_text_human_message():
    state = {"messages": [_human_msg("just text")]}

    out = await _resolve_attachments(state)  # type: ignore[arg-type]

    assert out == {}


@pytest.mark.asyncio
async def test_resolve_noop_for_ai_message():
    state = {"messages": [AIMessage(content="ok", id="a1")]}

    out = await _resolve_attachments(state)  # type: ignore[arg-type]

    assert out == {}


@pytest.mark.asyncio
async def test_resolve_replaces_file_block_with_extracted_text():
    state = {
        "messages": [
            _human_msg(
                [{"type": "file", "file": {"filename": "notes.md", "data_b64": "YWhvag=="}}]
            )
        ]
    }

    with patch.object(
        gugu_agent_module, "understand_document_func", return_value="# heading\nbody"
    ):
        out = await _resolve_attachments(state)  # type: ignore[arg-type]

    block = out["messages"][0].content[0]
    assert block["type"] == "text"
    assert "notes.md" in block["text"]
    assert "heading" in block["text"]


@pytest.mark.asyncio
async def test_resolve_handles_remote_image_url():
    remote = {"type": "image_url", "image_url": {"url": "https://x.com/a.png"}}
    state = {"messages": [_human_msg([remote])]}

    out = await _resolve_attachments(state)  # type: ignore[arg-type]

    assert out == {}


# --- _is_transient_tool_error ----------------------------------------------


def test_transient_detection_connection_error():
    assert _is_transient_tool_error(ConnectionError("boom"))


def test_transient_detection_timeout_error():
    assert _is_transient_tool_error(TimeoutError())


def test_transient_detection_empty_mcp_error():
    class McpError(Exception):
        pass

    assert _is_transient_tool_error(McpError(""))


def test_non_transient_mcp_error_with_message():
    class McpError(Exception):
        pass

    assert not _is_transient_tool_error(McpError("bad params"))


def test_non_transient_value_error():
    assert not _is_transient_tool_error(ValueError("nope"))


def test_transient_detection_message_substrings():
    assert _is_transient_tool_error(RuntimeError("Connection reset by peer"))
    assert _is_transient_tool_error(RuntimeError("EOF occurred in violation of protocol"))
    assert _is_transient_tool_error(RuntimeError("SSL: CERTIFICATE_VERIFY_FAILED"))


# --- RetryingToolNode ------------------------------------------------------


@pytest.mark.asyncio
async def test_retrying_tool_node_succeeds_after_one_transient_failure():
    inner = MagicMock()
    inner.ainvoke = AsyncMock(side_effect=[ConnectionError("reset"), {"messages": []}])
    node = RetryingToolNode.__new__(RetryingToolNode)
    node._node = inner
    node._attempts = 3

    with patch("asyncio.sleep", new=AsyncMock()):
        out = await node({"messages": []})

    assert out == {"messages": []}
    assert inner.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_retrying_tool_node_does_not_retry_value_error():
    inner = MagicMock()
    inner.ainvoke = AsyncMock(side_effect=ValueError("bad schema"))
    node = RetryingToolNode.__new__(RetryingToolNode)
    node._node = inner
    node._attempts = 3

    with patch("asyncio.sleep", new=AsyncMock()) as sleep:
        with pytest.raises(ValueError):
            await node({"messages": []})

    sleep.assert_not_called()
    assert inner.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_retrying_tool_node_exhausts_attempts_and_raises():
    inner = MagicMock()
    inner.ainvoke = AsyncMock(side_effect=ConnectionError("nope"))
    node = RetryingToolNode.__new__(RetryingToolNode)
    node._node = inner
    node._attempts = 3

    with patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(ConnectionError):
            await node({"messages": []})

    assert inner.ainvoke.await_count == 3