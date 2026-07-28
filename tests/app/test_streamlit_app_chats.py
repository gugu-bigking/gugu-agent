"""Tests for the gugu-agent sidebar history and chat metadata flow."""

import time
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, Mock

from streamlit.testing.v1 import AppTest

from client import AgentClientError
from schema import ChatHistory, ChatMessage, ChatMetaItem


def _meta(thread_id: str, title: str, updated_at: float) -> ChatMetaItem:
    return ChatMetaItem(
        thread_id=thread_id,
        user_id="u1",
        agent="test-agent",
        title=title,
        preview="",
        created_at=updated_at,
        updated_at=updated_at,
    )


def test_sidebar_lists_history(mock_agent_client):
    chats = [
        _meta("t1", "What is BM25?", time.time()),
        _meta("t2", "Find the bug in rag", time.time() - 60),
    ]
    mock_agent_client.alist_chats = AsyncMock(return_value=chats)

    at = AppTest.from_file("../../src/streamlit_app.py").run()
    # The new chat button is button[0]; the two history items are button[1], button[2].
    assert at.sidebar.button[0].label.endswith("New chat")
    history_labels = [at.sidebar.button[i].label for i in range(1, 3)]
    assert any("BM25" in label for label in history_labels)
    assert any("rag" in label for label in history_labels)


def test_switch_chat_replaces_thread_and_history(mock_agent_client):
    chats = [
        _meta("thread-a", "About apples", time.time()),
        _meta("thread-b", "About bananas", time.time() - 30),
    ]
    mock_agent_client.alist_chats = AsyncMock(return_value=chats)
    mock_agent_client.get_history.return_value = ChatHistory(
        messages=[ChatMessage(type="ai", content="Bananas are great")]
    )

    at = AppTest.from_file("../../src/streamlit_app.py").run()
    at.sidebar.button[2].click().run()  # thread-b

    assert at.session_state.thread_id == "thread-b"
    # The history is rendered in chat_message widgets.
    assert any("Bananas" in m.markdown[0].value for m in at.chat_message)
    mock_agent_client.get_history.assert_called_with(thread_id="thread-b", agent="test-agent")


def test_new_chat_resets_state(mock_agent_client):
    at = AppTest.from_file("../../src/streamlit_app.py").run()
    first_thread = at.session_state.thread_id
    at.sidebar.button[0].click().run()  # New chat
    assert at.session_state.thread_id != first_thread
    assert at.session_state.messages == []


def test_first_message_registers_chat_metadata(mock_agent_client):
    """The first user turn on a fresh thread should call acreate_chat."""

    async def stream() -> AsyncGenerator[ChatMessage, None]:
        yield ChatMessage(type="ai", content="pong")

    mock_agent_client.astream = Mock(return_value=stream())
    captured: list[dict] = []

    async def capture(**kwargs):
        captured.append(kwargs)
        return _meta("server-thread", "pinged", time.time())

    mock_agent_client.acreate_chat = capture

    at = AppTest.from_file("../../src/streamlit_app.py").run()
    at.chat_input[0].set_value("ping").run()

    assert captured, "acreate_chat should have been called"
    assert captured[0]["first_message"] == "ping"
    assert captured[0]["agent"] == "test-agent"
    # The server-minted thread id should land in the URL params.
    assert at.query_params["thread_id"] == ["server-thread"]


def test_chat_meta_list_failure_is_silent(mock_agent_client):
    """A failure to load history must not break the app — sidebar shows a hint."""
    mock_agent_client.alist_chats = AsyncMock(
        side_effect=AgentClientError("boom")
    )
    at = AppTest.from_file("../../src/streamlit_app.py").run()
    assert not at.exception
    # The empty-state caption should mention the absence of saved chats.
    assert any("No saved chats" in c.value for c in at.sidebar.caption)
