"""Tests for the gugu-agent queueing and force-interrupt flow.

AppTest runs each script to completion (including internal reruns) within a
single `.run()` call, so a "slow" generator that simply yields once and exits
finishes the turn in the same run — no queueing happens. To exercise the
queue, the generator has to stay paused after its first yield, which makes
the first `.run()` block until AppTest's default timeout fires.
"""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from schema import ChatMessage


def _blocking_stream(go: asyncio.Event) -> AsyncGenerator[ChatMessage, None]:
    """Yield one message, then wait forever for `go` to be set.

    The first `.run()` will block until AppTest's timeout raises; we catch
    that and inspect the session_state to verify queueing happened.
    """

    async def gen():
        yield ChatMessage(type="ai", content="ok")
        await go.wait()  # never set in these tests — guarantees an in-flight turn

    return gen()


@pytest.mark.asyncio
async def test_second_message_is_queued_while_first_runs(mock_agent_client):
    """While the first turn is still in flight, a second submission queues."""

    go = asyncio.Event()
    consumed: list[str] = []

    def astream(**kwargs):
        consumed.append(kwargs["message"])
        return _blocking_stream(go)

    mock_agent_client.astream = Mock(side_effect=astream)

    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=2).run()

    # First turn kicks off — script blocks on the never-completed stream.
    at.chat_input[0].set_value("first").run()

    # By the time AppTest's wait-for-completion raises a timeout, the first
    # turn's astream has been called and the script is paused mid-stream.
    # Releasing `go` lets the in-flight generator finish so we can submit a
    # second message in a clean rerun.
    go.set()

    at.chat_input[0].set_value("second").run()

    assert consumed == ["first"], (
        "Only the first message should have hit the network; second should be queued"
    )
    assert "second" in at.session_state.pending_messages[0]["text"]


@pytest.mark.asyncio
async def test_stop_button_clears_queue_and_flags_cancel(mock_agent_client):
    """Clicking Stop while a turn is in flight clears the queue and flags cancel."""

    go = asyncio.Event()
    mock_agent_client.astream = Mock(side_effect=lambda **_: _blocking_stream(go))

    at = AppTest.from_file("../../src/streamlit_app.py", default_timeout=2).run()
    at.chat_input[0].set_value("first").run()
    at.chat_input[0].set_value("second").run()
    assert len(at.session_state.pending_messages) == 1

    at.sidebar.button[0]  # sanity: still the new-chat button
    stop_buttons = [b for b in at.button if "Stop" in b.label]
    assert stop_buttons, "Stop button should be visible while a turn is in flight"
    stop_buttons[0].click().run()

    go.set()  # unblock the in-flight generator so the run completes cleanly

    assert not at.session_state.pending_messages
    assert at.session_state["cancel_requested"] is True