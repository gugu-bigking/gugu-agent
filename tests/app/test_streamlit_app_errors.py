"""Tests for the gugu-agent error banner and retry behavior."""

from collections.abc import AsyncGenerator
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

from client import StreamNetworkError, StreamServerError
from schema import ChatMessage


def test_stream_error_shows_typed_banner_and_retry(mock_agent_client):
    mock_agent_client.astream = Mock(
        side_effect=StreamNetworkError("Connection refused: 127.0.0.1:8080")
    )

    at = AppTest.from_file("../../src/streamlit_app.py").run()
    at.chat_input[0].set_value("hi").run()

    # The error banner must surface the same string the previous UI did.
    assert any(
        "Error generating response" in e.value
        and "Connection refused" in e.value
        for e in at.error
    )

    # Plus a friendly label and a Retry button.
    info = " ".join(m.value for m in at.markdown)
    assert "Can't reach the service" in info
    assert "Retry" in {b.label for b in at.button}


def test_retry_replays_last_user_input(mock_agent_client):
    """The Retry button should put the last user input back on the queue."""
    attempts: list[str] = []

    def astream(**kwargs):
        attempts.append(kwargs["message"])
        if len(attempts) == 1:
            raise StreamServerError("503 bad gateway")
        async def gen() -> AsyncGenerator[ChatMessage, None]:
            yield ChatMessage(type="ai", content=f"echo {kwargs['message']}")
        return gen()

    mock_agent_client.astream = Mock(side_effect=astream)

    at = AppTest.from_file("../../src/streamlit_app.py").run()
    at.chat_input[0].set_value("retry me").run()
    # First attempt fails — banner is visible.
    retry_btn = next(b for b in at.button if b.label == "Retry")
    retry_btn.click().run()

    assert attempts == ["retry me", "retry me"], (
        "Retry should re-send the same user input, in order, on a fresh stream"
    )
    # The second attempt succeeded, so no banner.
    assert not at.error
