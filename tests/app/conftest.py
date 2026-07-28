from unittest.mock import patch

import pytest

from schema import AgentInfo, ServiceMetadata
from schema.models import OpenAIModelName


@pytest.fixture
def mock_agent_client(mock_env):
    """Fixture for creating a mock AgentClient with a clean environment."""

    mock_info = ServiceMetadata(
        default_agent="test-agent",
        agents=[
            AgentInfo(key="test-agent", description="Test agent"),
            AgentInfo(key="chatbot", description="Chatbot"),
        ],
        default_model=OpenAIModelName.GPT_5_NANO,
        models=[OpenAIModelName.GPT_5_NANO, OpenAIModelName.GPT_5_MINI],
    )

    with (
        patch("client.AgentClient") as mock_agent_client,
        patch("voice.VoiceManager.from_env", return_value=None),
    ):
        mock_agent_client_instance = mock_agent_client.return_value
        mock_agent_client_instance.info = mock_info
        mock_agent_client_instance.agent = "test-agent"
        # Chat list endpoints are new — default the mocks to empty results
        # so the sidebar doesn't blow up in tests that don't care.
        mock_agent_client_instance.alist_chats = _empty_chat_list
        mock_agent_client_instance.acreate_chat = _echo_chat
        mock_agent_client_instance.aupdate_chat = _echo_chat
        mock_agent_client_instance.atouch_chat = _no_op
        yield mock_agent_client_instance


async def _empty_chat_list(*args, **kwargs):
    return []


async def _echo_chat(*args, **kwargs):
    from schema import ChatMetaItem

    return ChatMetaItem(
        thread_id=kwargs.get("thread_id", "test-thread"),
        user_id=kwargs.get("user_id", "test-user"),
        agent=kwargs.get("agent", "test-agent"),
        title="Mocked chat",
        preview="",
        created_at=0.0,
        updated_at=0.0,
    )


async def _no_op(*args, **kwargs):
    return None
