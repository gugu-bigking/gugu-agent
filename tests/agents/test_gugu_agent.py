"""Tests for the gugu-agent registration and graph wiring."""

import pytest

from agents import gugu_agent as gugu_agent_module
from agents.agents import DEFAULT_AGENT, agents, get_agent, load_agent
from agents.gugu_agent import GuguAgent
from agents.lazy_agent import LazyLoadingAgent
from agents.rag_tools import hybrid_search
from agents.tools import web_search


def test_gugu_agent_is_default():
    assert DEFAULT_AGENT == "gugu-agent"


def test_gugu_agent_registered():
    assert "gugu-agent" in agents
    entry = agents["gugu-agent"]
    assert "hybrid" in entry.description.lower() or "rerank" in entry.description.lower()
    assert isinstance(entry.graph_like, LazyLoadingAgent)


def test_gugu_agent_instance_is_lazy():
    assert isinstance(gugu_agent_module.gugu_agent, GuguAgent)
    assert gugu_agent_module.gugu_agent._loaded is False


@pytest.mark.asyncio
async def test_get_gugu_agent_before_load_raises():
    assert gugu_agent_module.gugu_agent._loaded is False
    with pytest.raises(RuntimeError, match="not loaded"):
        get_agent("gugu-agent")


@pytest.mark.asyncio
async def test_load_gugu_agent_without_mcp_creds(monkeypatch):
    monkeypatch.setattr("agents.gugu_agent.settings", type("S", (), {
        "GITHUB_PAT": None,
        "NOTION_API_KEY": None,
        "MCP_NOTION_SERVER_URL": "",
        "MCP_GITHUB_SERVER_URL": "",
        "DEFAULT_MODEL": "fake-model",
    })())

    gugu_agent_module._MCP_TOOLS = []
    agent_obj = GuguAgent()
    await agent_obj.load()

    assert agent_obj._loaded
    tool_names = {t.name for t in agent_obj._all_tools}
    assert "Hybrid_Search" in tool_names
    assert "Web_Search" in tool_names
    assert hybrid_search in agent_obj._all_tools
    assert web_search in agent_obj._all_tools
    assert agent_obj._graph is not None
    assert agent_obj._graph.name == "gugu-agent"


@pytest.mark.asyncio
async def test_load_agent_calls_gugu_load(monkeypatch):
    from unittest.mock import AsyncMock

    fake_loader = AsyncMock()
    monkeypatch.setattr(gugu_agent_module.gugu_agent, "load", fake_loader)
    await load_agent("gugu-agent")
    fake_loader.assert_awaited_once()