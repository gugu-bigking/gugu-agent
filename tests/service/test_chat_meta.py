"""Tests for the SQLite chat metadata store and /gugu/chats endpoints."""

import asyncio
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.types import StateSnapshot

from service import app
from service.chat_meta import ChatMetaStore, _auto_preview, _auto_title


@asynccontextmanager
async def _open_store(tmp_path):
    store = ChatMetaStore(str(tmp_path / "chat_meta.db"))
    await store.init()
    yield store


@pytest.mark.asyncio
async def test_store_init_creates_table(tmp_path):
    async with _open_store(tmp_path) as store:
        items = await store.list_for_user("u1")
        assert items == []


@pytest.mark.asyncio
async def test_create_and_get_round_trip(tmp_path):
    async with _open_store(tmp_path) as store:
        item = await store.create(
            thread_id="t1",
            user_id="u1",
            agent="gugu-agent",
            first_message="What is BM25?",
        )
        assert item.title.startswith("What is BM25")
        assert item.preview.startswith("What is BM25")
        loaded = await store.get("t1")
        assert loaded is not None
        assert loaded.title == item.title


@pytest.mark.asyncio
async def test_list_orders_by_updated_at_desc(tmp_path):
    async with _open_store(tmp_path) as store:
        await store.create(
            thread_id="old",
            user_id="u1",
            agent="gugu-agent",
            first_message="older",
            title="older",
        )
        await asyncio.sleep(0.01)
        await store.create(
            thread_id="new",
            user_id="u1",
            agent="gugu-agent",
            first_message="newer",
            title="newer",
        )
        items = await store.list_for_user("u1")
        assert [i.thread_id for i in items] == ["new", "old"]


@pytest.mark.asyncio
async def test_update_changes_title_and_preview(tmp_path):
    async with _open_store(tmp_path) as store:
        await store.create(
            thread_id="t2",
            user_id="u1",
            agent="gugu-agent",
            first_message="hi",
            title="hi",
        )
        await asyncio.sleep(0.01)
        updated = await store.update("t2", title="renamed", preview="new preview")
        assert updated is not None
        assert updated.title == "renamed"
        assert updated.preview == "new preview"


@pytest.mark.asyncio
async def test_update_unknown_returns_none(tmp_path):
    async with _open_store(tmp_path) as store:
        result = await store.update("missing", title="x")
        assert result is None


@pytest.mark.asyncio
async def test_delete_removes_row(tmp_path):
    async with _open_store(tmp_path) as store:
        await store.create(
            thread_id="t3",
            user_id="u1",
            agent="gugu-agent",
            first_message="bye",
            title="bye",
        )
        await store.delete("t3")
        assert await store.get("t3") is None


def test_auto_title_caps_long_input():
    long = "x" * 200
    title = _auto_title(long)
    assert title.endswith("…")
    assert len(title) < len(long)


def test_auto_title_collapses_whitespace_and_handles_empty():
    assert _auto_title("   \n\n  hello  world  \n").startswith("hello world")
    assert _auto_title("") == "New chat"


def test_auto_title_counts_cjk_as_two_columns():
    title = _auto_title("你好世界" * 20)
    assert title.endswith("…")
    # 60 width budget with CJK counted as 2 → at most 30 CJK chars before "…".
    assert len(title) <= 31


def test_auto_preview_caps_at_120_chars():
    msg = "a" * 500
    assert len(_auto_preview(msg)) == 120


@pytest.mark.asyncio
async def test_create_uses_explicit_title_when_provided(tmp_path):
    async with _open_store(tmp_path) as store:
        item = await store.create(
            thread_id="t4",
            user_id="u1",
            agent="gugu-agent",
            first_message="ignored",
            title="Custom Title",
        )
        assert item.title == "Custom Title"


# ---- Endpoint tests ----


@pytest.fixture
def chat_meta_endpoint_client(tmp_path):
    """Spin up TestClient around the real FastAPI app with a temp chat_meta store."""
    store = ChatMetaStore(str(tmp_path / "chat_meta.db"))

    agent_mock = AsyncMock()
    agent_mock.aget_state = AsyncMock(
        return_value=StateSnapshot(
            values={"messages": [AIMessage(content="ok")]},
            next=(),
            config={},
            metadata=None,
            created_at=None,
            parent_config=None,
            tasks=(),
            interrupts=(),
        )
    )

    with (
        TestClient(app) as client,
        patch("service.service.get_agent", return_value=agent_mock),
    ):
        # Lifespan would normally init the store; do it here.
        async def _init() -> None:
            await store.init()
            app.state.chat_meta_store = store

        import asyncio as _asyncio

        _asyncio.run(_init())
        try:
            yield client
        finally:
            app.state.__dict__.pop("chat_meta_store", None)


def test_list_endpoint_returns_user_chats(chat_meta_endpoint_client):
    client = chat_meta_endpoint_client
    # Seed two chats by POSTing create.
    r1 = client.post(
        "/gugu/chats",
        json={
            "user_id": "u1",
            "agent": "gugu-agent",
            "first_message": "What is RAG?",
        },
    )
    assert r1.status_code == 200
    r2 = client.post(
        "/gugu/chats",
        json={"user_id": "u1", "agent": "gugu-agent", "first_message": "Summarize X."},
    )
    assert r2.status_code == 200

    resp = client.get("/gugu/chats", params={"user_id": "u1"})
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2
    assert all(item["user_id"] == "u1" for item in items)


def test_update_endpoint_404_for_unknown_chat(chat_meta_endpoint_client):
    client = chat_meta_endpoint_client
    resp = client.patch(
        "/gugu/chats/does-not-exist",
        json={"title": "new"},
    )
    assert resp.status_code == 404


def test_update_endpoint_renames_chat(chat_meta_endpoint_client):
    client = chat_meta_endpoint_client
    created = client.post(
        "/gugu/chats",
        json={"user_id": "u1", "agent": "gugu-agent", "first_message": "hi"},
    ).json()
    resp = client.patch(
        f"/gugu/chats/{created['thread_id']}",
        json={"title": "renamed"},
    )
    assert resp.status_code == 200
    assert resp.json()["title"] == "renamed"


def test_touch_bumps_updated_at(chat_meta_endpoint_client):
    client = chat_meta_endpoint_client
    created = client.post(
        "/gugu/chats",
        json={"user_id": "u1", "agent": "gugu-agent", "first_message": "hi"},
    ).json()
    first_ts = created["updated_at"]
    time.sleep(0.05)
    resp = client.post(
        f"/gugu/chats/{created['thread_id']}/touch",
        params={"preview": "new"},
    )
    assert resp.status_code == 200
    assert resp.json()["updated_at"] >= first_ts
    assert resp.json()["preview"] == "new"
