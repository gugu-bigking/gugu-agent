"""SQLite-backed store for chat *metadata* (title, preview, ordering).

Messages themselves live in the LangGraph checkpointer. This store only
knows enough to render a sidebar of recent chats and to remember a
user-given title. The choice to keep the two stores separate is
intentional: a bad migration on metadata must not touch conversation
state, and LangGraph's checkpointer schema is versioned and managed
upstream.
"""

import time
from dataclasses import dataclass
from pathlib import Path

from aiosqlite import connect

from schema import ChatMetaCreate, ChatMetaItem, ChatMetaUpdate


@dataclass
class _Row:
    thread_id: str
    user_id: str
    agent: str
    title: str
    preview: str
    created_at: float
    updated_at: float


def _to_item(row: _Row) -> ChatMetaItem:
    return ChatMetaItem(
        thread_id=row.thread_id,
        user_id=row.user_id,
        agent=row.agent,
        title=row.title,
        preview=row.preview,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_meta (
    thread_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    title TEXT NOT NULL,
    preview TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS chat_meta_user_idx
    ON chat_meta (user_id, agent, updated_at DESC);
"""


_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3000, 0x303F), (0xFF00, 0xFFEF))
_TITLE_WIDTH = 60  # CJK chars count as 2 widths


def _char_width(cp: int) -> int:
    for lo, hi in _CJK_RANGES:
        if lo <= cp <= hi:
            return 2
    return 1


def _auto_title(message: str) -> str:
    """Pick a one-line title from the user's first message.

    Collapses whitespace, caps at ~60 display columns (CJK counted as 2).
    Falls back to "New chat" for empty input.
    """
    if not message:
        return "New chat"
    first_line = next((line.strip() for line in message.splitlines() if line.strip()), "")
    if not first_line:
        return "New chat"
    normalized = " ".join(first_line.split())
    width = 0
    out: list[str] = []
    for ch in normalized:
        width += _char_width(ord(ch))
        if width > _TITLE_WIDTH:
            out.append("…")
            break
        out.append(ch)
    return "".join(out) or "New chat"


def _auto_preview(message: str) -> str:
    if not message:
        return ""
    first_line = next((line.strip() for line in message.splitlines() if line.strip()), "")
    return " ".join(first_line.split())[:120]


class ChatMetaStore:
    """Async SQLite wrapper for chat metadata."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def init(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        async with connect(self._db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()

    async def list_for_user(
        self, user_id: str, agent: str | None = None
    ) -> list[ChatMetaItem]:
        async with connect(self._db_path) as db:
            if agent:
                cur = await db.execute(
                    "SELECT * FROM chat_meta WHERE user_id = ? AND agent = ? "
                    "ORDER BY updated_at DESC",
                    (user_id, agent),
                )
            else:
                cur = await db.execute(
                    "SELECT * FROM chat_meta WHERE user_id = ? ORDER BY updated_at DESC",
                    (user_id,),
                )
            rows = await cur.fetchall()
        return [_to_item(_Row(*row)) for row in rows]

    async def get(self, thread_id: str) -> ChatMetaItem | None:
        async with connect(self._db_path) as db:
            cur = await db.execute(
                "SELECT * FROM chat_meta WHERE thread_id = ?", (thread_id,)
            )
            row = await cur.fetchone()
        if not row:
            return None
        return _to_item(_Row(*row))

    async def create(
        self,
        thread_id: str,
        user_id: str,
        agent: str,
        title: str | None = None,
        preview: str | None = None,
        first_message: str | None = None,
    ) -> ChatMetaItem:
        now = time.time()
        final_title = title or _auto_title(first_message or "")
        final_preview = preview if preview is not None else _auto_preview(first_message or "")
        async with connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO chat_meta
                (thread_id, user_id, agent, title, preview, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    title = excluded.title,
                    preview = excluded.preview,
                    updated_at = excluded.updated_at
                """,
                (
                    thread_id,
                    user_id,
                    agent,
                    final_title,
                    final_preview,
                    now,
                    now,
                ),
            )
            await db.commit()
        return ChatMetaItem(
            thread_id=thread_id,
            user_id=user_id,
            agent=agent,
            title=final_title,
            preview=final_preview,
            created_at=now,
            updated_at=now,
        )

    async def update(
        self,
        thread_id: str,
        title: str | None = None,
        preview: str | None = None,
    ) -> ChatMetaItem | None:
        fields: list[str] = []
        values: list = []
        if title is not None:
            fields.append("title = ?")
            values.append(title)
        if preview is not None:
            fields.append("preview = ?")
            values.append(preview)
        fields.append("updated_at = ?")
        values.append(time.time())
        values.append(thread_id)
        async with connect(self._db_path) as db:
            await db.execute(
                f"UPDATE chat_meta SET {', '.join(fields)} WHERE thread_id = ?",
                values,
            )
            await db.commit()
        return await self.get(thread_id)

    async def touch(self, thread_id: str, preview: str | None = None) -> ChatMetaItem | None:
        """Bump updated_at (and optionally preview) — called after each turn."""
        return await self.update(thread_id, preview=preview)

    async def delete(self, thread_id: str) -> None:
        async with connect(self._db_path) as db:
            await db.execute(
                "DELETE FROM chat_meta WHERE thread_id = ?", (thread_id,)
            )
            await db.commit()


def build_store_from_create(payload: ChatMetaCreate, thread_id: str) -> dict[str, str | float | None]:
    """Helper to derive fields for a new chat from the create payload."""
    return {
        "thread_id": thread_id,
        "user_id": payload.user_id,
        "agent": payload.agent,
        "title": payload.title or _auto_title(payload.first_message or ""),
        "preview": _auto_preview(payload.first_message or ""),
    }


def apply_update_payload(
    existing: ChatMetaItem, payload: ChatMetaUpdate
) -> dict[str, str | float | None]:
    return {
        "thread_id": existing.thread_id,
        "user_id": existing.user_id,
        "agent": existing.agent,
        "title": payload.title if payload.title is not None else existing.title,
        "preview": payload.preview if payload.preview is not None else existing.preview,
        "created_at": existing.created_at,
        "updated_at": time.time(),
    }
