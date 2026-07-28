import base64
import json
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any, NotRequired

import httpx
from typing_extensions import TypedDict

from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    ChatMetaCreate,
    ChatMetaItem,
    Feedback,
    ServiceMetadata,
    StreamInput,
    UserInput,
)


class Attachment(TypedDict):
    """A single file or image attachment sent alongside a user message.

    Either provide `bytes` (raw file bytes — they will be base64-encoded here)
    or `data_b64` (already-encoded). Exactly one of the two must be set.
    `mime_type` and `filename` are required.
    """

    filename: str
    mime_type: str
    bytes: NotRequired[bytes]
    data_b64: NotRequired[str]


class AgentClientError(Exception):
    """Base class for all client-side errors."""


class StreamNetworkError(AgentClientError):
    """The service was unreachable (DNS, refused, disconnect, protocol error)."""


class StreamTimeoutError(AgentClientError):
    """Connect or read timed out before the response completed."""


class StreamServerError(AgentClientError):
    """The service returned a 5xx response."""


class StreamClientError(AgentClientError):
    """The service returned a 4xx response other than auth/not-found/validation."""


class StreamInterruptedError(AgentClientError):
    """The stream ended without the expected [DONE] marker."""


def _classify_http_error(e: httpx.HTTPError) -> AgentClientError:
    if isinstance(e, httpx.TimeoutException):
        return StreamTimeoutError(f"Request timed out: {e}")
    if isinstance(e, httpx.ConnectError):
        return StreamNetworkError(f"Couldn't reach the service: {e}")
    if isinstance(e, (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError)):
        return StreamInterruptedError(f"Connection interrupted: {e}")
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        body = ""
        try:
            body = e.response.text
        except Exception:  # noqa: BLE001
            body = ""
        if status >= 500:
            return StreamServerError(f"Service error {status}: {body[:200]}")
        return StreamClientError(f"Request rejected ({status}): {body[:200]}")
    return AgentClientError(f"HTTP error: {e}")


_IMAGE_MIME_PREFIXES = ("image/",)
_DOC_MIME_BY_SUFFIX = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".txt": "text/plain",
}


def _attachment_to_block(att: Attachment) -> dict[str, Any]:
    """Turn an Attachment into a multimodal content block.

    - Image MIME -> image_url with an inline data URL.
    - .docx / .md / .txt -> file block with base64 bytes.
    - Other -> file block (server will reject unsupported types).
    """
    if "bytes" in att and "data_b64" in att:
        raise AgentClientError(f"Attachment {att['filename']}: provide only one of bytes / data_b64")
    if "bytes" not in att and "data_b64" not in att:
        raise AgentClientError(f"Attachment {att['filename']}: must provide bytes or data_b64")

    data_b64 = att.get("data_b64") or base64.b64encode(att["bytes"]).decode("ascii")
    mime = att["mime_type"]
    filename = att["filename"]

    if mime.startswith(_IMAGE_MIME_PREFIXES):
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{data_b64}"},
        }
    return {
        "type": "file",
        "file": {"filename": filename, "mime_type": mime, "data_b64": data_b64},
    }


def _build_message(
    message: str,
    attachments: list[Attachment] | None,
) -> str | list[dict[str, Any]]:
    """Combine a user message with optional attachments into a UserInput payload."""
    blocks: list[dict[str, Any]] = []
    if message:
        blocks.append({"type": "text", "text": message})
    for att in attachments or []:
        blocks.append(_attachment_to_block(att))
    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0]["type"] == "text":
        return blocks[0]["text"]
    return blocks


class AgentClient:
    """Client for interacting with the agent service."""

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | None = None,
        get_info: bool = True,
    ) -> None:
        """
        Initialize the client.

        Args:
            base_url (str): The base URL of the agent service.
            agent (str): The name of the default agent to use.
            timeout (float, optional): The timeout for requests.
            get_info (bool, optional): Whether to fetch agent information on init.
                Default: True
        """
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout
        self.info: ServiceMetadata | None = None
        self.agent: str | None = None
        if get_info:
            self.retrieve_info()
        if agent:
            self.update_agent(agent)

    @property
    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    def retrieve_info(self) -> None:
        try:
            response = httpx.get(
                f"{self.base_url}/info",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error getting service info: {e}")

        self.info = ServiceMetadata.model_validate(response.json())
        if not self.agent or self.agent not in [a.key for a in self.info.agents]:
            self.agent = self.info.default_agent

    def update_agent(self, agent: str, verify: bool = True) -> None:
        if verify:
            if not self.info:
                self.retrieve_info()
            agent_keys = [a.key for a in self.info.agents]  # type: ignore[union-attr]
            if agent not in agent_keys:
                raise AgentClientError(
                    f"Agent {agent} not found in available agents: {', '.join(agent_keys)}"
                )
        self.agent = agent

    async def ainvoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        attachments: list[Attachment] | None = None,
    ) -> ChatMessage:
        """
        Invoke the agent asynchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            attachments (list[Attachment], optional): Files / images to send alongside the message

        Returns:
            AnyMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=_build_message(message, attachments))
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/{self.agent}/invoke",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def invoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        attachments: list[Attachment] | None = None,
    ) -> ChatMessage:
        """
        Invoke the agent synchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            attachments (list[Attachment], optional): Files / images to send alongside the message

        Returns:
            ChatMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=_build_message(message, attachments))
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        try:
            response = httpx.post(
                f"{self.base_url}/{self.agent}/invoke",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def _parse_stream_line(self, line: str) -> ChatMessage | str | None:
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise Exception(f"Error JSON parsing message from server: {e}")
            match parsed["type"]:
                case "message":
                    # Convert the JSON formatted message to an AnyMessage
                    try:
                        return ChatMessage.model_validate(parsed["content"])
                    except Exception as e:
                        raise Exception(f"Server returned invalid message: {e}")
                case "token":
                    # Yield the str token directly
                    return parsed["content"]
                case "error":
                    error_msg = "Error: " + parsed["content"]
                    return ChatMessage(type="ai", content=error_msg)
        return None

    def stream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
        attachments: list[Attachment] | None = None,
    ) -> Generator[ChatMessage | str, None, None]:
        """
        Stream the agent's response synchronously.

        Each intermediate message of the agent process is yielded as a ChatMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming models as they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            Generator[ChatMessage | str, None, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=_build_message(message, attachments), stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if user_id:
            request.user_id = user_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/{self.agent}/stream",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        parsed = self._parse_stream_line(line)
                        if parsed is None:
                            break
                        yield parsed
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

    async def astream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
        attachments: list[Attachment] | None = None,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """
        Stream the agent's response asynchronously.

        Each intermediate message of the agent process is yielded as an AnyMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming modelsas they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            AsyncGenerator[ChatMessage | str, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=_build_message(message, attachments), stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/{self.agent}/stream",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPError as e:
                        raise _classify_http_error(e) from None
                    saw_done = False
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        parsed = self._parse_stream_line(line)
                        if parsed is None:
                            saw_done = True
                            break
                        if parsed == "":
                            continue
                        yield parsed
                    if not saw_done:
                        raise StreamInterruptedError(
                            "Stream ended before completion. The service may have restarted."
                        )
            except httpx.HTTPError as e:
                raise _classify_http_error(e) from None

    async def acreate_feedback(
        self, run_id: str, key: str, score: float, kwargs: dict[str, Any] = {}
    ) -> None:
        """
        Create a feedback record for a run.

        This is a simple wrapper for the LangSmith create_feedback API, so the
        credentials can be stored and managed in the service rather than the client.
        See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
        """
        request = Feedback(run_id=run_id, key=key, score=score, kwargs=kwargs)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/feedback",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response.json()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    def get_history(self, thread_id: str, agent: str | None = None) -> ChatHistory:
        """
        Get chat history.

        Args:
            thread_id (str, optional): Thread ID for identifying a conversation
            agent (str, optional): The agent whose graph should interpret the thread.
        """
        agent = agent or self.agent
        request = ChatHistoryInput(thread_id=thread_id)
        url = f"{self.base_url}/{agent}/history" if agent else f"{self.base_url}/history"
        try:
            response = httpx.post(
                url,
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise _classify_http_error(e) from None

        return ChatHistory.model_validate(response.json())

    async def alist_chats(
        self, user_id: str, agent: str | None = None
    ) -> list[ChatMetaItem]:
        """List chat metadata for the sidebar, newest first.

        The list is metadata-only; messages stay in the LangGraph
        checkpointer and are fetched on demand via get_history.
        """
        params: dict[str, str] = {"user_id": user_id}
        if agent:
            params["agent"] = agent
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/gugu/chats",
                    params=params,
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise _classify_http_error(e) from None
        return [ChatMetaItem.model_validate(item) for item in response.json()]

    async def acreate_chat(
        self,
        user_id: str,
        agent: str,
        first_message: str | None = None,
        title: str | None = None,
    ) -> ChatMetaItem:
        """Register a brand-new chat in the metadata store.

        The server mints the thread_id and returns the persisted row. The
        client then writes the returned thread_id back into the URL and
        session state.
        """
        payload = ChatMetaCreate(
            user_id=user_id,
            agent=agent,
            title=title,
            first_message=first_message,
        )
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/gugu/chats",
                    json=payload.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise _classify_http_error(e) from None
        return ChatMetaItem.model_validate(response.json())

    async def aupdate_chat(
        self,
        thread_id: str,
        title: str | None = None,
        preview: str | None = None,
    ) -> ChatMetaItem:
        """Update a chat's title and/or preview. Server returns 404 if unknown."""
        body: dict[str, Any] = {}
        if title is not None:
            body["title"] = title
        if preview is not None:
            body["preview"] = preview
        async with httpx.AsyncClient() as client:
            try:
                response = await client.patch(
                    f"{self.base_url}/gugu/chats/{thread_id}",
                    json=body,
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise _classify_http_error(e) from None
        return ChatMetaItem.model_validate(response.json())

    async def atouch_chat(
        self, thread_id: str, preview: str | None = None
    ) -> None:
        """Bump a chat's updated_at (and optionally preview) after a turn.

        Best-effort: 404 and network errors are swallowed so the user
        flow keeps working even when metadata is out of sync with the
        LangGraph checkpointer.
        """
        params: dict[str, str] = {}
        if preview is not None:
            params["preview"] = preview
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/gugu/chats/{thread_id}/touch",
                    params=params,
                    headers=self._headers,
                    timeout=self.timeout,
                )
                if response.status_code == 404:
                    return
                response.raise_for_status()
            except httpx.HTTPError:
                return
