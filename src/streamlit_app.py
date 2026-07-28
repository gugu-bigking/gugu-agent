"""Streamlit UI for the gugu-agent service.

Major changes vs. the upstream template:

- **Branding.** "gugu-agent" header, "Made with ♥ by gugu with AI" footer.
- **Persistent chat list.** Sidebar shows recent chats from the
  gugu-chats metadata endpoint; messages stay in the LangGraph
  checkpointer. Clicking a row switches threads.
- **Queue + force interrupt.** While a stream is in flight, additional
  messages are queued; a single "Stop" button cancels the active run
  and clears the queue.
- **Typed errors with retry.** Network / 5xx / 4xx / timeout /
  stream-interrupt are surfaced as a red banner with a Retry button.
  The last user input is stashed so retry does not lose context.
- **Visible "thinking" indicator.** A local status placeholder shows
  the moment a turn is in flight and is replaced as soon as the first
  token lands.
"""

import asyncio
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from client import (
    AgentClient,
    AgentClientError,
    Attachment,
    StreamClientError,
    StreamInterruptedError,
    StreamNetworkError,
    StreamServerError,
    StreamTimeoutError,
)
from schema import ChatMessage, ChatMetaItem
from schema.task_data import TaskData, TaskDataStatus
from voice import VoiceManager

APP_TITLE = "gugu-agent"
APP_ICON = "💬"
APP_TAGLINE = "hybrid RAG · web · MCP"
APP_FOOTER = "Made with :material/favorite: by gugu with AI"
USER_ID_COOKIE = "user_id"
HISTORY_LIMIT = 30
STOP_FLAG = "cancel_requested"
ACTIVE_KEY = "active_request_id"

ATTACHMENT_TYPES = ["png", "jpg", "jpeg", "gif", "webp", "docx", "md", "txt"]
_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".txt": "text/plain",
}


def _build_attachments(uploaded_files) -> list[Attachment]:
    attachments: list[Attachment] = []
    for f in uploaded_files:
        suffix = "." + f.name.rsplit(".", 1)[-1].lower() if "." in f.name else ""
        mime = _MIME_BY_SUFFIX.get(suffix, f.type or "application/octet-stream")
        attachments.append(
            Attachment(filename=f.name, mime_type=mime, bytes=f.getvalue())
        )
    return attachments


def get_or_create_user_id() -> str:
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id
    user_id = str(uuid.uuid4())
    st.session_state[USER_ID_COOKIE] = user_id
    st.query_params[USER_ID_COOKIE] = user_id
    return user_id


def _format_relative_time(epoch: float) -> str:
    if not epoch:
        return ""
    delta = datetime.now().timestamp() - epoch
    if delta < 0:
        delta = 0
    if delta < 60:
        return "now"
    if delta < 3600:
        return f"{int(delta // 60)}m"
    if delta < 86400:
        return f"{int(delta // 3600)}h"
    if delta < 86400 * 2:
        return "Yesterday"
    if delta < 86400 * 7:
        return f"{int(delta // 86400)}d"
    return datetime.fromtimestamp(epoch).strftime("%m-%d")


def _ensure_state() -> None:
    defaults: dict[str, Any] = {
        "pending_messages": [],
        ACTIVE_KEY: None,
        STOP_FLAG: False,
        "error_state": None,
        "last_user_input": None,
        "chat_meta_cache": {},
        "chat_meta_loaded_for": None,
        "last_attachment_meta": [],
        "last_message": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _on_new_chat() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.last_user_input = None
    st.session_state.error_state = None
    st.session_state.last_attachment_meta = []
    st.session_state.pop("last_audio", None)


def _on_switch_chat(thread_id: str):
    def _cb() -> None:
        if thread_id == st.session_state.thread_id:
            return
        st.session_state.thread_id = thread_id
        st.session_state.messages = None
        st.session_state.error_state = None
        st.session_state.last_user_input = None
        st.session_state.last_attachment_meta = []
        st.session_state.pop("last_audio", None)

    return _cb


def _on_stop() -> None:
    st.session_state[STOP_FLAG] = True
    st.session_state.pending_messages = []


def _on_retry() -> None:
    last = st.session_state.last_user_input
    if not last:
        return
    st.session_state.error_state = None
    if last not in st.session_state.pending_messages:
        st.session_state.pending_messages.insert(0, last)


def _on_dismiss_error() -> None:
    st.session_state.error_state = None


def _classify_error_label(kind: str) -> str:
    return {
        "StreamNetworkError": "Can't reach the service",
        "StreamTimeoutError": "Service timed out",
        "StreamServerError": "Service error",
        "StreamClientError": "Request was rejected",
        "StreamInterruptedError": "Stream interrupted",
    }.get(kind, "Request failed")


def _classify_error_help(kind: str) -> str:
    return {
        "StreamNetworkError": "Check the connection or whether the service is running, then retry.",
        "StreamTimeoutError": "The service took too long. Retry, or pick a smaller model.",
        "StreamServerError": "The service hit an error. Retry — it may be transient.",
        "StreamClientError": "The service rejected this request. Adjust the message or settings.",
        "StreamInterruptedError": "The stream ended before the response finished. The service may have restarted.",
    }.get(kind, "Something went wrong. Retry to try again.")


async def _cancellable(gen: AsyncGenerator[Any, None], flag_key: str) -> AsyncGenerator[Any, None]:
    # Just `return` — `async for` cleans up the inner generator on exit.
    # Calling `aclose()` on `gen` from inside its own `async for` triggers
    # `RuntimeError: aclose(): asynchronous generator is already running`.
    async for event in gen:
        if st.session_state.get(flag_key):
            return
        yield event


async def _first_event_evict(gen: AsyncGenerator[Any, None], holder) -> AsyncGenerator[Any, None]:
    cleared = False
    async for event in gen:
        if not cleared:
            holder.empty()
            cleared = True
        yield event


async def main() -> None:
    # set_page_config() must be the first Streamlit call. The toolbar
    # manipulation below may schedule a rerun, so guard it with a
    # session_state flag — never call set_page_config on the rerun path.
    if (
        "toolbar_minimal_applied" not in st.session_state
        and st.get_option("client.toolbarMode") != "minimal"
    ):
        st.session_state.toolbar_minimal_applied = True
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()
        return

    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, menu_items={})

    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )

    _ensure_state()

    # Recover from a previous run that was cancelled mid-stream.
    if st.session_state.get(STOP_FLAG) and st.session_state.get(ACTIVE_KEY) is not None:
        st.session_state[ACTIVE_KEY] = None
        st.session_state[STOP_FLAG] = False
        st.rerun()

    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    if "voice_manager" not in st.session_state:
        st.session_state.voice_manager = VoiceManager.from_env()
    voice = st.session_state.voice_manager

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages: list[ChatMessage] = []
        else:
            resume_agent = st.query_params.get("agent") or agent_client.agent
            try:
                messages = (
                    await asyncio.to_thread(
                        agent_client.get_history,
                        thread_id=thread_id,
                        agent=resume_agent,
                    )
                ).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id
    elif st.session_state.get("messages") is None:
        try:
            st.session_state.messages = (
                await asyncio.to_thread(
                    agent_client.get_history,
                    thread_id=st.session_state.thread_id,
                    agent=agent_client.agent,
                )
            ).messages
        except AgentClientError:
            st.session_state.messages = []

    st.query_params["thread_id"] = st.session_state.thread_id

    # Refresh chat metadata list lazily.
    if st.session_state.get("chat_meta_loaded_for") != user_id:
        try:
            chats = await agent_client.alist_chats(user_id=user_id)
            st.session_state.chat_meta_cache = {c.thread_id: c for c in chats}
        except AgentClientError:
            st.session_state.chat_meta_cache = {}
        st.session_state.chat_meta_loaded_for = user_id

    with st.sidebar:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;"
            f"margin:6px 0 2px;'>"
            f"<span style='font-size:1.45em;line-height:1;'>{APP_ICON}</span>"
            f"<span style='font-weight:600;font-size:1.1em;'>{APP_TITLE}</span>"
            f"</div>"
            f"<div style='color:#6b7280;font-size:0.78em;margin-bottom:10px;'>"
            f"{APP_TAGLINE}</div>",
            unsafe_allow_html=True,
        )

        st.button(
            ":material/chat: New chat",
            use_container_width=True,
            key="new_chat_btn",
            on_click=_on_new_chat,
        )

        st.divider()
        st.caption("HISTORY")
        chats: list[ChatMetaItem] = list(st.session_state.chat_meta_cache.values())
        chats.sort(key=lambda c: c.updated_at, reverse=True)
        current_thread = st.session_state.thread_id
        if not chats:
            st.caption("No saved chats yet.")
        for chat in chats[:HISTORY_LIMIT]:
            is_active = chat.thread_id == current_thread
            label = chat.title[:34] or "New chat"
            time_str = _format_relative_time(chat.updated_at)
            if time_str:
                label = f"{label}  ·  {time_str}"
            if is_active:
                label = f"● {label}"
            st.button(
                label,
                key=f"chat_{chat.thread_id}",
                use_container_width=True,
                disabled=is_active,
                on_click=_on_switch_chat(chat.thread_id),
            )

        st.divider()

        with st.popover(":material/settings: Settings", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=agent_list,
                index=agent_idx,
                key="agent",
                bind="query-params",
            )
            use_streaming = st.toggle("Stream results", value=True)
            enable_audio = st.toggle(
                "Enable audio generation",
                value=True,
                disabled=not voice or not voice.tts,
                on_change=lambda: (
                    st.session_state.pop("last_audio", None)
                    if not st.session_state.get("enable_audio", True)
                    else None
                ),
                key="enable_audio",
            )
            st.text_input("User ID (read-only)", value=user_id, disabled=True)

        @st.dialog("Architecture")
        def architecture_dialog() -> None:
            st.image(
                "https://github.com/gugu-bigking/gugu-agent/blob/main/media/agent_architecture.png?raw=true"
            )
            "[View full size on Github](https://github.com/gugu-bigking/gugu-agent)"
            st.caption(
                "App hosted on [Streamlit Cloud](https://share.streamlit.io/) with FastAPI service running in [Azure](https://learn.microsoft.com/en-us/azure/app-service/)"
            )

        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
            )

        @st.dialog("Share/resume chat")
        def share_chat_dialog() -> None:
            if not st.context.url:
                st.error("Could not determine the app URL to build a shareable link.")
                return
            query = urllib.parse.urlencode(
                {
                    "thread_id": st.session_state.thread_id,
                    "agent": agent_client.agent,
                    USER_ID_COOKIE: user_id,
                }
            )
            st.markdown(f"**Chat URL:**\n```text\n{st.context.url}?{query}\n```")
            st.info("Copy the above URL to share or revisit this chat")

        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()

        "[View the source code](https://github.com/gugu-bigking/gugu-agent)"
        st.caption(APP_FOOTER)

    messages: list[ChatMessage] = st.session_state.messages or []

    if len(messages) == 0:
        match agent_client.agent:
            case "chatbot":
                WELCOME = "Hello! I'm a simple chatbot. Ask me anything!"
            case "interrupt-agent":
                WELCOME = "Hello! I'm an interrupt agent. Tell me your birthday and I will predict your personality!"
            case "research-assistant":
                WELCOME = "Hello! I'm an AI-powered research assistant with web search and a calculator. Ask me anything!"
            case "rag-assistant":
                WELCOME = """Hello! I'm an AI-powered Company Policy & HR assistant with access to AcmeTech's Employee Handbook.
                I can help you find information about benefits, remote work, time-off policies, company values, and more. Ask me anything!"""
            case "gugu-agent":
                WELCOME = (
                    "Hey, I'm gugu — your hybrid RAG assistant. I search the handbook with "
                    "BM25 + dense retrieval and rerank the results, and I'll fall back to the "
                    "live web for anything not internal. Ask me anything!"
                )
            case _:
                WELCOME = "Hello! I'm an AI agent. Ask me anything!"

        with st.chat_message("ai"):
            st.write(WELCOME)

    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    if (
        voice
        and enable_audio
        and "last_audio" in st.session_state
        and st.session_state.last_message
        and len(messages) > 0
        and messages[-1].type == "ai"
    ):
        with st.session_state.last_message:
            audio_data = st.session_state.last_audio
            st.audio(audio_data["data"], format=audio_data["format"])

    # Error banner with retry.
    err = st.session_state.error_state
    if err:
        raw = err.get("error", "")
        kind = err.get("kind", "AgentClientError")
        st.error(f"Error generating response: {raw}")
        st.markdown(f"**{_classify_error_label(kind)}.** {_classify_error_help(kind)}")
        cols = st.columns([1, 1, 4])
        cols[0].button("Retry", key="retry_btn", type="primary", on_click=_on_retry)
        cols[1].button("Dismiss", key="dismiss_btn", on_click=_on_dismiss_error)

    # Active run banner.
    if st.session_state[ACTIVE_KEY] is not None:
        queued = len(st.session_state.pending_messages)
        msg = "Gugu is thinking…"
        if queued:
            msg = f"Gugu is thinking… ({queued} queued)"
        st.info(msg, icon="⏳")
        st.button(
            ":material/stop: Stop",
            key="stop_btn",
            type="secondary",
            on_click=_on_stop,
        )

    # Chat input.
    chat_input = voice.get_chat_input(file_type=ATTACHMENT_TYPES) if voice else None
    if chat_input is None:
        chat_input = st.chat_input(accept_file="multiple", file_type=ATTACHMENT_TYPES)
    user_text = chat_input["text"] if chat_input else ""
    uploaded_files = chat_input["files"] if chat_input else []

    if chat_input and uploaded_files and not user_text.strip():
        st.toast("请输入文字后再发送附件", icon="⚠️")
        chat_input = None

    if chat_input:
        attachments = _build_attachments(uploaded_files) if uploaded_files else None
        display_text = user_text or "(attached file)"
        new_input = {
            "text": user_text,
            "display_text": display_text,
            "attachments": attachments,
            "model": model,
        }
        human_msg = ChatMessage(type="human", content=display_text)
        st.session_state.messages.append(human_msg)
        st.session_state.last_user_input = new_input
        st.session_state.last_attachment_meta = [a.name for a in (uploaded_files or [])]
        st.session_state.pending_messages.append(new_input)
        st.rerun()

    # Process the queue when nothing is active.
    if st.session_state[ACTIVE_KEY] is None and st.session_state.pending_messages:
        next_input = st.session_state.pending_messages.pop(0)
        st.session_state[ACTIVE_KEY] = id(next_input)
        st.session_state[STOP_FLAG] = False
        st.session_state.last_user_input = next_input
        thread_id = st.session_state.thread_id

        # Register the chat in metadata on the first message of a new thread.
        cached = st.session_state.chat_meta_cache.get(thread_id)
        if cached is None:
            try:
                item = await agent_client.acreate_chat(
                    user_id=user_id,
                    agent=agent_client.agent,
                    first_message=next_input["text"],
                )
                st.session_state.thread_id = item.thread_id
                st.query_params["thread_id"] = item.thread_id
                st.session_state.chat_meta_cache[item.thread_id] = item
            except AgentClientError:
                # Degrade gracefully — the run still proceeds.
                pass

        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=next_input["text"],
                    model=next_input["model"],
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                    attachments=next_input["attachments"],
                )
                thinking_holder = st.empty()
                thinking_holder.status("Thinking…", state="running", expanded=False)
                wrapped = _first_event_evict(
                    _cancellable(stream, STOP_FLAG), thinking_holder
                )
                await draw_messages(wrapped, is_new=True)
                msgs = st.session_state.messages
                last_ai = msgs[-1] if msgs else None
                if last_ai and last_ai.type == "ai" and last_ai.content:
                    try:
                        await agent_client.atouch_chat(
                            thread_id=st.session_state.thread_id,
                            preview=last_ai.content[:120],
                        )
                    except Exception:
                        pass
                cached = st.session_state.chat_meta_cache.get(st.session_state.thread_id)
                if cached is not None:
                    cached.updated_at = datetime.now().timestamp()
                if voice and enable_audio and last_ai and last_ai.type == "ai" and last_ai.content:
                    voice.render_message(
                        last_ai.content,
                        container=st.session_state.last_message,
                        audio_only=True,
                    )
            else:
                response = await agent_client.ainvoke(
                    message=next_input["text"],
                    model=next_input["model"],
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                    attachments=next_input["attachments"],
                )
                st.session_state.messages.append(response)
                messages.append(response)
                with st.chat_message("ai"):
                    if voice and enable_audio:
                        voice.render_message(response.content)
                    else:
                        st.write(response.content)
                try:
                    await agent_client.atouch_chat(
                        thread_id=st.session_state.thread_id,
                        preview=response.content[:120],
                    )
                except Exception:
                    pass
        except AgentClientError as e:
            st.session_state.error_state = {"error": str(e), "kind": type(e).__name__}
        finally:
            st.session_state[ACTIVE_KEY] = None
            st.session_state[STOP_FLAG] = False
        st.rerun()

    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    last_message_type = None
    st.session_state.last_message = None
    streaming_content = ""
    streaming_placeholder = None

    while msg := await anext(messages_agen, None):
        if isinstance(msg, str):
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()
            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            case "ai":
                if is_new:
                    st.session_state.messages.append(msg)
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            if "transfer_to" in tool_call["name"]:
                                label = f"""💼 Sub Agent: {tool_call["name"]}"""
                            else:
                                label = f"""🛠️ Tool Call: {tool_call["name"]}"""
                            call_results[tool_call["id"]] = st.status(
                                label,
                                state="running" if is_new else "complete",
                            )

                        for tool_call in msg.tool_calls:
                            if "transfer_to" in tool_call["name"]:
                                status = call_results[tool_call["id"]]
                                with status:
                                    status.update(expanded=True)
                                    await handle_sub_agent_msgs(messages_agen, status, is_new)
                                # Force a new chat_message for the next ai message
                                # so it renders as a sibling, not a child.
                                last_message_type = None
                                break
                            status = call_results[tool_call["id"]]
                            with status:
                                status.write("Input:")
                                status.write(tool_call["args"])
                                tool_result = await anext(messages_agen, None)
                                if tool_result is None:
                                    st.error(f"Stream ended before tool {tool_call['name']} returned a result")
                                    return
                                if tool_result.type != "tool":
                                    st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                    st.write(tool_result)
                                    st.stop()
                                if is_new:
                                    st.session_state.messages.append(tool_result)
                                if tool_result.tool_call_id:
                                    status = call_results[tool_result.tool_call_id]
                                status.write("Output:")
                                status.write(tool_result.content)
                                status.update(state="complete")

            case "custom":
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Unexpected CustomData message received from agent")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)
    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        normalized_score = (feedback + 1) / 5.0
        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


async def handle_sub_agent_msgs(messages_agen, status, is_new):
    nested_popovers: dict[str, Any] = {}
    first_msg = await anext(messages_agen, None)
    if first_msg is None:
        return
    if is_new:
        st.session_state.messages.append(first_msg)
    while True:
        sub_msg = await anext(messages_agen, None)
        if sub_msg is None:
            return
        if is_new:
            st.session_state.messages.append(sub_msg)
        if sub_msg.type == "tool" and sub_msg.tool_call_id in nested_popovers:
            popover = nested_popovers[sub_msg.tool_call_id]
            popover.write("**Output:**")
            popover.write(sub_msg.content)
            continue
        if (
            hasattr(sub_msg, "tool_calls")
            and sub_msg.tool_calls
            and "transfer_back_to" in sub_msg.tool_calls[0]["name"]
        ):
            status.write(sub_msg.content)
            status.update(state="complete")
            # Consume the matching tool result so it doesn't surface in the
            # main draw_messages loop as an "Unexpected ChatMessage type: tool".
            tail = await anext(messages_agen, None)
            if tail is not None and is_new:
                st.session_state.messages.append(tail)
            return
        if sub_msg.type == "ai":
            if sub_msg.content:
                status.write(sub_msg.content)
            for tool_call in sub_msg.tool_calls or []:
                if "transfer_to" in tool_call["name"]:
                    nested_status = status.status(
                        f"💼 Sub Agent: {tool_call['name']}",
                        state="running" if is_new else "complete",
                    )
                    nested_status.update(expanded=True)
                    await handle_sub_agent_msgs(messages_agen, nested_status, is_new)
                    return
                popover = status.popover(tool_call["name"], icon="🛠️")
                popover.write(f"**Tool:** {tool_call['name']}")
                popover.write("**Input:**")
                popover.json(tool_call["args"])
                nested_popovers[tool_call["id"]] = popover
        elif sub_msg.type == "tool":
            status.write(sub_msg.content)


if __name__ == "__main__":
    asyncio.run(main())
