"""Gugu Agent — hybrid RAG + web + MCP tools.

Lazy-loads MCP tools (GitHub, Notion) at service startup so they can be bound
to the model up-front. Local tools (hybrid_search, web_search) are always
present; MCP tools are appended when their env credentials are configured.
"""

import asyncio
import base64
import logging
from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection, StreamableHttpConnection
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.lazy_agent import LazyLoadingAgent
from agents.rag_tools import hybrid_search
from agents.safeguard import Safeguard, SafeguardOutput, SafetyAssessment
from agents.tools import (
    understand_document,
    understand_document_func,
    understand_image,
    understand_image_func,
    web_search,
)
from core import get_model, settings

logger = logging.getLogger(__name__)


class AgentState(MessagesState, total=False):
    safety: SafeguardOutput
    remaining_steps: RemainingSteps


_LOCAL_TOOLS: list[BaseTool] = [hybrid_search, web_search, understand_image, understand_document]

current_date = datetime.now().strftime("%B %d, %Y")

_BASE_INSTRUCTIONS = f"""
    You are gugu, a precise and friendly AI assistant built by an AI application engineer.
    Today's date is {current_date}.

    You have tools. Pick the right one based on the question:
    - Hybrid_Search: internal company knowledge base (handbook, policies, products).
      Returns top-5 passages after BM25+dense retrieval and reranking.
    - Web_Search: live public web for news, current events, anything not internal.
    - Understand_Image: server-side vision model for images already attached to the
      user message (vision-capable models may also see the image directly — use
      this tool only if you need a deeper description or the model is text-only).
    - Understand_Document: extract text from `.docx`, `.md`, or `.txt` attachments.
      Files are pre-extracted by the server, so call this only if the attachment
      text seems missing or you want to re-parse.
    - GitHub MCP tools (prefix `mcp_github__` or `github__`): repos, issues, PRs, branches,
      files, commits. Use when the user references GitHub repos, code reviews, or workflows.
    - Notion MCP tools (prefix `mcp_notion__` or `notion__`): pages, databases, blocks.
      Use when the user references Notion docs, wikis, or notes.

    Rules:
    - Combine internal + external sources when useful; cite both clearly.
    - If Hybrid_Search is empty, say so explicitly rather than guessing.
    - MCP tools require explicit user intent — don't call them speculatively.
    - Keep answers concise. Use markdown links for citations (only URLs returned by tools).
    - You don't see the raw tool output; only the user-facing response is shown.
"""


def _mcp_tool_prefixes() -> list[str]:
    """Names of any MCP tools loaded into the agent, for the system prompt."""
    return [t.name for t in _MCP_TOOLS]


def _build_instructions() -> str:
    names = _mcp_tool_prefixes()
    if not names:
        return _BASE_INSTRUCTIONS
    listed = ", ".join(sorted(names))
    return _BASE_INSTRUCTIONS + f"\n    Loaded MCP tools: {listed}\n"


_MCP_TOOLS: list[BaseTool] = []


def _build_mcp_connections() -> dict[str, Connection]:
    """Build MCP server connections from settings; empty if no credentials."""
    connections: dict[str, Connection] = {}

    if settings.GITHUB_PAT:
        connections["github"] = StreamableHttpConnection(
            transport="streamable_http",
            url=settings.MCP_GITHUB_SERVER_URL,
            headers={
                "Authorization": f"Bearer {settings.GITHUB_PAT.get_secret_value()}",
            },
        )

    if settings.NOTION_API_KEY and settings.MCP_NOTION_SERVER_URL:
        connections["notion"] = StreamableHttpConnection(
            transport="streamable_http",
            url=settings.MCP_NOTION_SERVER_URL,
            headers={
                "Authorization": f"Bearer {settings.NOTION_API_KEY.get_secret_value()}",
            },
        )

    return connections


async def _load_mcp_tools() -> list[BaseTool]:
    """Connect to MCP servers and fetch their tools; empty list on failure."""
    connections = _build_mcp_connections()
    if not connections:
        logger.info("No MCP credentials configured; gugu-agent will run without MCP tools.")
        return []

    try:
        client = MultiServerMCPClient(connections)
        tools = await client.get_tools()
        logger.info(
            "gugu-agent loaded %d MCP tools from servers: %s",
            len(tools),
            sorted(connections.keys()),
        )
        return tools
    except Exception as e:
        logger.error("Failed to load MCP tools for gugu-agent: %s", e)
        return []


# Inline base64 images larger than this get pre-described via `understand_image`
# and replaced with text in the message history. The 500 KB threshold catches
# screenshots and high-resolution photos that bloat prompts and occasionally
# trigger upstream content moderation, while leaving small icons/thumbnails
# alone so vision-capable models still see them directly.
_LARGE_IMAGE_B64_BYTES = 500 * 1024

# Tool retry policy for transient failures (network resets, empty MCP errors,
# brief TLS hiccups). Non-transient errors propagate immediately.
_TOOL_RETRY_ATTEMPTS = 3
_TOOL_RETRY_BASE_DELAY_S = 1.0


def _parse_data_url(url: str) -> tuple[str | None, str | None]:
    """Return (mime_type, base64_payload) for a data: URL, else (None, None)."""
    if not url.startswith("data:"):
        return None, None
    head, _, payload = url.partition(",")
    # `data:image/png;base64,iVBOR...`
    meta = head[len("data:"):]  # "image/png;base64"
    parts = meta.split(";")
    mime = parts[0] if parts else ""
    if "base64" not in parts:
        return mime or None, None
    return mime or None, payload


async def _describe_image_block(b64: str, mime: str) -> str:
    """Run `understand_image` in a worker thread (the SDK is sync)."""
    return await asyncio.to_thread(
        understand_image_func, b64, mime, "Describe this image concisely."
    )


async def _resolve_attachments(state: AgentState) -> AgentState:
    """Pre-process large image/file attachments in the newest human message.

    Anthropic's content moderation occasionally rejects prompts containing
    large inline base64 images, which is the dominant failure mode in
    LangSmith. Stripping the bytes and replacing them with a text description
    keeps the LLM informed while keeping the prompt small.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}
    last = messages[-1]
    if not isinstance(last, HumanMessage):
        return {}
    content = last.content
    if not isinstance(content, list):
        return {}

    new_blocks: list = []
    changed = False

    for block in content:
        if not isinstance(block, dict):
            new_blocks.append(block)
            continue

        btype = block.get("type")

        if btype == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            mime, b64 = _parse_data_url(url)
            if mime and b64 and len(b64) > _LARGE_IMAGE_B64_BYTES:
                try:
                    desc = await _describe_image_block(b64, mime)
                except Exception as e:
                    logger.warning(
                        "understand_image failed, keeping original block: %s", e
                    )
                    new_blocks.append(block)
                    continue
                new_blocks.append(
                    {"type": "text", "text": f"[Attached image]\n{desc}"}
                )
                changed = True
                continue
            new_blocks.append(block)

        elif btype == "file":
            file_meta = block.get("file") or {}
            filename = file_meta.get("filename", "uploaded")
            b64 = file_meta.get("data_b64", "")
            if b64:
                try:
                    text = await asyncio.to_thread(
                        understand_document_func, b64, filename
                    )
                except Exception as e:
                    logger.warning(
                        "understand_document failed, keeping original block: %s", e
                    )
                    new_blocks.append(block)
                    continue
                new_blocks.append(
                    {"type": "text", "text": f"[Attached file: {filename}]\n{text}"}
                )
                changed = True
                continue
            new_blocks.append(block)

        else:
            new_blocks.append(block)

    if not changed:
        return {}

    return {"messages": [HumanMessage(content=new_blocks, id=last.id)]}


def _is_transient_tool_error(exc: BaseException) -> bool:
    """Heuristic: which tool errors are worth retrying."""
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    name = type(exc).__name__
    msg = str(exc).strip().lower()
    if name == "McpError" and not msg:
        # Empty MCP errors are almost always a connection/TLS hiccup that
        # the upstream library surfaced as "".
        return True
    if "ssl" in msg or "reset" in msg or "eof" in msg or "timed out" in msg:
        return True
    return False


class RetryingToolNode:
    """Drop-in replacement for `ToolNode` with transient-error retry."""

    def __init__(self, tools: list[BaseTool], attempts: int = _TOOL_RETRY_ATTEMPTS) -> None:
        self._node = ToolNode(tools)
        self._attempts = attempts

    async def __call__(self, state: AgentState, config: RunnableConfig | None = None) -> AgentState:
        last: BaseException | None = None
        for i in range(self._attempts):
            try:
                return await self._node.ainvoke(state, config)  # type: ignore[arg-type]
            except Exception as e:
                last = e
                if not _is_transient_tool_error(e) or i == self._attempts - 1:
                    break
                wait = _TOOL_RETRY_BASE_DELAY_S * (2 ** i)
                logger.warning(
                    "tool_node retry %d/%d after %.1fs: %s",
                    i + 1,
                    self._attempts,
                    wait,
                    e,
                )
                await asyncio.sleep(wait)
        assert last is not None
        raise last


class GuguAgent(LazyLoadingAgent):
    """Gugu agent with async MCP tool loading."""

    def __init__(self) -> None:
        super().__init__()
        self._all_tools: list[BaseTool] = []

    async def load(self) -> None:
        """Load MCP tools then compile the graph with local + MCP tools bound."""
        global _MCP_TOOLS
        _MCP_TOOLS = await _load_mcp_tools()
        self._all_tools = [*_LOCAL_TOOLS, *_MCP_TOOLS]
        self._graph = _build_graph(self._all_tools)
        self._loaded = True


def _build_graph(bound_tools: list[BaseTool]):
    instructions = _build_instructions()
    tool_node = RetryingToolNode(bound_tools)

    def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
        bound = model.bind_tools(bound_tools)
        preprocessor = RunnableLambda(
            lambda state: [SystemMessage(content=instructions)] + state["messages"],
            name="StateModifier",
        )
        return preprocessor | bound  # type: ignore[return-value]

    async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
        m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
        response = await wrap_model(m).ainvoke(state, config)

        if state["remaining_steps"] < 2 and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        return {"messages": [response]}

    async def safeguard_input(state: AgentState, config: RunnableConfig) -> AgentState:
        safeguard = Safeguard()
        safety_output = await safeguard.ainvoke(state["messages"])
        return {"safety": safety_output, "messages": []}

    async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
        safety: SafeguardOutput = state["safety"]
        return {"messages": [_format_safety_message(safety)]}

    def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
        safety: SafeguardOutput = state["safety"]
        match safety.safety_assessment:
            case SafetyAssessment.UNSAFE:
                return "unsafe"
            case _:
                return "safe"

    def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError(f"Expected AIMessage, got {type(last_message)}")
        if last_message.tool_calls:
            return "tools"
        return "done"

    graph = StateGraph(AgentState)
    graph.add_node("model", acall_model)
    graph.add_node("tools", tool_node)
    graph.add_node("guard_input", safeguard_input)
    graph.add_node("block_unsafe_content", block_unsafe_content)
    graph.add_node("resolve_attachments", _resolve_attachments)
    graph.set_entry_point("resolve_attachments")
    graph.add_edge("resolve_attachments", "guard_input")

    graph.add_conditional_edges(
        "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
    )
    graph.add_edge("block_unsafe_content", END)
    graph.add_edge("tools", "model")
    graph.add_conditional_edges(
        "model", pending_tool_calls, {"tools": "tools", "done": END}
    )

    compiled = graph.compile()
    compiled.name = "gugu-agent"
    return compiled


def _format_safety_message(safety: SafeguardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


gugu_agent = GuguAgent()