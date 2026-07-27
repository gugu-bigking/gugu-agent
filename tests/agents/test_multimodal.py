"""Tests for the multimodal additions in agents.tools + service + client."""

import base64
from unittest.mock import MagicMock, patch

import pytest

from agents import tools as tools_module
from client.client import Attachment, _build_message
from schema import ChatMessage, UserInput
from service.service import _build_human_content


def test_understand_document_extracts_markdown():
    md_bytes = b"# Title\n\nHello *world*."
    md_b64 = base64.b64encode(md_bytes).decode("ascii")
    out = tools_module.understand_document_func(md_b64, "notes.md")
    assert "Hello *world*" in out


def test_understand_document_extracts_plain_text():
    txt = "plain text content"
    out = tools_module.understand_document_func(
        base64.b64encode(txt.encode()).decode("ascii"), "readme.txt"
    )
    assert out == txt


def test_understand_document_unsupported_type_raises():
    with pytest.raises(ValueError, match="Unsupported file type"):
        tools_module.understand_document_func(base64.b64encode(b"x").decode("ascii"), "x.pdf")


def test_understand_image_calls_openai_with_data_url(monkeypatch):
    captured = {}

    class _Choice:
        message = MagicMock(content="a cat sitting on a chair")

    class _Resp:
        choices = [_Choice()]

    def _fake_create(**kwargs):
        captured.update(kwargs)
        return _Resp()

    monkeypatch.setattr("openai.OpenAI", lambda *a, **kw: MagicMock(chat=MagicMock(completions=MagicMock(create=_fake_create))))

    out = tools_module.understand_image_func(
        base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii"),
        mime_type="image/png",
        prompt="What do you see?",
    )

    assert out == "a cat sitting on a chair"
    msg_content = captured["messages"][0]["content"]
    assert msg_content[0] == {"type": "text", "text": "What do you see?"}
    assert msg_content[1]["type"] == "image_url"
    assert msg_content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_human_content_passes_string_through():
    assert _build_human_content("hello") == "hello"


def test_build_human_content_resolves_docx_attachment():
    fake_text = "extracted docx text"
    with patch.object(tools_module, "understand_document_func", return_value=fake_text):
        result = _build_human_content([
            {"type": "file", "file": {"filename": "policy.docx", "data_b64": "AAAA"}}
        ])
    # Single resolved text block collapses to a plain string.
    assert isinstance(result, str)
    assert "policy.docx" in result
    assert fake_text in result


def test_build_human_content_passes_image_url_through():
    blocks = _build_human_content([
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        {"type": "text", "text": "describe"},
    ])
    assert isinstance(blocks, list)
    assert {b["type"] for b in blocks} == {"image_url", "text"}


def test_build_human_content_collapses_single_text_block():
    blocks = _build_human_content([{"type": "text", "text": "only text"}])
    assert blocks == "only text"


def test_user_input_accepts_list_of_content_blocks():
    payload = UserInput(message=[
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
    ])
    assert isinstance(payload.message, list)


def test_chat_message_accepts_list_content():
    msg = ChatMessage(type="ai", content=[
        {"type": "text", "text": "ok"}
    ])
    assert isinstance(msg.content, list)


def test_client_build_message_text_only_collapses_to_str():
    assert _build_message("hi", None) == "hi"


def test_client_build_message_with_image_attachment():
    att = Attachment(filename="a.png", mime_type="image/png", bytes=b"\x89PNG\r\n\x1a\n")
    out = _build_message("what?", [att])
    assert isinstance(out, list)
    types = [b["type"] for b in out]
    assert types == ["text", "image_url"]


def test_client_build_message_with_doc_attachment():
    att = Attachment(filename="notes.md", mime_type="text/markdown", bytes=b"# hi")
    out = _build_message("", [att])
    assert isinstance(out, list)
    assert out[0]["type"] == "file"
    assert out[0]["file"]["filename"] == "notes.md"


def test_client_attachment_rejects_both_bytes_and_b64():
    att = Attachment(filename="a.png", mime_type="image/png", bytes=b"x", data_b64="eA==")
    with pytest.raises(Exception, match="only one"):
        from client.client import _attachment_to_block

        _attachment_to_block(att)


def test_client_attachment_requires_bytes_or_b64():
    att = Attachment(filename="a.png", mime_type="image/png")
    with pytest.raises(Exception, match="must provide"):
        from client.client import _attachment_to_block

        _attachment_to_block(att)