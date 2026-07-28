"""Tests for the text-only extraction in agents.safeguard."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.safeguard import Safeguard, safeguard_instructions


def _compile(messages):
    """Build a no-model Safeguard instance and call _compile_messages on it."""
    s = Safeguard.__new__(Safeguard)
    s.model = None
    s.system_prompt = SystemMessage(content=safeguard_instructions)
    return s._compile_messages(messages)


def test_compile_string_content_passes_through():
    out = _compile([HumanMessage(content="hi there")])
    user_msg = out[-1]
    assert "User: hi there" in user_msg.content


def test_compile_list_content_extracts_text_blocks_only():
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "x" * 6000}},
            {"type": "text", "text": "thanks"},
        ]
    )
    out = _compile([msg])
    body = out[-1].content
    assert "describe this" in body
    assert "thanks" in body
    # The base64 payload must NOT leak into the safeguard prompt.
    assert "image/png;base64" not in body
    assert "x" * 100 not in body


def test_compile_skips_ai_tool_messages_without_role():
    # Only human + ai are mapped. A list-only HumanMessage should still work.
    msg = HumanMessage(content=[{"type": "text", "text": "q1"}])
    out = _compile([msg, AIMessage(content="answer", id="a1")])
    body = out[-1].content
    assert "User: q1" in body
    assert "Agent: answer" in body


def test_compile_empty_list_content_yields_empty_user_string():
    msg = HumanMessage(content=[])
    out = _compile([msg])
    # Just ensure no crash; user segment should be present but empty.
    assert "User:" in out[-1].content