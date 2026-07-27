"""Tests for the hybrid retrieval pipeline in agents.rag_tools."""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from agents import rag_tools


def _fake_docs(n: int) -> list[Document]:
    return [Document(page_content=f"doc {i}", metadata={"src": f"s{i}"}) for i in range(n)]


def test_jieba_tokenize_splits_chinese():
    tokens = rag_tools.jieba_tokenize("员工手册")
    assert "员工" in tokens
    assert "手册" in tokens


def test_jieba_tokenize_preserves_english_whitespace_split():
    tokens = rag_tools.jieba_tokenize("remote work policy")
    assert "remote" in tokens
    assert "work" in tokens
    assert "policy" in tokens


def test_hybrid_search_returns_top_n_without_rerank(monkeypatch):
    docs = _fake_docs(20)
    monkeypatch.setattr(rag_tools, "_ensemble_retriever", lambda: MagicMock(invoke=lambda q: docs))
    monkeypatch.setattr(rag_tools, "_reranker", lambda: None)

    out = rag_tools.hybrid_search_func("anything")

    assert "[doc 1]" in out
    assert "[doc 5]" in out
    assert "[doc 6]" not in out
    assert out.count("[doc") == rag_tools.RERANK_TOP_N


def test_hybrid_search_rerank_trims_to_top_n(monkeypatch):
    docs = _fake_docs(40)
    monkeypatch.setattr(rag_tools, "_ensemble_retriever", lambda: MagicMock(invoke=lambda q: docs))

    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [
        {"id": i, "text": f"reordered {i}", "meta": {"src": f"r{i}"}, "score": 1.0 / (i + 1)}
        for i in range(40)
    ]
    monkeypatch.setattr(rag_tools, "_reranker", lambda: fake_reranker)

    out = rag_tools.hybrid_search_func("anything")

    assert out.count("[doc") == rag_tools.RERANK_TOP_N
    # Top score (id=0) should appear as [doc 1]
    assert "reordered 0" in out


def test_hybrid_search_falls_back_when_rerank_fails(monkeypatch):
    docs = _fake_docs(20)
    monkeypatch.setattr(rag_tools, "_ensemble_retriever", lambda: MagicMock(invoke=lambda q: docs))

    fake_reranker = MagicMock()
    fake_reranker.rerank.side_effect = RuntimeError("boom")
    monkeypatch.setattr(rag_tools, "_reranker", lambda: fake_reranker)

    out = rag_tools.hybrid_search_func("anything")

    assert out.count("[doc") == rag_tools.RERANK_TOP_N
    assert "doc 0" in out  # original ordering preserved
