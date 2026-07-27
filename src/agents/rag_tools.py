"""Hybrid retrieval: BM25 (sparse) + Chroma (dense) fused with FlashRank rerank.

Pipeline: query -> BM25 top-K + Chroma top-K -> RRF ensemble -> FlashRank rerank -> top-N.
"""

import logging
from functools import lru_cache

import jieba
from flashrank import Ranker
from flashrank.Ranker import RerankRequest
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

BM25_K = 20
DENSE_K = 20
RERANK_TOP_N = 5
ENSEMBLE_WEIGHTS = [0.5, 0.5]
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
PERSIST_DIR = "./chroma_db"


def jieba_tokenize(text: str) -> list[str]:
    """BM25 tokenizer: jieba for CJK, preserves ASCII tokens."""
    return [t for t in jieba.cut(text) if t.strip()]


@lru_cache(maxsize=1)
def _embeddings():
    return OpenAIEmbeddings()


@lru_cache(maxsize=1)
def _chroma():
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings())


@lru_cache(maxsize=1)
def _dense_retriever():
    return _chroma().as_retriever(search_kwargs={"k": DENSE_K})


@lru_cache(maxsize=1)
def _bm25_corpus() -> list[str]:
    return _chroma().get(include=["documents"])["documents"]


@lru_cache(maxsize=1)
def _bm25_retriever() -> BM25Retriever:
    return BM25Retriever.from_documents(
        [Document(page_content=t) for t in _bm25_corpus()],
        preprocess_func=jieba_tokenize,
        k=BM25_K,
    )


@lru_cache(maxsize=1)
def _ensemble_retriever() -> EnsembleRetriever:
    return EnsembleRetriever(
        retrievers=[_dense_retriever(), _bm25_retriever()],
        weights=ENSEMBLE_WEIGHTS,
    )


@lru_cache(maxsize=1)
def _reranker() -> Ranker | None:
    try:
        return Ranker(model_name=RERANK_MODEL)
    except Exception as e:
        logger.warning("FlashRank unavailable, rerank step will be skipped: %s", e)
        return None


def _format(docs: list[Document]) -> str:
    return "\n\n".join(f"[doc {i + 1}] {d.page_content}" for i, d in enumerate(docs))


def hybrid_search_func(query: str) -> str:
    """Search the in-house knowledge base with hybrid retrieval + cross-encoder rerank.

    Returns the top-5 most relevant passages reranked from a ~40-candidate pool
    (BM25 + dense, fused via Reciprocal Rank Fusion). Use for questions about
    internal documents, policies, or handbook content. Do NOT use for real-time
    information (news, weather) — call other tools instead.

    Args:
        query: natural-language question.

    Returns:
        Top-5 passages joined by blank lines, each prefixed with `[doc N]`.
    """
    candidates = _ensemble_retriever().invoke(query)
    reranker = _reranker()
    if reranker is None:
        return _format(candidates[:RERANK_TOP_N])

    passages = [
        {"id": i, "text": d.page_content, "meta": d.metadata}
        for i, d in enumerate(candidates)
    ]
    try:
        ranked = reranker.rerank(RerankRequest(query=query, passages=passages))
        ranked.sort(key=lambda r: r["score"], reverse=True)
        return _format(
            [Document(page_content=r["text"], metadata=r["meta"]) for r in ranked[:RERANK_TOP_N]]
        )
    except Exception as e:
        logger.warning("Rerank failed, returning ensemble order: %s", e)
        return _format(candidates[:RERANK_TOP_N])


hybrid_search: BaseTool = tool(hybrid_search_func)
hybrid_search.name = "Hybrid_Search"
