"""Retrieval a due stadi: ricerca vettoriale (top-K) + reranking (top-N)."""
from __future__ import annotations

from .clients import Embedder, Reranker
from .config import Settings
from .store import VectorStore


def retrieve(
    query: str, s: Settings, store: VectorStore, embedder: Embedder, reranker: Reranker
) -> list[dict]:
    qvec = embedder.embed([query], input_type="query")[0]
    candidates = store.search(qvec, s.retrieve_k)
    if not candidates:
        return []
    ranked = reranker.rerank(query, [c["text"] for c in candidates])
    top = []
    for idx, score in ranked[: s.rerank_top_n]:
        top.append({**candidates[idx], "rerank_score": score})
    return top
