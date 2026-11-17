"""Vector store: Qdrant in modalità locale (su disco, nessun server)."""
from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

from .config import Settings


def chunk_id(source: str, idx: int) -> str:
    """ID deterministico: ri-ingerire lo stesso file sovrascrive i chunk."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{idx}"))


class VectorStore:
    def __init__(self, s: Settings):
        s.qdrant_path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(s.qdrant_path))
        self._name = s.collection

    def ensure_collection(self, dim: int) -> None:
        if not self._client.collection_exists(self._name):
            self._client.create_collection(
                self._name,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    def delete_source(self, source: str) -> None:
        if not self._client.collection_exists(self._name):
            return
        self._client.delete(
            self._name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
                )
            ),
        )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        points = [
            models.PointStruct(id=i, vector=v, payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]
        self._client.upsert(self._name, points=points)

    def search(self, vector: list[float], k: int) -> list[dict]:
        if not self._client.collection_exists(self._name):
            return []
        res = self._client.query_points(
            self._name, query=vector, limit=k, with_payload=True
        ).points
        return [{**(p.payload or {}), "score": p.score} for p in res]

    def count(self) -> int:
        if not self._client.collection_exists(self._name):
            return 0
        return self._client.count(self._name).count
