"""Client HTTP per gli endpoint locali: embedding, reranking, LLM.

Versione didattica con la libreria `requests`.
Ogni client fa la stessa cosa: manda un POST con un JSON a un URL e legge il JSON di risposta.
"""
from __future__ import annotations

import requests

from .config import Settings


class BaseClient:
    """Logica comune: costruisce l'URL, manda la richiesta POST, controlla gli errori."""

    def __init__(self, base_url: str, s: Settings):
        self.base_url = base_url.rstrip("/")  # es. http://IP:8001/v1
        self.timeout = s.request_timeout
        # La Session riusa la connessione e mantiene gli header per tutte le chiamate
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {s.api_key}",
                "Content-Type": "application/json",
            }
        )

    def post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}/{path}"  # es. http://IP:8001/v1/embeddings
        response = self.session.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()  # se il server risponde con un errore (4xx/5xx), solleva eccezione
        return response.json()


class Embedder(BaseClient):
    """POST {base}/embeddings (formato OpenAI, con input_type di NIM)."""

    def __init__(self, s: Settings, batch_size: int = 32):
        super().__init__(s.embed_base_url, s)
        self.model = s.embed_model
        self.batch_size = batch_size

    def embed(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        """input_type: 'passage' per i documenti, 'query' per le domande."""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            data = self.post(
                "embeddings",
                {
                    "model": self.model,
                    "input": batch,
                    "input_type": input_type,
                    "encoding_format": "float",
                    "truncate": "END",
                },
            )["data"]
            data = sorted(data, key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        return out


class Reranker(BaseClient):
    """POST {base}/ranking (API reranker dei NIM)."""

    def __init__(self, s: Settings):
        super().__init__(s.rerank_base_url, s)
        self.model = s.rerank_model

    def rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        """Ritorna [(indice_passaggio, score)] ordinato dal più al meno rilevante."""
        if not passages:
            return []
        rankings = self.post(
            "ranking",
            {
                "model": self.model,
                "query": {"text": query},
                "passages": [{"text": p} for p in passages],
                "truncate": "END",
            },
        )["rankings"]
        pairs = [(item["index"], float(item["logit"])) for item in rankings]
        return sorted(pairs, key=lambda x: x[1], reverse=True)


class LLM(BaseClient):
    """POST {base}/chat/completions (formato OpenAI)."""

    def __init__(self, s: Settings):
        super().__init__(s.llm_base_url, s)
        self.model = s.llm_model

    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 1024) -> str:
        data = self.post(
            "chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        return data["choices"][0]["message"]["content"]