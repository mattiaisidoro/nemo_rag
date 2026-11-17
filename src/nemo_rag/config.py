"""Configurazione caricata da .env (nessun URL hardcoded nel codice)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Variabile d'ambiente mancante: {name} (controlla il file .env)")
    return value


@dataclass(frozen=True)
class Settings:
    embed_base_url: str
    embed_model: str
    rerank_base_url: str
    rerank_model: str
    llm_base_url: str
    llm_model: str
    llm_system_extra: str
    api_key: str
    qdrant_path: Path
    collection: str
    chunk_size: int
    chunk_overlap: int
    retrieve_k: int
    rerank_top_n: int
    request_timeout: float


@lru_cache
def get_settings() -> Settings:
    return Settings(
        embed_base_url=_required("EMBED_BASE_URL"),
        embed_model=_required("EMBED_MODEL"),
        rerank_base_url=_required("RERANK_BASE_URL"),
        rerank_model=_required("RERANK_MODEL"),
        llm_base_url=_required("LLM_BASE_URL"),
        llm_model=_required("LLM_MODEL"),
        llm_system_extra=os.getenv("LLM_SYSTEM_EXTRA", ""),
        api_key=os.getenv("NIM_API_KEY", "not-used"),
        qdrant_path=Path(os.getenv("QDRANT_PATH", "./data/qdrant")),
        collection=os.getenv("QDRANT_COLLECTION", "nego_docs"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        retrieve_k=int(os.getenv("RETRIEVE_K", "20")),
        rerank_top_n=int(os.getenv("RERANK_TOP_N", "5")),
        request_timeout=float(os.getenv("REQUEST_TIMEOUT", "120")),
    )
