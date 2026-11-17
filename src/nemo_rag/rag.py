"""Generazione della risposta a partire dai chunk recuperati."""
from __future__ import annotations

from .clients import LLM
from .config import Settings

SYSTEM_PROMPT = (
    "Sei un assistente aziendale. Rispondi usando ESCLUSIVAMENTE il contesto fornito. "
    "Se la risposta non è nel contesto, dì chiaramente che non lo sai. "
    "Cita le fonti con il loro numero, ad esempio [1], [2]. "
    "Rispondi nella stessa lingua della domanda."
)


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        loc = c["filename"] + (f", p.{c['page']}" if c.get("page") else "")
        blocks.append(f"[{i}] ({loc})\n{c['text']}")
    return "\n\n".join(blocks)


def answer(query: str, chunks: list[dict], llm: LLM, s: Settings) -> str:
    system = SYSTEM_PROMPT + (f"\n{s.llm_system_extra}" if s.llm_system_extra else "")
    user = f"Contesto:\n{build_context(chunks)}\n\nDomanda: {query}"
    return llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
