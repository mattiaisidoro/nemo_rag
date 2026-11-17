"""Chunking per paragrafi con overlap (in caratteri)."""
from __future__ import annotations

import re


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    step = max(size - overlap, 1)
    return [text[i : i + size] for i in range(0, len(text), step)]


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for p in paragraphs:
        units.extend(_split_long(p, size, overlap) if len(p) > size else [p])

    chunks: list[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) + len(u) + 2 > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = f"{tail}\n\n{u}" if tail else u
        else:
            cur = f"{cur}\n\n{u}" if cur else u
    if cur:
        chunks.append(cur)
    return chunks
