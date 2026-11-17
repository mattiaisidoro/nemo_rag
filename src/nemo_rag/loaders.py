"""Lettura dei file: PDF (tramite Nemotron Parse), DOCX, TXT, MD."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import get_settings
from .parser import parse_pdf

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class Page:
    text: str
    page: int | None = None  # numero pagina (solo PDF)


def load_file(path: Path) -> list[Page]:
    ext = path.suffix.lower()

    if ext == ".pdf":
        s = get_settings()
        if not s.parse_url:
            raise RuntimeError("PARSE_URL mancante nel file .env")
        texts = parse_pdf(path, s.parse_url)  # una stringa per pagina, da Nemotron Parse
        return [Page(text, i + 1) for i, text in enumerate(texts)]

    if ext == ".docx":
        from docx import Document

        doc = Document(str(path))
        return [Page("\n\n".join(p.text for p in doc.paragraphs if p.text.strip()))]

    if ext in {".txt", ".md"}:
        return [Page(path.read_text(encoding="utf-8", errors="ignore"))]

    raise ValueError(f"Formato non supportato: {ext}")


def iter_files(root: Path):
    if root.is_file():
        if root.suffix.lower() in SUPPORTED:
            yield root
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            yield p