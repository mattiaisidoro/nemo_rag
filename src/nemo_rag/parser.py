"""Parsing dei PDF con Nemotron Parse (endpoint NIM /v1/parse).

Il modello accetta UNA pagina alla volta, come immagine. Quindi:
PDF -> immagini (una per pagina, 300 DPI) -> una chiamata per pagina -> testo Markdown.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pypdfium2 as pdfium
import requests

# Un elemento della risposta ha questa forma:
# <x_0.07><y_0.47>testo dell'elemento<x_0.50><y_0.50><class_Section-header>
# (angolo in alto a sinistra, testo, angolo in basso a destra, classe semantica)
_ELEMENT = re.compile(
    r"<x_[\d.]+><y_[\d.]+>(.*?)<x_[\d.]+><y_[\d.]+><class_([^>]+)>", re.DOTALL
)
_TOKENS = re.compile(r"<(?:x|y)_[\d.]+>|<class_[^>]+>")


def clean_content(content: str) -> str:
    """Toglie i token di coordinate/classe e tiene solo il testo, un elemento per paragrafo."""
    elements = [text.strip() for text, _cls in _ELEMENT.findall(content)]
    if elements:
        return "\n\n".join(e for e in elements if e)
    return _TOKENS.sub("", content).strip()  # fallback se il formato fosse diverso


def parse_image(png_bytes: bytes, parse_url: str, timeout: float = 600) -> str:
    response = requests.post(
        parse_url,
        files={"document": ("page.png", io.BytesIO(png_bytes), "image/png")},
        headers={"Accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("finish_reason") == "length":
        print("  [attenzione] pagina troncata (limite di output raggiunto)")
    return clean_content(result["content"])


def parse_pdf(path: Path, parse_url: str, timeout: float = 600) -> list[str]:
    """Ritorna una lista di testi, uno per pagina."""
    pdf = pdfium.PdfDocument(str(path))
    pages: list[str] = []
    for page in pdf:
        image = page.render(scale=300 / 72).to_pil()  # 72 unità PDF per pollice -> 300 DPI
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        pages.append(parse_image(buffer.getvalue(), parse_url, timeout))
    return pages