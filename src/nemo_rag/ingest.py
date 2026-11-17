"""Pipeline di ingestione: file -> testo -> chunk -> embedding -> vector store."""
from __future__ import annotations

from pathlib import Path

from .chunking import chunk_text
from .clients import Embedder
from .config import Settings
from .loaders import iter_files, load_file
from .store import VectorStore, chunk_id


def ingest_path(path: Path, s: Settings, store: VectorStore, embedder: Embedder) -> tuple[int, int]:
    """Ritorna (file_processati, chunk_totali)."""
    n_files = n_chunks = 0
    for f in iter_files(path):
        source = str(f.resolve())
        texts: list[str] = []
        payloads: list[dict] = []
        for page in load_file(f):
            for piece in chunk_text(page.text, s.chunk_size, s.chunk_overlap):
                texts.append(piece)
                payloads.append({"source": source, "filename": f.name, "page": page.page, "text": piece})
        if not texts:
            print(f"  [skip] {f.name}: nessun testo estratto")
            continue

        vectors = embedder.embed(texts, input_type="passage")
        store.ensure_collection(len(vectors[0]))
        store.delete_source(source)  # evita chunk orfani se il file è cambiato
        store.upsert([chunk_id(source, i) for i in range(len(texts))], vectors, payloads)

        n_files += 1
        n_chunks += len(texts)
        print(f"  [ok] {f.name}: {len(texts)} chunk")
    return n_files, n_chunks
