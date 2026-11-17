"""CLI: nemorag ingest | search | ask | chat | stats"""
from __future__ import annotations

import argparse
from pathlib import Path

from .clients import LLM, Embedder, Reranker
from .config import get_settings
from .ingest import ingest_path
from .rag import answer
from .retrieve import retrieve
from .store import VectorStore


def _print_sources(chunks: list[dict]) -> None:
    print("\nFonti:")
    for i, c in enumerate(chunks, 1):
        page = f", p.{c['page']}" if c.get("page") else ""
        print(f"  [{i}] {c['filename']}{page}  (rerank={c['rerank_score']:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(prog="nemorag", description="RAG con Nemotron su NVIDIA Spark")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="Indicizza un file o una cartella")
    p.add_argument("path", type=Path)

    p = sub.add_parser("search", help="Solo retrieval (nessuna generazione)")
    p.add_argument("query")

    p = sub.add_parser("ask", help="Domanda singola con risposta generata")
    p.add_argument("query")

    sub.add_parser("chat", help="Sessione interattiva")
    sub.add_parser("stats", help="Numero di chunk indicizzati")

    args = parser.parse_args()
    s = get_settings()
    store = VectorStore(s)

    if args.cmd == "stats":
        print(f"Chunk indicizzati: {store.count()}")
        return

    embedder = Embedder(s)

    if args.cmd == "ingest":
        files, chunks = ingest_path(args.path, s, store, embedder)
        print(f"Fatto: {files} file, {chunks} chunk.")
        return

    reranker = Reranker(s)

    if args.cmd == "search":
        for i, c in enumerate(retrieve(args.query, s, store, embedder, reranker), 1):
            print(f"\n[{i}] {c['filename']} (rerank={c['rerank_score']:.2f})\n{c['text'][:500]}")
        return

    llm = LLM(s)

    def run(q: str) -> None:
        chunks = retrieve(q, s, store, embedder, reranker)
        if not chunks:
            print("Nessun documento indicizzato o nessun risultato.")
            return
        print("\n" + answer(q, chunks, llm, s))
        _print_sources(chunks)

    if args.cmd == "ask":
        run(args.query)
    elif args.cmd == "chat":
        print("Scrivi una domanda (vuoto o 'exit' per uscire).")
        while True:
            q = input("\n> ").strip()
            if not q or q.lower() in {"exit", "quit"}:
                break
            run(q)


if __name__ == "__main__":
    main()
