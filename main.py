from pathlib import Path

from nemo_rag.clients import LLM, Embedder, Reranker
from nemo_rag.config import get_settings
from nemo_rag.ingest import ingest_path
from nemo_rag.rag import answer
from nemo_rag.retrieve import retrieve
from nemo_rag.store import VectorStore

# --- COSA VUOI FARE? ---
MODALITA = "ingest"  # "ingest" = indicizza i documenti | "ask" = fai una domanda
DOMANDA = "Scrivi qui la tua domanda"

s = get_settings()
store = VectorStore(s)
embedder = Embedder(s)

if MODALITA == "ingest":
    files, chunks = ingest_path(Path("docs"), s, store, embedder)
    print(f"Fatto: {files} file, {chunks} chunk. Totale nel database: {store.count()}")
else:
    reranker = Reranker(s)
    llm = LLM(s)
    chunks = retrieve(DOMANDA, s, store, embedder, reranker)
    print(answer(DOMANDA, chunks, llm, s))
    print("\nFonti:")
    for c in chunks:
        print("-", c["filename"], c.get("page"))