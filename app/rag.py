from pypdf import PdfReader
import os
import chromadb
from sentence_transformers import SentenceTransformer

CONTEXT_DIR = "/data/context"
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 100   # overlap between consecutive chunks

client = chromadb.EphemeralClient()
collection = client.get_or_create_collection("agora_context")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def read_file(path):
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks so sentence-transformers can embed them."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_context():
    """Read every PDF/TXT file in CONTEXT_DIR, chunk it, and index in ChromaDB."""
    print(f"Loading context documents from {CONTEXT_DIR}...")

    chunk_id = 0
    for filename in sorted(os.listdir(CONTEXT_DIR)):
        path = os.path.join(CONTEXT_DIR, filename)
        if not os.path.isfile(path):
            continue
        if not (filename.endswith(".pdf") or filename.endswith(".txt")):
            continue

        print(f"  Indexing: {filename}")
        text = read_file(path)

        if not text.strip():
            print(f"  Warning: {filename} is empty — skipping.")
            continue

        chunks = chunk_text(text)
        print(f"  → {len(chunks)} chunks")

        embeddings = embed_model.encode(chunks).tolist()
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[str(chunk_id + i) for i in range(len(chunks))],
            metadatas=[{"source": filename} for _ in chunks],
        )
        chunk_id += len(chunks)

    print(f"Context ready: {chunk_id} chunks indexed.\n")


def query_context(query, n_results=5):
    """Return the most relevant chunks from historical minutes for the given query."""
    emb = embed_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[emb],
        n_results=n_results,
    )

    parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        parts.append(f"[Source: {meta['source']}]\n{doc}")

    return "\n\n---\n\n".join(parts)
