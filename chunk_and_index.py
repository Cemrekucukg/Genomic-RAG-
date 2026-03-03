import json
import os
import re
from typing import List, Dict, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# Variant pattern (DNA + protein)
# Examples: c.2T>C, c.5A>G, (p.Met1Thr)
VARIANT_PATTERN = re.compile(
    r"(c\.\d+[A-Za-z0-9_+-]+>[A-Za-z0-9_+-]+|\(p\.[A-Za-z]+\d+[A-Za-z]+\))"
)


def load_articles(path: str) -> List[Dict]:
    articles: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            articles.append(json.loads(line))
    return articles


def smart_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Safeguard to reduce cutting at the end of a variant pattern.
    """
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]

        # If a variant ends very close to boundary, extend a bit
        last_match = None
        for m in VARIANT_PATTERN.finditer(chunk):
            last_match = m

        if last_match is not None and (len(chunk) - last_match.end() < 8) and end < n:
            end = min(end + 40, n)
            chunk = text[start:end]

        chunks.append(chunk)
        start += max(chunk_size - overlap, 1)

    return chunks


def write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_chunks(articles: List[Dict]) -> List[Dict]:
    all_chunks: List[Dict] = []
    for art in articles:
        abstract = art.get("abstract", "") or ""
        pmid = str(art.get("pmid", "") or "").strip()
        doi: Optional[str] = art.get("doi")

        chunks = smart_chunk(abstract)
        for i, c in enumerate(chunks):
            all_chunks.append(
                {
                    "id": f"{pmid}_{i}",
                    "pmid": pmid,
                    "doi": doi,
                    "text": c,
                }
            )
    return all_chunks


def index_chroma(
    chunks: List[Dict],
    persist_dir: str = "chroma_db",
    collection_name: str = "rars1_pubmed",
    embedding_model: str = "BAAI/bge-base-en-v1.5",
) -> Tuple[int, int]:
    """
    Returns (n_added, n_total_in_collection_after).
    """
    os.makedirs(persist_dir, exist_ok=True)

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    # Recreate collection to avoid duplicates during dev
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    col = client.get_or_create_collection(name=collection_name)

    model = SentenceTransformer(embedding_model)

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"pmid": c["pmid"], "doi": c["doi"]} for c in chunks]

    # Embed in batches
    embeddings = model.encode(documents, batch_size=32, show_progress_bar=True).tolist()

    col.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    count = col.count()
    return len(ids), count


if __name__ == "__main__":
    # 1) Load PubMed abstracts
    articles = load_articles("data/pubmed_rars1_abstracts.jsonl")

    # 2) Build chunks
    chunks = build_chunks(articles)

    # 3) Save chunks (runtime output)
    write_jsonl("data/rars1_chunks.jsonl", chunks)
    print("Total chunks:", len(chunks))

    # 4) Index into ChromaDB
    added, total = index_chroma(chunks)
    print(f"Indexed into Chroma. Added={added}, CollectionCount={total}")