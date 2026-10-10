"""
Requires:
    pip install chromadb sentence-transformers tqdm
 
Run this AFTER docker compose up -d (the Chroma server must be running;
its address comes from settings.py: .env, else localhost:8000).

Safe to re-run: chunks are upserted by chunk_id, and vectors whose chunk is
no longer in the chunk files are deleted, so "book_content" always matches
Extracting/Processed/*_chunks.jsonl (see stale_vectors.py).
"""
# --- MUST come before any sentence_transformers / transformers import ------
# transformers tries to import TensorFlow if it is installed, and a clashing
# TF + protobuf install then breaks the import. We only need PyTorch, so tell
# it to skip TF (the other embed scripts do the same).
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ---------------------------------------------------------------------------

import json
import sys
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from stale_vectors import remove_stale_vectors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                      # settings.py
from settings import CHROMA_HOST, CHROMA_PORT      # noqa: E402

COLLECTION_NAME = "book_content"
PROCESSED_DIR = ROOT / "Extracting" / "Processed"

EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

def  load_all_chunks(processed_dir: Path):
    chunks = []
    for jsonl_file in processed_dir.glob("*_chunks.jsonl"):
        if jsonl_file.name.endswith("_tutorial_chunks.jsonl"):
            continue
        with open (jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

    return chunks

def main():
    print("Connecting to ChromaDB...")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    print(f"Loading local embedding model: {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Reading chunk files...")
    chunks = load_all_chunks(PROCESSED_DIR)
    if any("topic_id" not in c for c in chunks):
        print("WARNING: some chunks have no topic_id - run  python tag_topics.py  first.")
    print(f"Found {len(chunks)} chunks across "
          f"{len(set(c['subject'] for c in chunks))} subjects.")

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding + upserting"):
        batch = chunks[i:i + BATCH_SIZE]
 
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()


        collection.upsert(
            ids = [c["chunk_id"] for c in batch],
            embeddings = embeddings,
            documents = texts,
            metadatas = [{
                "subject": c["subject"],
                "section_title": c["section_title"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "source_file": c["source_file"],
                "type": "book",

                "topic_id": c.get("topic_id", "NONE"),
                "topic_subject": c.get("topic_subject", "NONE"),
                "topic_semester": c.get("topic_semester", 0),
                "is_prereq_content": c.get("is_prereq_content", False),
                "chapter": c.get("chapter", -1),
                "tag_method": c.get("tag_method", "untagged"),
                "needs_review": c.get("needs_review", False),
            }for c in batch],
        )

    stale = remove_stale_vectors(collection, [c["chunk_id"] for c in chunks])
    if stale:
        print(f"Removed {len(stale)} stale vector(s) whose chunk is no longer in the "
              f"chunk files (e.g. {stale[:3]}).")
    print(f"Done. Collection '{COLLECTION_NAME}' now has "
    f"{collection.count()} vectors.")

if __name__ == "__main__":
    main()
