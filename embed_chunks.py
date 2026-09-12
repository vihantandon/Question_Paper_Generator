"""
Requires:
    pip install chromadb sentence-transformers tqdm
 
Run this AFTER docker-compose up -d (the Chroma server must be running
on localhost:8000).
"""
import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "book_content"
PROCESSED_DIR = Path("Extracting/processed")

EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

def  load_all_chunks(processed_dir: Path):
    chunks = []
    for jsonl_file in processed_dir.glob("*_chunks.jsonl"):
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

            }for c in batch],
        )
    print(f"Done. Collection '{COLLECTION_NAME}' now has "
    f"{collection.count()} vectors.")

if __name__ == "__main__":
    main()
