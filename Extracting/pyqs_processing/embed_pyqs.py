# embed_pyqs.py
import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_HOST, CHROMA_PORT = "localhost", 8000
COLLECTION_NAME = "pyq_bank"
EMBED_MODEL = "all-MiniLM-L6-v2"

def load_pyq_file(path: Path, skip_headers: bool = True) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if skip_headers:
        records = [r for r in records if r["metadata"].get("type") != "header"]
    return records

def main(processed_dir="Extracting/Processed"):
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBED_MODEL)

    all_records = []
    for f in Path(processed_dir).glob("*_pyq.json"):
        all_records.extend(load_pyq_file(f))

    if not all_records:
        print("No PYQ files found (expected *_pyq.json)")
        return

    ids = [r["id"] for r in all_records]
    docs = [r["document"] for r in all_records]
    metas = [r["metadata"] for r in all_records]
    embeddings = model.encode(docs, show_progress_bar=True).tolist()

    collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    print(f"Done. '{COLLECTION_NAME}' now has {collection.count()} vectors.")

if __name__ == "__main__":
    main()