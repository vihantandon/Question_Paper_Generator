"""
embed_pyqs.py -- embeds the structured PYQ records (one atomic record per
question) into their OWN Chroma collection, "pyq_bank", on the docker
Chroma server (localhost:8000), same as the book/tutorial pipelines.

Requires:
    pip install chromadb sentence-transformers tqdm
Run AFTER `docker compose up -d`.

USAGE
    python embed_pyqs.py                 # embed everything found
    python embed_pyqs.py --dry-run       # show what would be embedded
    python embed_pyqs.py --dir <path>    # folder containing *_pyq.json
                                         # (default: Extracting/Processed/pyq_structured)
    python embed_pyqs.py --reset         # delete + rebuild the collection
"""
# --- MUST come before any sentence_transformers / transformers import ------
# transformers tries to import TensorFlow if it is installed, and your
# TF + protobuf versions clash. We only need PyTorch, so tell it to skip TF.
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ---------------------------------------------------------------------------

import argparse
import json
from pathlib import Path

CHROMA_HOST, CHROMA_PORT = "localhost", 8000
COLLECTION_NAME = "pyq_bank"
EMBED_MODEL = "all-MiniLM-L6-v2"      # same model as book_content / tut_content
BATCH_SIZE = 64
MIN_DOC_CHARS = 60                    # shorter than this = extraction junk

HERE = Path(__file__).resolve().parent            # Extracting/pyqs_processing
ROOT = HERE.parent.parent                          # repo root
DEFAULT_DIR = ROOT / "Extracting" / "Processed" / "pyq_structured"


def sanitize_metadata(meta: dict) -> dict:
    """Chroma metadata values must be str / int / float / bool.
    None and lists/dicts are not allowed, so convert them."""
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = -1 if k == "marks" else "NONE"
        elif isinstance(v, (list, dict)):
            clean[k] = json.dumps(v, ensure_ascii=False)   # e.g. marks_breakdown
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def load_records(folder: Path):
    records, skipped = [], []
    files = sorted(folder.glob("*_pyq.json"))
    for f in files:
        for r in json.loads(f.read_text(encoding="utf-8")):
            if r["metadata"].get("type") == "header":
                continue
            if len(r["document"].strip()) < MIN_DOC_CHARS:
                skipped.append((r["id"], r["document"].strip()[:40]))
                continue
            records.append(r)
    return files, records, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true",
                    help="delete the pyq_bank collection first, then re-embed")
    args = ap.parse_args()

    folder = Path(args.dir)
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}\nPass --dir <folder with *_pyq.json>")

    files, records, skipped = load_records(folder)
    print(f"{len(files)} PYQ file(s) in {folder}")
    print(f"{len(records)} question record(s) to embed")
    if skipped:
        print(f"Skipped {len(skipped)} junk record(s) (too short):")
        for rid, snippet in skipped:
            print(f"   {rid}: {snippet!r}")

    if not records:
        print("Nothing to embed (expected *_pyq.json files).")
        return

    ids = [r["id"] for r in records]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SystemExit(f"Duplicate ids found: {sorted(dupes)}")

    if args.dry_run:
        for r in records[:5]:
            print(f"  {r['id']} | topic_id={r['metadata'].get('topic_id')} "
                  f"| marks={r['metadata'].get('marks')}")
        print("--dry-run: nothing sent to Chroma.")
        return

    import chromadb
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    client.heartbeat()   # fails fast if docker isn't up
    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted old '{COLLECTION_NAME}'")
        except Exception:
            pass
    collection = client.get_or_create_collection(COLLECTION_NAME)

    print(f"Loading embedding model: {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Embedding + upserting"):
        batch = records[i:i + BATCH_SIZE]
        docs = [r["document"] for r in batch]
        embeddings = model.encode(docs, show_progress_bar=False).tolist()
        collection.upsert(
            ids=[r["id"] for r in batch],
            embeddings=embeddings,
            documents=docs,
            metadatas=[sanitize_metadata(r["metadata"]) for r in batch],
        )

    print(f"Done. '{COLLECTION_NAME}' now has {collection.count()} vectors.")


if __name__ == "__main__":
    main()