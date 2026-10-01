"""
check_chroma.py -- is everything actually inside ChromaDB?

Compares what is IN Chroma against what SHOULD be there (the chunk / PYQ
files in this repo), for all three collections:

    book_content  <- Extracting/Processed/*_chunks.jsonl            (embed_chunks.py)
    tut_content   <- Extracting/Processed/*_tutorial_chunks.jsonl   (embed_tutorial_chunks.py)
    pyq_bank      <- Extracting/Processed/pyq_structured/*_pyq.json         (Embedding/embed_pyqs.py)

It does NOT load the embedding model -- it only reads ids + metadata, so it
runs in a few seconds.

USAGE (run from the repo root, after `docker compose up -d`)
    python check_chroma.py
    python check_chroma.py --port 8000 --host localhost
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "Extracting" / "Processed"
PYQ_DIR = ROOT / "Extracting" / "Processed" / "pyq_structured"
PYQ_MIN_DOC_CHARS = 60          # same filter embed_pyqs.py uses
PAGE = 1000


# ------------------------------------------------------------------ expected ids
def expected_book_ids():
    ids = set()
    for f in PROCESSED.glob("*_chunks.jsonl"):
        if f.name.endswith("_tutorial_chunks.jsonl"):
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(json.loads(line)["chunk_id"])
    return ids


def expected_tutorial_ids():
    ids = set()
    for f in PROCESSED.glob("*_tutorial_chunks.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                for t in (c.get("topic_ids") or ["NONE"]):
                    ids.add(f"{c['chunk_id']}__{t}")
    return ids


def expected_pyq_ids():
    ids = set()
    for f in PYQ_DIR.glob("*_pyq.json"):
        for r in json.loads(f.read_text(encoding="utf-8")):
            if r["metadata"].get("type") == "header":
                continue
            if len(r["document"].strip()) < PYQ_MIN_DOC_CHARS:
                continue
            ids.add(r["id"])
    return ids


# ------------------------------------------------------------------ chroma helpers
def collection_names(client):
    # chromadb 0.6 returns names, chromadb 1.x returns Collection objects
    return {c if isinstance(c, str) else c.name for c in client.list_collections()}


def fetch_all(coll):
    """All ids + metadatas, paged so a big collection doesn't time out."""
    ids, metas, offset = [], [], 0
    while True:
        part = coll.get(include=["metadatas"], limit=PAGE, offset=offset)
        if not part["ids"]:
            break
        ids += part["ids"]
        metas += part["metadatas"]
        offset += len(part["ids"])
    return ids, metas


# ------------------------------------------------------------------ report
def check_collection(client, name, expected, script, group_key="subject"):
    print(f"\n=== {name} ===")
    if name not in collection_names(client):
        print(f"  NOT FOUND in Chroma.  ->  run: python {script}")
        return None

    coll = client.get_collection(name)
    ids, metas = fetch_all(coll)
    have = set(ids)
    missing = expected - have
    extra = have - expected

    print(f"  vectors in Chroma : {len(have)}")
    print(f"  expected from files: {len(expected)}")
    by_group = Counter(m.get(group_key, "?") for m in metas)
    print("  per subject       : " + ", ".join(f"{k}={v}" for k, v in sorted(by_group.items())))

    if not missing and not extra:
        print("  STATUS            : OK - Chroma matches the files exactly")
    if missing:
        print(f"  STATUS            : INCOMPLETE - {len(missing)} expected vector(s) missing "
              f"(e.g. {sorted(missing)[:3]})  ->  re-run: python {script}")
    if extra:
        print(f"  NOTE              : {len(extra)} vector(s) in Chroma that are NOT in the files "
              f"anymore (e.g. {sorted(extra)[:3]}). Stale leftovers from an older run; "
              f"retrieval skips them only if they are tagged NONE / needs_review.")

    no_topic = sum(1 for m in metas if m.get("topic_id") in (None, "NONE"))
    print(f"  topic_id = NONE   : {no_topic} of {len(metas)}")
    return metas


def topic_coverage(book_metas, tut_metas):
    """How many usable grounding vectors each graph topic has (book + tutorial)."""
    usable = defaultdict(Counter)
    for label, metas in (("book", book_metas or []), ("tutorial", tut_metas or [])):
        for m in metas:
            t = m.get("topic_id")
            if t and t != "NONE" and not m.get("needs_review", False):
                usable[t][label] += 1

    all_topics = set(usable)
    subjects_dir = ROOT / "prerequisite_graph" / "subjects"
    try:
        import yaml
        for f in subjects_dir.glob("*.y*ml"):
            for t in yaml.safe_load(f.read_text(encoding="utf-8")).get("topics", []):
                all_topics.add(t["id"])
    except ImportError:
        pass

    print("\n=== grounding available per topic (book + tutorial) ===")
    for t in sorted(all_topics):
        b, tu = usable[t]["book"], usable[t]["tutorial"]
        flag = "   <-- NO grounding text: generation for this topic will fail" if b + tu == 0 else ""
        print(f"  {t:9} book={b:4}  tutorial={tu:3}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import chromadb
    client = chromadb.HttpClient(host=args.host, port=args.port)
    try:
        client.heartbeat()
    except Exception as e:
        raise SystemExit(f"Cannot reach Chroma at {args.host}:{args.port} ({e}).\n"
                         f"Is it running?  ->  docker compose up -d")

    print(f"Collections on the server: {sorted(collection_names(client))}")

    book = check_collection(client, "book_content", expected_book_ids(),
                            "Embedding/embed_chunks.py")
    tut = check_collection(client, "tut_content", expected_tutorial_ids(),
                           "Embedding/embed_tutorial_chunks.py")
    check_collection(client, "pyq_bank", expected_pyq_ids(),
                     "Embedding/embed_pyqs.py")
    topic_coverage(book, tut)


if __name__ == "__main__":
    main()
