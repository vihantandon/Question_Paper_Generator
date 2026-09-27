"""
embed_tutorial_chunks.py -- embeds ONLY the tutorial chunk files produced by
ingest_tutorials.py. Kept separate from embed_chunks.py on purpose (that
script stays untouched for the book pipeline); this one only ever looks at
files named "*_tutorial_chunks.jsonl", so running both is safe and neither
will double-process the other's files.

Requires:
    pip install chromadb sentence-transformers pyyaml tqdm

Run this AFTER docker compose up -d (the Chroma server must be running
on localhost:8000).

WHY MULTI-TOPIC CHUNKS ARE SPLIT
    A tutorial chunk's "topic_ids" is a LIST (one chunk can cover more than
    one topic), but a Chroma metadata value must be a single scalar -- you
    can't filter on a list field. So a chunk with N topic_ids becomes N
    Chroma documents here (ids "<chunk_id>__<topic_id>"), same text, each
    with ONE topic_id. This keeps retrieval.py's simple
    where={"topic_id": topic_id} filter working without any changes, for
    either content type. The full original topic set is kept too, as a
    comma-joined reference field (topic_ids_all / topics_all) -- not meant
    to be filtered on, just so you can see everything a chunk covered.

Writes into the SAME Chroma collection the book pipeline uses
("book_content"), tagged "type": "tutorial", so retrieval.py finds both
book and tutorial grounding text for a given topic_id without extra code.

USAGE
    python embed_tutorial_chunks.py                  # process every subject
    python embed_tutorial_chunks.py --subject APS     # just one subject
    python embed_tutorial_chunks.py --dry-run         # show counts, no Chroma/model calls
"""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "tut_content"
PROCESSED_DIR = ROOT / "Extracting" / "Processed"
TOPIC_MAP_FILE = ROOT / "Extracting" / "topic_map.yaml"
SUBJECTS_DIR = ROOT / "prerequisite_graph" / "subjects"

EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
NONE_TOPIC = "NONE"


# --------------------------------------------------------------------------- graph metadata
def load_topic_index():
    """topic_id -> {subject, semester}, straight from the graph's own subject
    YAMLs -- the same source tag_topics.py and ingest_tutorials.py read."""
    index = {}
    subject_semester = {}
    for f in sorted(SUBJECTS_DIR.glob("*.y*ml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        subject_semester[data["subject"]] = data["semester"]
        for t in data.get("topics", []):
            index[t["id"]] = {"subject": data["subject"], "semester": data["semester"]}
    return index, subject_semester


def load_book_to_yaml_subject():
    """Tutorial-folder-name -> graph subject (e.g. 'APS' -> 'ALGO'), reusing
    topic_map.yaml's own books: bridge -- same as ingest_tutorials.py."""
    if not TOPIC_MAP_FILE.exists():
        return {}
    cfg = yaml.safe_load(TOPIC_MAP_FILE.read_text(encoding="utf-8"))
    return {book: b["yaml_subject"] for book, b in cfg.get("books", {}).items()}


# --------------------------------------------------------------------------- loading + normalizing
def load_tutorial_files(processed_dir: Path, only_subject: str | None):
    """(filename, subject, [raw dicts]) for every *_tutorial_chunks.jsonl file."""
    for jsonl_file in sorted(processed_dir.glob("*_tutorial_chunks.jsonl")):
        subject = jsonl_file.name[: -len("_tutorial_chunks.jsonl")]
        if only_subject and subject != only_subject:
            continue
        with open(jsonl_file, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        yield jsonl_file.name, subject, records


def explode_chunk(c: dict, source_file: str, tutorial_subject: str,
                   tutorial_semester: int, topic_index: dict) -> list[dict]:
    """One tutorial chunk -> one row PER topic_id (see module docstring)."""
    topic_ids = c.get("topic_ids") or [NONE_TOPIC]
    topics = c.get("topics") or []
    topics_all = ",".join(topics)
    topic_ids_all = ",".join(topic_ids)
    cos_all = ",".join(c.get("cos", []))

    rows = []
    for topic_id in topic_ids:
        info = topic_index.get(topic_id)
        topic_subject = info["subject"] if info else NONE_TOPIC
        topic_semester = info["semester"] if info else 0
        if info is None:
            print(f"  WARNING: topic_id '{topic_id}' (chunk {c['chunk_id']}) not found "
                  f"in prerequisite_graph/subjects/*.yaml -- check for a typo.")
        rows.append({
            "id": f"{c['chunk_id']}__{topic_id}",
            "text": c["description"],
            "subject": tutorial_subject,
            "section_title": "Tutorial",
            "page_start": -1,
            "page_end": -1,
            "source_file": source_file,
            "type": "tutorial",
            "topic_id": topic_id,
            "topic_subject": topic_subject,
            "topic_semester": topic_semester,
            "is_prereq_content": bool(info and info["semester"] < tutorial_semester),
            "chapter": -1,
            "tag_method": "gemini-tutorial",
            "needs_review": topic_id == NONE_TOPIC,
            "chunk_group_id": c["chunk_id"],
            "topics_all": topics_all,
            "topic_ids_all": topic_ids_all,
            "cos_all": cos_all,
        })
    return rows


def build_rows(processed_dir: Path, only_subject: str | None,
                book_to_yaml: dict, topic_index: dict, subject_semester: dict):
    rows = []
    chunk_count = 0
    for source_file, tutorial_subject, records in load_tutorial_files(processed_dir, only_subject):
        yaml_subject = book_to_yaml.get(tutorial_subject, tutorial_subject)
        tutorial_semester = subject_semester.get(yaml_subject, 0)
        print(f"{source_file}: {len(records)} chunk(s), subject '{tutorial_subject}' "
              f"-> graph subject '{yaml_subject}' (semester {tutorial_semester})")
        for c in records:
            rows.extend(explode_chunk(c, source_file, tutorial_subject, tutorial_semester, topic_index))
        chunk_count += len(records)
    return rows, chunk_count


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default=None,
                    help="Only embed <SUBJECT>_tutorial_chunks.jsonl (default: all subjects found)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be embedded, without touching Chroma or loading the model")
    args = ap.parse_args()

    print("Reading tutorial chunk files...")
    topic_index, subject_semester = load_topic_index()
    book_to_yaml = load_book_to_yaml_subject()
    rows, chunk_count = build_rows(PROCESSED_DIR, args.subject, book_to_yaml, topic_index, subject_semester)

    if not rows:
        print(f"No *_tutorial_chunks.jsonl files found under {PROCESSED_DIR}"
              + (f" for subject '{args.subject}'" if args.subject else "") + ".")
        return

    print(f"\n{chunk_count} tutorial chunk(s) -> {len(rows)} vector(s) to upsert "
          f"(a multi-topic chunk becomes one vector per topic_id).")

    if args.dry_run:
        print("\n--dry-run: not connecting to Chroma or loading the embedding model. Sample rows:")
        for r in rows[:5]:
            print(f"  {r['id']}  | topic_id={r['topic_id']}  | topic_subject={r['topic_subject']}")
        return

    print("\nConnecting to ChromaDB...")
    import chromadb
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    print(f"Loading local embedding model: {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Embedding + upserting"):
        batch = rows[i:i + BATCH_SIZE]
        texts = [r["text"] for r in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=[r["id"] for r in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{k: v for k, v in r.items() if k not in ("id", "text")} for r in batch],
        )

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} vectors.")


if __name__ == "__main__":
    main()
