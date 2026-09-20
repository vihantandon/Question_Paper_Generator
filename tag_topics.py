"""
tag_topics.py  --  give every book chunk a topic_id (the graph <-> vector bridge)
 
WHAT IT DOES
    1. Reads Extracting/Processed/*_chunks.jsonl
    2. Works out which BOOK CHAPTER each chunk sits in
    3. Looks the chapter up in Extracting/topic_map.yaml  -> graph topic_id
       (chapters that mix topics are split chunk-by-chunk with keyword scoring)
    4. Checks every topic_id really exists in prerequisite_graph/subjects/*.yaml
    5. Writes the new fields back into the jsonl files (safe to re-run)
    6. Prints a coverage report (which topics have NO book text = gaps)
    7. Optionally (--push) patches the metadata of the vectors ALREADY in
       ChromaDB, so you do not have to re-embed anything.
 
USAGE
    pip install pyyaml chromadb
    python tag_topics.py                 # tag the jsonl files + print report
    python tag_topics.py --report-only   # print report, change nothing
    python tag_topics.py --push          # also update the running Chroma server
 
NEW METADATA FIELDS (all Chroma-safe: str / int / bool only)
    topic_id          "DS_03"   the graph node this chunk teaches ("NONE" if none)
    topic_subject     "DS"      which subject owns that topic
    topic_semester    3         semester of that topic (0 if NONE)
    is_prereq_content True      topic is from an EARLIER semester than the book's subject
    chapter           5         chapter number in the book (-1 if unknown)
    tag_method        chapter-rule | section-rule | keyword-split | keyword-lowconf | unlisted-chapter
                      | front-back-matter | suspect-page-range
    needs_review      True      a human should glance at this one
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
 
import yaml
 
ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "Extracting" / "Processed"
MAP_FILE = ROOT / "Extracting" / "topic_map.yaml"
SUBJECTS_DIR = ROOT / "prerequisite_graph" / "subjects"
 
CHROMA_HOST, CHROMA_PORT, COLLECTION = "localhost", 8000, "book_content"
NONE = "NONE"
MAX_PAGE_SPAN = 200      # a real book section never spans more pages than this
 
TAG_FIELDS = ["topic_id", "topic_subject", "topic_semester",
              "is_prereq_content", "chapter", "tag_method", "needs_review"]
 
 
# ----------------------------------------------------------------- loading
def load_topics():
    """topic_id -> {name, subject, semester}   (straight from the graph YAMLs)"""
    topics = {}
    for f in sorted(SUBJECTS_DIR.glob("*.y*ml")):
        d = yaml.safe_load(f.read_text(encoding="utf-8"))
        for t in d["topics"]:
            topics[t["id"]] = {"name": t["name"], "subject": d["subject"],
                               "semester": d["semester"]}
    return topics
 
 
def load_map(topics):
    cfg = yaml.safe_load(MAP_FILE.read_text(encoding="utf-8"))
    kw = cfg.get("topic_keywords", {})
    subject_sem = {t["subject"]: t["semester"] for t in topics.values()}
    problems = []
    for book, b in cfg["books"].items():
        if b["yaml_subject"] not in subject_sem:
            problems.append(f"{book}: yaml_subject '{b['yaml_subject']}' not found in subject YAMLs")
        for ch, cands in b["chapters"].items():
            for t in cands:
                if t not in topics:
                    problems.append(f"{book} ch{ch}: unknown topic id '{t}'")
                if len(cands) > 1 and t not in kw:
                    problems.append(f"{book} ch{ch}: '{t}' needs topic_keywords (multi-topic chapter)")
    if problems:
        sys.exit("topic_map.yaml problems:\n  - " + "\n  - ".join(problems))
    return cfg, subject_sem
 
 
# ------------------------------------------------------------ chapter logic
def chapter_of(title, bcfg, current):
    """Chapter number for this chunk's section title (or None)."""
    fb = bcfg.get("frontback_regex")
    if fb and re.match(fb, title, re.I):
        return None                                  # front/back matter, Part headers
    m = re.match(bcfg["chapter_regex"], title)
    if m:
        return int(m.group(1))
    return current if bcfg.get("inherit_unnumbered") else None
 
 
def rule_topic(title, bcfg, cands):
    """Optional hand-written rule: section-number pattern -> topic (beats keywords)."""
    for r in bcfg.get("section_rules", []):
        if r["topic"] in cands and re.match(r["regex"], title):
            return r["topic"]
    return None
 
 
def keyword_scores(text, cands, kw):
    low = text.lower()
    return {t: sum(low.count(k.lower()) for k in kw[t]) for t in cands}
 
 
def tag_book(chunks, book, bcfg, topics, subject_sem, kw, min_margin):
    book_sem = subject_sem[bcfg["yaml_subject"]]
    current = None
    for c in chunks:                                  # file order == book order
        current = chapter_of(c["section_title"], bcfg, current)
        cands = bcfg["chapters"].get(current) if current is not None else None
 
        if current is None:
            topic, method, review = NONE, "front-back-matter", False
        elif cands is None:
            topic, method, review = NONE, "unlisted-chapter", False
        elif len(cands) == 1:
            topic, method, review = cands[0], "chapter-rule", False
        elif rule_topic(c["section_title"], bcfg, cands):
            topic, method, review = rule_topic(c["section_title"], bcfg, cands), "section-rule", False
        else:
            s = keyword_scores(c["text"], cands, kw)
            ranked = sorted(cands, key=lambda t: (-s[t], cands.index(t)))
            gap = s[ranked[0]] - s[ranked[1]]
            if gap >= min_margin:
                topic, method, review = ranked[0], "keyword-split", False
            else:                                     # too close to call
                topic, method, review = cands[0], "keyword-lowconf", True
 
        # Extraction bug guard: a bad PDF bookmark can make one "section" cover
        # the whole book (DS: "53.3.3 Corner Block List" = pages 1-1086).
        if c["page_end"] - c["page_start"] > MAX_PAGE_SPAN:
            topic, method, review = NONE, "suspect-page-range", True
 
        info = topics.get(topic)
        c.update({
            "topic_id": topic,
            "topic_subject": info["subject"] if info else NONE,
            "topic_semester": info["semester"] if info else 0,
            "is_prereq_content": bool(info and info["semester"] < book_sem),
            "chapter": current if current is not None else -1,
            "tag_method": method,
            "needs_review": review,
        })
    return chunks
 
 
# ------------------------------------------------------------------ report
def report(all_chunks, topics, cfg):
    print("\n=== TAGGING SUMMARY ===")
    for book, chunks in all_chunks.items():
        m = Counter(c["tag_method"] for c in chunks)
        tagged = sum(1 for c in chunks if c["topic_id"] != NONE)
        pre = sum(1 for c in chunks if c["is_prereq_content"])
        rev = sum(1 for c in chunks if c["needs_review"])
        print(f"{book:6} {len(chunks):5} chunks | with a topic: {tagged:5} "
              f"({100*tagged//len(chunks)}%) | prerequisite-content: {pre:4} | needs review: {rev}")
        print("       methods:", dict(m))
 
    per_topic = defaultdict(lambda: Counter())
    for book, chunks in all_chunks.items():
        for c in chunks:
            if c["topic_id"] != NONE:
                per_topic[c["topic_id"]][book] += 1
 
    sus = [c for chunks in all_chunks.values() for c in chunks
           if c["tag_method"] == "suspect-page-range"]
    if sus:
        secs = sorted({(c["chunk_id"].rsplit("_", 1)[0], c["section_title"],
                        c["page_start"], c["page_end"]) for c in sus})
        print(f"\n!! {len(sus)} chunks come from sections with an impossible page range "
              f"(extraction bug, mostly duplicate text):")
        for sid, t, a, b in secs:
            print(f"   {sid}  '{t}'  pages {a}-{b}")
        print("   They are tagged NONE. Use --drop-suspect to delete them.")
 
    print("\n=== CHUNKS PER TOPIC (from all books) ===")
    for tid, info in topics.items():
        n = per_topic.get(tid)
        total = sum(n.values()) if n else 0
        flag = "   <-- GAP: no book text" if total == 0 else ""
        detail = ", ".join(f"{b}:{k}" for b, k in n.items()) if n else ""
        print(f"{tid:9} {total:5}  {detail:28} {info['name'][:52]}{flag}")
 
    print("\n=== BIGGEST 'UNLISTED CHAPTER' BUCKETS (tagged NONE) - skim for mistakes ===")
    for book, chunks in all_chunks.items():
        cnt = Counter((c["chapter"], c["section_title"]) for c in chunks
                      if c["tag_method"] == "unlisted-chapter" and c["chapter"] >= 0)
        by_ch = Counter()
        title_of = {}
        for (ch, t), k in cnt.items():
            by_ch[ch] += k
            title_of.setdefault(ch, t)
        top = by_ch.most_common(5)
        if top:
            print(f"{book}: " + "; ".join(f"ch{ch} ({k} chunks)" for ch, k in top))
 
 
# -------------------------------------------------------------------- push
def push_to_chroma(client, all_chunks, batch=500):
    """Patch metadata on existing vectors. Merges with the old metadata first,
    so it works whether Chroma's update() merges or replaces."""
    coll = client.get_collection(COLLECTION)
    flat = [c for chunks in all_chunks.values() for c in chunks]
    done = missing = 0
    for i in range(0, len(flat), batch):
        part = flat[i:i + batch]
        got = coll.get(ids=[c["chunk_id"] for c in part], include=["metadatas"])
        old = dict(zip(got["ids"], got["metadatas"]))
        ids, metas = [], []
        for c in part:
            if c["chunk_id"] not in old:
                missing += 1
                continue
            merged = dict(old[c["chunk_id"]])
            merged.update({k: c[k] for k in TAG_FIELDS})
            ids.append(c["chunk_id"]); metas.append(merged)
        if ids:
            coll.update(ids=ids, metadatas=metas)
            done += len(ids)
    print(f"\nChroma: updated {done} vectors"
          + (f"  ({missing} chunk_ids not found in the collection - run embed_chunks.py)" if missing else ""))
 
 
# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--drop-suspect", action="store_true",
                    help="delete chunks with an impossible page range (jsonl + Chroma)")
    args = ap.parse_args()
 
    topics = load_topics()
    cfg, subject_sem = load_map(topics)
    kw, min_margin = cfg.get("topic_keywords", {}), cfg.get("min_margin", 2)
 
    files = {}
    for f in sorted(PROCESSED.glob("*_chunks.jsonl")):
        book = f.name.replace("_chunks.jsonl", "")
        if book not in cfg["books"]:
            print(f"skip {f.name}: no entry for '{book}' in topic_map.yaml")
            continue
        files[book] = f
 
    all_chunks = {}
    for book, f in files.items():
        chunks = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        all_chunks[book] = tag_book(chunks, book, cfg["books"][book], topics,
                                    subject_sem, kw, min_margin)
 
    dropped_ids = []
    if args.drop_suspect:
        for book in all_chunks:
            keep = [c for c in all_chunks[book] if c["tag_method"] != "suspect-page-range"]
            dropped_ids += [c["chunk_id"] for c in all_chunks[book]
                            if c["tag_method"] == "suspect-page-range"]
            all_chunks[book] = keep
 
    report(all_chunks, topics, cfg)
    if dropped_ids:
        print(f"\nDropped {len(dropped_ids)} suspect chunks from the run.")
 
    if not args.report_only:
        for book, f in files.items():
            with open(f, "w", encoding="utf-8") as out:
                for c in all_chunks[book]:
                    out.write(json.dumps(c, ensure_ascii=False) + "\n")
        print("\nWrote topic fields into Extracting/Processed/*_chunks.jsonl")
 
    if args.push:
        import chromadb
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        if dropped_ids:
            coll = client.get_collection(COLLECTION)
            for i in range(0, len(dropped_ids), 500):
                coll.delete(ids=dropped_ids[i:i + 500])
            print(f"Chroma: deleted {len(dropped_ids)} suspect vectors")
        push_to_chroma(client, all_chunks)
 
 
if __name__ == "__main__":
    main()