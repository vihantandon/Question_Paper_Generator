"""
resolve_pyq_labels.py -- clean up the labels on the PYQ records.

The PYQs are the project's "answer key": the classifier will be tested
against their CO / Bloom labels, and the drafting LLM will imitate them as
style examples. So their labels have to be right. This script fixes three
problems left over from split_pyq_questions.py:

  1. Bloom levels   16 spellings ("Apply L3", "Applying", "Apply Level", ...)
                    -> the 6 standard levels (Remember .. Create).
  2. OCR'd COs      "C03" / "C05" (a zero instead of the letter O) -> CO3 / CO5.
  3. Topics         A CO usually covers several topics, so the splitter could
                    not choose one. Here the question is compared with each
                    topic's book + tutorial chunks in ChromaDB, and the
                    closest topic wins. This also catches wrong tags such as
                    a quicksort question tagged "Linear DS" only because the
                    paper printed CO2.

HOW THE TOPIC IS CHOSEN
    score(topic) = average similarity between the question and that topic's
                   3 closest grounding chunks (book_content + tut_content).
    - Prefer the best topic among the question's CO topics.
    - If some OTHER topic of the subject matches clearly better
      (by MISMATCH_MARGIN), take it, but flag co_topic_mismatch so a human
      checks whether the YAML's CO -> topic mapping is missing a link.
    - If the winner beats the runner-up by less than LOW_CONFIDENCE_MARGIN,
      keep the choice but set needs_review.
    - Topics fixed by hand (review_reason mentions "manual") are never touched.

WHY IT RUNS AFTER EMBEDDING
    It reuses the question vectors already stored in pyq_bank, so it needs
    no embedding model (no PyTorch). Order of the PYQ pipeline:
        pdf_extract_full.py -> split_pyq_questions.py
        -> Embedding/embed_pyqs.py -> resolve_pyq_labels.py

USAGE (Chroma must be running: docker compose up -d)
    python3 Extracting/pyqs_processing/resolve_pyq_labels.py          # preview only
    python3 Extracting/pyqs_processing/resolve_pyq_labels.py --write  # save JSON + update pyq_bank
Re-running is safe: it gives the same result every time.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent            # Extracting/pyqs_processing
ROOT = HERE.parent.parent                          # repo root
PYQ_DIR = ROOT / "Extracting" / "Processed" / "pyq_structured"
SUBJECTS_DIR = ROOT / "prerequisite_graph" / "subjects"

sys.path.insert(0, str(HERE))                      # split_pyq_questions.py
sys.path.insert(0, str(ROOT / "Embedding"))        # embed_pyqs.py
from split_pyq_questions import normalize_code      # noqa: E402  one OCR rule, one place
from embed_pyqs import sanitize_metadata            # noqa: E402  same metadata rules as embedding

CHROMA_HOST, CHROMA_PORT = "localhost", 8000
PYQ_COLLECTION = "pyq_bank"
GROUNDING_COLLECTIONS = ("book_content", "tut_content")

TOP_K = 3                     # chunks averaged per topic
MISMATCH_MARGIN = 0.05        # a non-CO topic must win by this much to override the CO
LOW_CONFIDENCE_MARGIN = 0.01  # closer than this to the runner-up -> needs_review

BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
# word stems checked in order; the level NUMBER ("L3") is only a fallback,
# because a word survives OCR better than a single digit
BLOOM_STEMS = [
    ("remember", "Remember"), ("recall", "Remember"), ("knowledge", "Remember"),
    ("understand", "Understand"), ("comprehen", "Understand"),
    ("appl", "Apply"),
    ("analy", "Analyze"),
    ("evaluat", "Evaluate"),
    ("creat", "Create"), ("synthes", "Create"),
]
BLOOM_NUMBER_RE = re.compile(r"\bl\s*([1-6])\b")


# ----------------------------------------------------------------- pure logic
def normalize_bloom(raw):
    """'Apply L3' / 'Applying' / 'apply level' -> 'Apply'. None if unknown."""
    if not raw:
        return None
    text = str(raw).lower()
    for stem, level in BLOOM_STEMS:
        if stem in text:
            return level
    m = BLOOM_NUMBER_RE.search(text)
    return BLOOM_LEVELS[int(m.group(1)) - 1] if m else None


def decide_topic(co_candidates, scores, co_label="the CO"):
    """Pick a topic from similarity scores.

    co_candidates: topic ids the question's CO maps to ([] if CO unknown)
    scores:        {topic_id: similarity} for every subject topic that HAS
                   grounding text (topics with no text simply are not in it)
    Returns {topic_id, method, confidence, mismatch, reasons}.
    """
    result = {"topic_id": None, "method": "unresolved", "confidence": None,
              "mismatch": False, "reasons": []}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    scored_cands = {t: scores[t] for t in co_candidates if t in scores}

    # Nothing to compare against -> trust the paper's CO if it names one topic
    if not ranked or (co_candidates and not scored_cands):
        if len(co_candidates) == 1:
            result.update(topic_id=co_candidates[0], method="co-only")
            return result
        if not ranked:
            result["reasons"].append("no grounding text to compare the question with")
            return result
        result["reasons"].append(f"{co_label} topics have no grounding text; picked by content")

    best_all, best_all_score = ranked[0]
    if not scored_cands:                               # CO unknown / unusable
        chosen, method, pool = best_all, "content", dict(ranked)
    else:
        best_co, best_co_score = max(scored_cands.items(), key=lambda kv: kv[1])
        if best_all not in scored_cands and best_all_score - best_co_score >= MISMATCH_MARGIN:
            chosen, method, pool = best_all, "content-override", dict(ranked)
            result["mismatch"] = True
            result["reasons"].append(
                f"content matches {best_all} ({best_all_score:.3f}) better than every "
                f"{co_label} topic (best {best_co}, {best_co_score:.3f}); check the "
                f"CO -> topic mapping in the subject YAML")
        else:
            chosen, method, pool = best_co, "co+content", scored_cands

    others = [s for t, s in pool.items() if t != chosen]
    confidence = round(pool[chosen] - max(others), 4) if others else None
    if confidence is not None and confidence < LOW_CONFIDENCE_MARGIN:
        result["reasons"].append(
            f"low confidence: {chosen} beats the runner-up by only {confidence:.3f}")
    result.update(topic_id=chosen, method=method, confidence=confidence)
    return result


def resolve_record(meta, subject, scores):
    """Return a NEW metadata dict with cleaned labels (input is not modified).

    subject: {"cos": {...}, "topics": {topic_id: {"name", "co"}}} from the YAML
    scores:  {topic_id: similarity}, or None if the question has no vector yet
    """
    new = dict(meta)
    reasons = []

    # 1. Bloom level (keep the paper's original wording in bloom_level_raw)
    raw_bloom = meta.get("bloom_level_raw", meta.get("bloom_level"))
    new["bloom_level_raw"] = raw_bloom
    new["bloom_level"] = normalize_bloom(raw_bloom)
    new["bloom_rank"] = (BLOOM_LEVELS.index(new["bloom_level"]) + 1
                         if new["bloom_level"] else None)
    if raw_bloom and not new["bloom_level"]:
        reasons.append(f"unknown Bloom level '{raw_bloom}'")

    # 2. CO: repair OCR'd codes the splitter could not resolve
    if not meta.get("course_outcome") and meta.get("raw_co_code"):
        fixed = normalize_code(meta["raw_co_code"])
        if fixed in subject["cos"]:
            new["course_outcome"] = fixed
            new["co_description"] = subject["cos"][fixed]
    co = new.get("course_outcome")
    if meta.get("raw_co_code") and not co:
        reasons.append(f"CO code '{meta['raw_co_code']}' still unresolved")

    # 3. Topic
    new.setdefault("topic_id_from_co", meta.get("topic_id"))   # what the splitter chose
    manual = meta.get("tag_method") == "manual" or "manual" in (meta.get("review_reason") or "").lower()
    if manual:
        new["tag_method"] = "manual"
        return new                                  # a human decided: keep everything

    if scores is None:
        reasons.append("no vector in pyq_bank yet (run Embedding/embed_pyqs.py)")
        new.update(needs_review=True, review_reason="; ".join(reasons))
        return new

    co_candidates = [t for t, info in subject["topics"].items() if co and co in info["co"]]
    d = decide_topic(co_candidates, scores, co_label=co or "the CO")
    top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    new.update(
        topic_id=d["topic_id"],
        topic_name=subject["topics"][d["topic_id"]]["name"] if d["topic_id"] else None,
        tag_method=d["method"],
        topic_confidence=d["confidence"],
        co_topic_mismatch=d["mismatch"],
        topic_scores=json.dumps([[t, round(s, 4)] for t, s in top3]),
    )
    reasons += d["reasons"]
    new["needs_review"] = bool(reasons) or d["topic_id"] is None
    new["review_reason"] = "; ".join(reasons) or None
    return new


# ----------------------------------------------------------------- I/O
def load_subjects(subjects_dir=SUBJECTS_DIR):
    out = {}
    for f in sorted(Path(subjects_dir).glob("*.y*ml")):
        y = yaml.safe_load(f.read_text(encoding="utf-8"))
        out[y["subject"]] = {
            "cos": y.get("course_outcomes", {}),
            "topics": {t["id"]: {"name": t["name"], "co": t.get("co", [])} for t in y["topics"]},
        }
    return out


def topic_scores(question_vec, topic_ids, collections, k=TOP_K):
    """{topic_id: mean similarity of its k closest chunks}. Chroma returns
    squared L2 distance; for the unit-length MiniLM vectors that means
    cosine similarity = 1 - distance / 2."""
    scores = {}
    for t in topic_ids:
        dists = []
        for coll in collections:
            res = coll.query(query_embeddings=[question_vec], n_results=k,
                             where={"$and": [{"topic_id": t}, {"needs_review": False}]},
                             include=["distances"])
            dists += res["distances"][0]
        if dists:
            best = sorted(dists)[:k]
            scores[t] = sum(1 - d / 2 for d in best) / len(best)
    return scores


def load_pyq_files(folder=PYQ_DIR):
    files = sorted(Path(folder).glob("*_pyq.json"))
    if not files:
        raise SystemExit(f"No *_pyq.json files in {folder}")
    return {f: json.loads(f.read_text(encoding="utf-8")) for f in files}


def describe_change(old, new):
    parts = []
    if old.get("bloom_level") != new.get("bloom_level"):
        parts.append(f"bloom '{old.get('bloom_level')}'->{new.get('bloom_level')}")
    if old.get("course_outcome") != new.get("course_outcome"):
        parts.append(f"CO {old.get('course_outcome')}->{new.get('course_outcome')}")
    if old.get("topic_id") != new.get("topic_id"):
        parts.append(f"topic {old.get('topic_id')}->{new.get('topic_id')}")
    return ", ".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Clean Bloom / CO / topic labels on the PYQ records")
    ap.add_argument("--write", action="store_true",
                    help="save the JSON files and update pyq_bank (default: preview only)")
    ap.add_argument("--host", default=CHROMA_HOST)
    ap.add_argument("--port", type=int, default=CHROMA_PORT)
    args = ap.parse_args()

    import chromadb
    client = chromadb.HttpClient(host=args.host, port=args.port)
    existing = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    missing = [n for n in (PYQ_COLLECTION, *GROUNDING_COLLECTIONS) if n not in existing]
    if missing:
        raise SystemExit(f"Missing Chroma collection(s): {missing}. Run the embed scripts first.")
    pyq_coll = client.get_collection(PYQ_COLLECTION)
    grounding = [client.get_collection(n) for n in GROUNDING_COLLECTIONS]

    subjects = load_subjects()
    files = load_pyq_files()

    # one round-trip for all question vectors
    q_ids = [r["id"] for recs in files.values() for r in recs if r["metadata"]["type"] == "question"]
    got = pyq_coll.get(ids=q_ids, include=["embeddings"])
    vectors = dict(zip(got["ids"], got["embeddings"]))

    updated, counts = [], {}
    before_review = after_review = 0
    print(f"{'question':24} {'CO':4} {'method':16} {'conf':>6}  change / flags")
    for path, recs in files.items():
        for i, rec in enumerate(recs):
            old = rec["metadata"]
            if old["type"] != "question":
                continue
            subject = subjects.get(old["subject"])
            if subject is None:
                print(f"{rec['id']:24} skipped: no YAML for subject {old['subject']}")
                continue
            scores = None
            if rec["id"] in vectors:
                scores = topic_scores(vectors[rec["id"]], list(subject["topics"]), grounding)
            new = resolve_record(old, subject, scores)
            recs[i] = {**rec, "metadata": new}
            before_review += bool(old.get("needs_review"))
            after_review += bool(new.get("needs_review"))
            counts[new.get("tag_method")] = counts.get(new.get("tag_method"), 0) + 1
            if rec["id"] in vectors:
                updated.append((rec["id"], new))

            conf = new.get("topic_confidence")
            flags = []
            if new.get("co_topic_mismatch"):
                flags.append("MISMATCH")
            if new.get("needs_review"):
                flags.append("REVIEW")
            print(f"{rec['id']:24} {str(new.get('course_outcome')):4} {str(new.get('tag_method')):16} "
                  f"{('%.3f' % conf) if conf is not None else '  -  ':>6}  "
                  f"{describe_change(old, new)} {' '.join(flags)}")

    print(f"\nmethods: {counts}")
    print(f"needs_review: {before_review} before -> {after_review} after")
    print("Flagged questions: read their review_reason in the JSON, fix by hand if needed, "
          "and add \"tag_method\": \"manual\" so this script leaves them alone.")

    if not args.write:
        print("\nPreview only. Run again with --write to save.")
        return
    for path, recs in files.items():
        path.write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
    pyq_coll.update(ids=[i for i, _ in updated],
                    metadatas=[sanitize_metadata(m) for _, m in updated])
    print(f"\nSaved {len(files)} JSON file(s) and updated metadata of {len(updated)} vector(s) in {PYQ_COLLECTION}.")


if __name__ == "__main__":
    main()
