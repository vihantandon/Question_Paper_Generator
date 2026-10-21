"""
pipeline/dedup.py -- is the new question too close to one that exists?

Part (c) of the project: no duplicate or near-duplicate questions across
successive papers. After a draft passes the quality check, it is compared with
  * past papers     pyq_bank, the same subject
  * tutorial sheets tut_content (an exam should not copy a tutorial question)
  * earlier output  generated_questions: every question this system accepted
                    and saved (the "question bank"; see save_question)

TWO WAYS TO BE TOO CLOSE (either one is enough)
  meaning   the embedding similarity (all-MiniLM-L6-v2, cosine) is at least
            DEDUP_SIMILARITY (.env, default 0.78). This catches the same
            question in other words, or with new numbers. Measured on 21 Oct
            (--calibrate): different real questions reach at most 0.72 (past
            paper to past paper, and past paper to tutorial), while two
            generated rod-cutting-with-cut-cost questions -- the same problem
            with new numbers -- were 0.81. 0.78 sits between.
  wording   at least DEDUP_OVERLAP (default 0.5) of the new question's 3-word
            phrases also appear in the old one: copied text. Embeddings only
            read the first 256 tokens, so long copied code can slip past them.

A duplicate goes back to draft with the question it is too close to, and the
instruction to write a different problem -- not the same one with new
numbers. That counts as one of the MAX_DRAFTS drafts.

The thresholds are a guess until measured on your data:
    python3 -m pipeline.dedup --calibrate
shows how similar the real PYQs already are to each other and to the tutorial
questions (they are different questions, so the threshold should sit above
nearly all of them). It reads the vectors stored in Chroma: no embedding model.

The question bank:
    python3 -m pipeline.dedup --list [--subject DS]
    python3 -m pipeline.dedup --remove ID [ID ...]
    python3 -m pipeline.dedup --clear          # forget every saved question

Reads:  draft, current_topic, draft_attempts
Writes: dedup (the report), dedup_passed, revision_feedback (on a duplicate),
        dedup_reports (every report so far, for this question)
"""
import argparse
import json
import re
from datetime import datetime, timezone

BANK_COLLECTION = "generated_questions"
PYQ_COLLECTION = "pyq_bank"
TUTORIAL_COLLECTION = "tut_content"
DEFAULT_SIMILARITY = 0.78
DEFAULT_OVERLAP = 0.5
NEAREST = 3            # closest questions looked at in each place


# ---------------------------------------------------------------- the measures
def words(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def trigrams(text):
    w = words(text)
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def overlap(new, old):
    """Share of the new text's 3-word phrases that the old text also has."""
    mine = trigrams(new)
    return len(mine & trigrams(old)) / len(mine) if mine else 0.0


def similarity(distance):
    """Chroma's squared L2 distance between unit vectors -> cosine similarity."""
    return 1.0 - float(distance) / 2.0


def subject_of(topic_id):
    """"ALGO_06" -> "ALGO": how pyq_bank and the question bank name subjects."""
    return str(topic_id).split("_")[0]


# ---------------------------------------------------------------- looking
def places(pyq_collection=None, tutorial_collection=None, bank_collection=None):
    """[(name, collection, where(subject))] for the places that exist."""
    out = []
    if pyq_collection is not None:
        out.append(("past paper", pyq_collection,
                    lambda s: {"$and": [{"subject": s}, {"type": "question"}]}))
    if tutorial_collection is not None:
        # tutorial rows are stored once per topic they cover, under the
        # tutorial's own name (APS, not ALGO): look at all of them
        out.append(("tutorial", tutorial_collection, lambda s: None))
    if bank_collection is not None:
        out.append(("earlier question", bank_collection, lambda s: {"subject": s}))
    return out


def _is_empty(collection):
    count = getattr(collection, "count", None)
    return callable(count) and count() == 0


def nearest(collection, vector, where, n=NEAREST):
    """The n closest stored questions, the same text only once:
    [{"id", "text", "similarity"}], closest first."""
    if _is_empty(collection):
        return []
    kwargs = {"query_embeddings": [vector], "n_results": n * 2,
              "include": ["documents", "metadatas", "distances"]}
    if where:
        kwargs["where"] = where
    got = collection.query(**kwargs)
    out, seen = [], set()
    for id_, text, dist in zip(got["ids"][0], got["documents"][0], got["distances"][0]):
        key = " ".join(words(text))
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": id_, "text": text or "", "similarity": round(similarity(dist), 4)})
        if len(out) >= n:
            break
    return out


def check_duplicates(text, subject, embed_fn, where_to_look, *,
                     similarity_limit=DEFAULT_SIMILARITY, overlap_limit=DEFAULT_OVERLAP):
    """Compare `text` with the closest questions in each place.
    Returns {"passed", "duplicates": [...], "closest": [...], "looked_in": [...],
             "similarity_limit", "overlap_limit"}; each match is
    {"place", "id", "text", "similarity", "overlap", "why"}."""
    vector = embed_fn(text)
    closest, duplicates, looked = [], [], []
    for place, collection, where in where_to_look:
        looked.append(place)
        hits = nearest(collection, vector, where(subject))
        for hit in hits:
            hit.update(place=place, overlap=round(overlap(text, hit["text"]), 4))
            why = []
            if hit["similarity"] >= similarity_limit:
                why.append(f"meaning {hit['similarity']:.2f}")
            if hit["overlap"] >= overlap_limit:
                why.append(f"{hit['overlap']:.0%} of the wording")
            if why:
                duplicates.append(dict(hit, why=" and ".join(why)))
        if hits:
            closest.append(max(hits, key=lambda h: (h["similarity"], h["overlap"])))
    return {"passed": not duplicates, "duplicates": duplicates, "closest": closest, "looked_in": looked,
            "similarity_limit": similarity_limit, "overlap_limit": overlap_limit}


def feedback(report):
    """What the writer is told about a duplicate."""
    parts = []
    for d in report["duplicates"]:
        snippet = " ".join(d["text"].split())[:200]
        parts.append(f'a {d["place"]} question ({d["id"]}; same {d["why"]}): "{snippet}"')
    return ("The question is too close to " + "; and to ".join(parts) + ". Write a different problem: "
            "a different task or situation that tests the same topic -- not the same question in other "
            "words or with new numbers.")


# ---------------------------------------------------------------- the node
def make_dedup_node(embed_fn, pyq_collection=None, tutorial_collection=None, bank_collection=None,
                    similarity_limit=DEFAULT_SIMILARITY, overlap_limit=DEFAULT_OVERLAP):
    from pipeline.nodes.quality_check import draft_as_text
    where_to_look = places(pyq_collection, tutorial_collection, bank_collection)

    def dedup_node(state):
        draft = state.get("draft")
        if not draft:
            return {"dedup": None, "dedup_passed": False}
        report = check_duplicates(draft_as_text(draft), subject_of(state["current_topic"]["id"]), embed_fn,
                                  where_to_look, similarity_limit=similarity_limit, overlap_limit=overlap_limit)
        report["check"] = len(state.get("quality_reports") or [])     # the check this draft passed
        out = {"dedup": report, "dedup_passed": report["passed"],
               "dedup_reports": (state.get("dedup_reports") or []) + [report]}
        if not report["passed"]:
            out["revision_feedback"] = feedback(report)
        return out

    return dedup_node


# ---------------------------------------------------------------- the question bank
def question_record(result, now=None):
    """(id, text, metadata) to store an accepted question in the bank."""
    from pipeline.nodes.quality_check import draft_as_text
    draft, topic = result["draft"], result["current_topic"]
    labels = result.get("classification") or {}
    when = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S")
    # every main-topic source the writer was given: the next question on this
    # topic gets other ones (retrieve), so it is not the same problem again
    given = [c["id"] for c in result.get("relevant_chunks") or [] if c.get("role") == "main" and c.get("id")]
    meta = {"subject": subject_of(topic["id"]), "topic_id": topic["id"], "marks": draft["marks"],
            "source_ids": json.dumps(given),
            "difficulty": result.get("target_difficulty") or "", "co": labels.get("co") or "",
            "bloom": labels.get("bloom") or "", "model": draft.get("model") or "", "created": when,
            "type": "generated", "answer_key": json.dumps(draft["answer_outline"], ensure_ascii=False)}
    return f"{topic['id']}_{when}", draft_as_text(draft), meta


def save_question(bank_collection, embed_fn, result, now=None):
    """Put an accepted question in the bank; later questions are checked against it."""
    id_, text, meta = question_record(result, now)
    bank_collection.upsert(ids=[id_], documents=[text], metadatas=[meta], embeddings=[embed_fn(text)])
    return id_


def earlier_questions(bank_collection, topic_id, limit=5):
    """The questions already in the bank for this topic, newest first:
    [{"id", "text", "source_ids"}]. The writer is shown them, and their
    sources are avoided."""
    if bank_collection is None or _is_empty(bank_collection):
        return []
    got = bank_collection.get(where={"topic_id": topic_id}, include=["documents", "metadatas"])
    rows = sorted(zip(got["ids"], got["documents"], got["metadatas"]),
                  key=lambda r: r[2].get("created", ""), reverse=True)
    return [{"id": i, "text": t or "", "source_ids": json.loads(m.get("source_ids") or "[]")}
            for i, t, m in rows[:limit]]


def get_bank(client):
    return client.get_or_create_collection(BANK_COLLECTION)


# ---------------------------------------------------------------- calibration
def calibrate(pyq_collection, tutorial_collection, limits=(0.80, 0.85, 0.88, 0.90, 0.95), top=8):
    """How similar are the real PYQs to each other and to tutorial questions?
    Uses the vectors stored in Chroma. Returns the pairs and prints a report."""
    got = pyq_collection.get(where={"type": "question"}, include=["embeddings", "documents", "metadatas"])
    pyq_pairs, tut_pairs, seen = [], [], set()
    for id_, vec, text, meta in zip(got["ids"], got["embeddings"], got["documents"], got["metadatas"]):
        vec = [float(x) for x in vec]
        for hit in nearest(pyq_collection, vec, {"$and": [{"subject": meta.get("subject")},
                                                          {"type": "question"}]}, n=2):
            pair = tuple(sorted((id_, hit["id"])))
            if hit["id"] != id_ and pair not in seen:
                seen.add(pair)
                pyq_pairs.append((hit["similarity"], overlap(text, hit["text"]), id_, hit["id"], text, hit["text"]))
                break
        if tutorial_collection is not None:
            for hit in nearest(tutorial_collection, vec, None, n=1):
                tut_pairs.append((hit["similarity"], overlap(text, hit["text"]), id_, hit["id"], text, hit["text"]))
    for name, pairs in (("past paper <-> its closest other past paper", pyq_pairs),
                        ("past paper <-> its closest tutorial question", tut_pairs)):
        pairs.sort(reverse=True)
        print(f"\n{name} ({len(pairs)} pairs)")
        print("  similarity at least: " + "   ".join(
            f"{t:.2f}: {sum(p[0] >= t for p in pairs):2d}" for t in limits))
        print(f"  wording overlap at least {DEFAULT_OVERLAP:.0%}: {sum(p[1] >= DEFAULT_OVERLAP for p in pairs)}")
        for sim, ov, a, b, ta, tb in pairs[:top]:
            print(f"  {sim:.3f}  words {ov:4.0%}  {a}  <->  {b}")
            print(f"           \"{' '.join(ta.split())[:90]}\"")
            print(f"           \"{' '.join(tb.split())[:90]}\"")
    return pyq_pairs, tut_pairs


# ---------------------------------------------------------------- the command line
def main(argv=None):
    ap = argparse.ArgumentParser(description="Duplicate check: calibrate it, or look after the question bank")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--calibrate", action="store_true",
                       help="how similar are the real PYQs to each other and to tutorial questions?")
    group.add_argument("--list", action="store_true", help="the questions saved in the bank")
    group.add_argument("--remove", nargs="+", metavar="ID", help="take these questions out of the bank")
    group.add_argument("--clear", action="store_true", help="forget every saved question")
    ap.add_argument("--subject", help="--list: only this subject")
    args = ap.parse_args(argv)

    import chromadb
    from settings import CHROMA_HOST, CHROMA_PORT
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    names = {c if isinstance(c, str) else c.name for c in client.list_collections()}

    if args.calibrate:
        if PYQ_COLLECTION not in names:
            raise SystemExit("pyq_bank is not in Chroma: run python3 Embedding/embed_pyqs.py first")
        calibrate(client.get_collection(PYQ_COLLECTION),
                  client.get_collection(TUTORIAL_COLLECTION) if TUTORIAL_COLLECTION in names else None)
        print("\nThese are different questions, so DEDUP_SIMILARITY (.env) should sit above nearly all of them.")
        return 0

    bank = get_bank(client)
    if args.list:
        got = bank.get(where={"subject": args.subject} if args.subject else None,
                       include=["documents", "metadatas"])
        rows = sorted(zip(got["ids"], got["documents"], got["metadatas"]), key=lambda r: r[2].get("created", ""))
        for id_, text, meta in rows:
            print(f"{id_:28} {meta.get('marks')} marks  {meta.get('difficulty'):6} {meta.get('co'):4} "
                  f"{meta.get('bloom'):10} \"{' '.join(text.split())[:70]}\"")
        print(f"{len(rows)} question(s) in the bank")
    elif args.remove:
        bank.delete(ids=args.remove)
        print(f"removed {len(args.remove)} question(s); {bank.count()} left")
    elif args.clear:
        client.delete_collection(BANK_COLLECTION)
        print("the question bank is empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
