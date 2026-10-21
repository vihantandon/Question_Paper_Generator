"""
retrieval.py -- the payoff of topic_id: ask Neo4j WHAT to ask about,
then ask ChromaDB for the grounding text about exactly that.

Grounding text lives in TWO Chroma collections, and both are searched:
    book_content  -- book chunks      (Embedding/embed_chunks.py)
    tut_content   -- tutorial chunks  (Embedding/embed_tutorial_chunks.py)
Results from both are merged and sorted by distance (closest first), so a
topic with no book text (e.g. ALGO_05 Backtracking) still gets grounded in
its tutorial questions.

When the topic HAS tutorial questions, some places are kept for them
(min_tutorial): the course's own tutorial questions are pitched at exactly
the syllabus level, while the closest book text is often a research
handbook's advanced section. For "Heaps" (DS_03) the four closest chunks
were all from the DS handbook (amortized analysis, symmetric min-max heaps,
a reference list) and none of its 18 tutorial questions made it in. Each
kept tutorial question then brings the book text closest to IT, so the
facts needed to answer it are there too.

    from retrieval import get_grounding_collections, fetch_grounding_chunks
    collections = get_grounding_collections(chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT))
    chunks = fetch_grounding_chunks(collections, embed_fn,
                                    topic={"id": "ALGO_06", "name": "Dynamic Programming"},
                                    blend_topic={"id": "SDF1_04", "name": "Pointers"})

Past exam questions (pyq_bank) are fetched separately with
fetch_similar_pyqs(). They are STYLE examples -- how a real paper phrases a
question for N marks -- not facts to ground the answer in:

    pyqs = fetch_similar_pyqs(get_pyq_collection(client), embed_fn,
                              "Dynamic Programming", topic_id="ALGO_06", n=3)

`topic` / `blend_topic` come straight from the graph node (see pipeline/nodes/graph_lookup.py:
current_topic_id and state["blend_topic"], which already carries the id).

Always filter by topic_id, never by "subject": book/tutorial chunks use the
book's name ("APS", "SDF-1") while pyq_bank uses the graph's name ("ALGO",
"SDF1"). topic_id is the one key every collection shares.
"""

from typing import Callable, Optional

from settings import CHROMA_HOST, CHROMA_PORT      # .env / terminal / default, see settings.py

GROUNDING_COLLECTIONS = ("book_content", "tut_content")
TUTORIAL_COLLECTION = "tut_content"
PYQ_COLLECTION = "pyq_bank"


def get_grounding_collections(client, names=GROUNDING_COLLECTIONS):
    """Every grounding collection that exists on the server. A missing one
    (e.g. tutorials not embedded yet) is skipped with a warning, not a crash."""
    existing = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    out = []
    for name in names:
        if name in existing:
            out.append(client.get_collection(name))
        else:
            print(f"WARNING: Chroma collection '{name}' not found - skipping it. "
                  f"Run check_chroma.py to see what is missing.")
    return out


def get_pyq_collection(client, name=PYQ_COLLECTION):
    """The past-paper collection, or None (with a warning) if it isn't embedded yet."""
    existing = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    if name not in existing:
        print(f"WARNING: Chroma collection '{name}' not found - no PYQ style examples. "
              f"Run: python3 Embedding/embed_pyqs.py")
        return None
    return client.get_collection(name)


def _query(collection, query_embedding, topic_id, n, core_only=False):
    conditions = [{"topic_id": topic_id}, {"needs_review": False}]
    if core_only and collection.name != TUTORIAL_COLLECTION:
        conditions.append({"advanced": False})      # tutorial questions are all syllabus level
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        where={"$and": conditions},
        include=["documents", "metadatas", "distances"],
    )
    # "id" lets a later step say exactly which chunks a question was built from
    return [{"id": i,
             "text": d,
             "topic_id": m["topic_id"],
             "subject": m.get("subject", "?"),
             "type": m.get("type", collection.name),          # "book" | "tutorial"
             "collection": collection.name,
             "section": m.get("section_title", ""),
             "pages": (m.get("page_start", -1), m.get("page_end", -1)),  # (-1, -1) for tutorials
             "is_prereq_content": m.get("is_prereq_content", False),
             "distance": dist}
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                                     res["distances"][0])]


def _query_all(collections, embed_fn, text, topic_id, n, exclude_texts=(), core_only=False):
    """Search every collection, merge, keep the n closest DISTINCT texts.

    The same text can be stored more than once (a tutorial chunk has one row
    per topic_id; a book page can sit under several section titles), so
    copies are dropped. When that leaves fewer than n, more rows are fetched
    until n distinct texts are found or the collections run out.
    exclude_texts: texts to skip (e.g. ones already picked for another role).
    """
    if not isinstance(collections, (list, tuple)):
        collections = [collections]          # old callers passed one collection
    if n <= 0 or not collections:
        return []
    emb = embed_fn(text)
    fetch = n
    while True:
        hits, unread = [], []       # unread: last distance of each collection that may hold more
        for coll in collections:
            got = _query(coll, emb, topic_id, fetch, core_only)
            if len(got) == fetch:
                unread.append(got[-1]["distance"])
            hits += got
        hits.sort(key=lambda h: h["distance"])
        seen, out = set(exclude_texts), []
        for h in hits:
            if h["text"] not in seen:
                seen.add(h["text"])
                out.append(h)
        # Done when every collection is exhausted, or n distinct texts are in
        # hand and no collection could still hold a closer one.
        if not unread or (len(out) >= n and min(unread) >= out[n - 1]["distance"]):
            return out[:n]
        fetch *= 2


def closest_chunks(collections, query_embedding, topic_id, n):
    """The n closest DISTINCT grounding texts of a topic to a vector that is
    already in hand (resolve_pyq_labels.py scores each past question with the
    vector stored in pyq_bank, so it needs no embedding model)."""
    return _query_all(collections, lambda _text: query_embedding, "", topic_id, n)


def _closest_keeping_tutorials(collections, embed_fn, text, topic_id, n, min_tutorial,
                               exclude_texts=(), core_only=False):
    """n distinct texts about one topic, chosen in three steps:

      1. up to min_tutorial tutorial questions closest to `text` (the topic);
      2. for each of them, the book text closest to THAT QUESTION -- the
         explanation a student would read to answer it;
      3. whatever places are left: the closest text to `text`, from any collection.

    Step 2 exists because a tutorial question alone holds no facts. With only
    "Analyse the performance of the Binary Heap ... (a) array" to go on, the
    model wrote the parent/child index formulas from its own memory; searching
    the book with the question itself finds the section that explains them,
    where searching with the topic's long name found amortized analysis.

    Returned in reading order: step 3's text (closest first), then each
    explanation followed by its question. "found_by" says which step found
    a text: "topic" | "tutorial" | "explains <tutorial id>"."""
    if not isinstance(collections, (list, tuple)):
        collections = [collections]
    tutorials = [c for c in collections if getattr(c, "name", None) == TUTORIAL_COLLECTION]
    books = [c for c in collections if c not in tutorials]
    taken = set(exclude_texts)

    questions = _query_all(tutorials, embed_fn, text, topic_id, min(min_tutorial, n), taken, core_only)
    taken |= {q["text"] for q in questions}
    pairs = []                                   # [(explanation or None, question)]
    room = n - len(questions)
    for q in questions:
        found = _query_all(books, embed_fn, q["text"], topic_id, 1, taken, core_only) if room > 0 else []
        if found:
            room -= 1
            taken.add(found[0]["text"])
        pairs.append((dict(found[0], found_by=f"explains {q['id']}") if found else None,
                      dict(q, found_by="tutorial")))

    rest = [dict(h, found_by="topic")
            for h in _query_all(collections, embed_fn, text, topic_id, room, taken, core_only)]
    return rest + [hit for pair in pairs for hit in pair if hit is not None]


def fetch_grounding_chunks(collections, embed_fn: Callable[[str], list],
                           topic: dict, blend_topic: Optional[dict] = None,
                           n_main: int = 5, n_blend: int = 2,
                           min_tutorial: int = 2, min_tutorial_blend: int = 1,
                           core_only: bool = False, avoid_texts=()):
    """Grounding text for the main topic, plus (if the graph chose one) the
    blended prerequisite topic. Each result is labelled role='main' or 'blend'
    so the drafting prompt can tell the model which facts belong to which.

    min_tutorial / min_tutorial_blend: how many of the main / blend places
    are kept for tutorial questions, when the topic has them (see the top of
    this file); each such question also brings the book text that explains it
    (see _closest_keeping_tutorials). 0 = closest text only, from any collection.
    With the defaults: main = 2 tutorial questions + their 2 explanations +
    1 by topic name; blend = 1 tutorial question + its explanation.

    core_only: leave out book text from a book's advanced_chapters
    (topic_map.yaml; "advanced" in the metadata) -- for easy and medium
    questions. When a topic has NO core book text (or the vectors were
    tagged before this flag existed), all of its text is used after all.

    avoid_texts: main-topic texts earlier questions on this topic were built
    from (the question bank). Other text is taken first, so the next question
    is not the same problem again; the avoided texts only fill places that
    nothing else can.

    `collections` is a list from get_grounding_collections() (a single
    collection still works)."""
    def pick(text, topic_id, n, min_tut, exclude=()):
        if core_only:
            hits = _closest_keeping_tutorials(collections, embed_fn, text, topic_id, n, min_tut,
                                              exclude, core_only=True)
            if any(h["collection"] != TUTORIAL_COLLECTION for h in hits):   # some core book text
                return hits
        return _closest_keeping_tutorials(collections, embed_fn, text, topic_id, n, min_tut, exclude)

    main = pick(topic["name"], topic["id"], n_main, min_tutorial, tuple(avoid_texts))
    if avoid_texts and len(main) < n_main:          # not enough new text: fill up with the old
        main += pick(topic["name"], topic["id"], n_main - len(main), 0, {c["text"] for c in main})
    out = [dict(c, role="main") for c in main]
    if blend_topic:
        q = f'{blend_topic["name"]} (used within {topic["name"]})'
        # a chunk tagged with BOTH topics is already in as "main": don't send it twice
        main_texts = {c["text"] for c in out}
        out += [dict(c, role="blend")
                for c in pick(q, blend_topic["id"], n_blend, min_tutorial_blend, main_texts)]
    return out


def blend_fit(collections, embed_fn, topic: dict, candidate: dict, k: int = 3):
    """How well a prerequisite fits a topic: the average similarity (cosine,
    0..1) between the topic's name and the k closest texts about the
    prerequisite. None when the prerequisite has no text at all. "Graphs"
    text (shortest paths, DAGs) scores higher for "Dynamic Programming" than
    "Searching and Sorting" text (counting sort) does."""
    hits = _query_all(collections, embed_fn, topic["name"], candidate["id"], k)
    if not hits:
        return None
    return round(sum(1 - h["distance"] / 2 for h in hits) / len(hits), 4)


def _pyq_hits(collection, query_embedding, where, n):
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [{"id": i,
             "text": d,
             "topic_id": m.get("topic_id"),
             "subject": m.get("subject"),
             "marks": m.get("marks"),
             "bloom_level": m.get("bloom_level"),
             "course_outcome": m.get("course_outcome"),
             "year": m.get("year"),
             "exam": m.get("exam"),
             "distance": dist}
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                     res["metadatas"][0], res["distances"][0])]


MARKS_TOLERANCE = 1      # a past question worth target +- 1 marks counts as "the same size"
MARKS_POOL = 4           # with a mark target, look at 4x as many questions to choose from


def _marks_differ(hit, marks):
    """0 when the past question is worth about `marks`, else 1 (for sorting)."""
    m = hit.get("marks")
    is_number = isinstance(m, (int, float)) and not isinstance(m, bool)
    return 0 if is_number and abs(m - marks) <= MARKS_TOLERANCE else 1


def fetch_similar_pyqs(pyq_collection, embed_fn: Callable[[str], list], query_text: str,
                       topic_id: str, n: int = 3, exclude_ids=(), marks: Optional[int] = None):
    """Up to n past exam questions similar to query_text, as style examples.

    1. Same topic_id first (most useful: same topic, real exam wording).
    2. If that gives fewer than n, top up with the closest questions from the
       same subject. The subject is the topic_id prefix ("ALGO_06" -> "ALGO"),
       which is exactly how pyq_bank names subjects.
    Each result says which rule found it: match = "topic" | "subject".
    exclude_ids: PYQ ids to skip (e.g. ones already shown for this paper).
    marks: the marks of the question being written. Within each rule, past
       questions worth about as much (+- MARKS_TOLERANCE) come first, because
       marks set how long and deep a question is: a 2-mark example teaches
       the model little about writing a 6-mark question.
    """
    if pyq_collection is None:
        return []
    emb = embed_fn(query_text)
    subject = topic_id.split("_")[0]
    taken = set(exclude_ids)
    out = []
    for where, match in (({"topic_id": topic_id}, "topic"), ({"subject": subject}, "subject")):
        if len(out) >= n:
            break
        # ask for extra results because some may be skipped as already taken
        want = n + len(taken)
        hits = _pyq_hits(pyq_collection, emb, where, want if marks is None else want * MARKS_POOL)
        if marks is not None:
            hits.sort(key=lambda h: _marks_differ(h, marks))     # stable: keeps the similarity order
        for hit in hits:
            if hit["id"] not in taken:
                taken.add(hit["id"])
                out.append(dict(hit, match=match))
                if len(out) >= n:
                    break
    return out


if __name__ == "__main__":
    import chromadb
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    colls = get_grounding_collections(client)
    embed = lambda t: model.encode(t).tolist()

    # ALGO_05 has no book text at all, so this proves tutorials are used.
    for topic, blend in [({"id": "ALGO_06", "name": "Dynamic Programming"},
                          {"id": "SDF1_04", "name": "Pointers"}),
                         ({"id": "ALGO_05", "name": "Backtracking Algorithms"}, None)]:
        print(f"\n=== {topic['id']} {topic['name']} ===")
        for c in fetch_grounding_chunks(colls, embed, topic, blend):
            print(c["role"], c["type"], c["topic_id"], c["subject"], c["section"],
                  c["pages"], round(c["distance"], 3))
            print("   ", c["text"][:160].replace("\n", " "))

        print(f"--- similar past questions ({topic['id']}) ---")
        for q in fetch_similar_pyqs(get_pyq_collection(client), embed, topic["name"], topic["id"]):
            print(q["match"], q["id"], f"{q['marks']} marks", q["bloom_level"], round(q["distance"], 3))
            print("   ", q["text"][:160].replace("\n", " "))
