"""
retrieval.py -- the payoff of topic_id: ask Neo4j WHAT to ask about,
then ask ChromaDB for the grounding text about exactly that.

Grounding text lives in TWO Chroma collections, and both are searched:
    book_content  -- book chunks      (Embedding/embed_chunks.py)
    tut_content   -- tutorial chunks  (Embedding/embed_tutorial_chunks.py)
Results from both are merged and sorted by distance (closest first), so a
topic with no book text (e.g. ALGO_05 Backtracking) still gets grounded in
its tutorial questions.

    from retrieval import get_grounding_collections, fetch_grounding_chunks
    collections = get_grounding_collections(chromadb.HttpClient(host="localhost", port=8000))
    chunks = fetch_grounding_chunks(collections, embed_fn,
                                    topic={"id": "ALGO_06", "name": "Dynamic Programming"},
                                    blend_topic={"id": "SDF1_04", "name": "Pointers"})

`topic` / `blend_topic` come straight from the graph node (see integration.py:
current_topic_id and state["blend_topic"], which already carries the id).

Always filter by topic_id, never by "subject": book/tutorial chunks use the
book's name ("APS", "SDF-1") while pyq_bank uses the graph's name ("ALGO",
"SDF1"). topic_id is the one key every collection shares.
"""

import os
from typing import Callable, Optional

CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))   # docker-compose exposes 8000
GROUNDING_COLLECTIONS = ("book_content", "tut_content")


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


def _query(collection, query_embedding, topic_id, n):
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        where={"$and": [{"topic_id": topic_id}, {"needs_review": False}]},
        include=["documents", "metadatas", "distances"],
    )
    return [{"text": d,
             "topic_id": m["topic_id"],
             "subject": m.get("subject", "?"),
             "type": m.get("type", collection.name),          # "book" | "tutorial"
             "collection": collection.name,
             "section": m.get("section_title", ""),
             "pages": (m.get("page_start", -1), m.get("page_end", -1)),  # (-1, -1) for tutorials
             "is_prereq_content": m.get("is_prereq_content", False),
             "distance": dist}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]


def _query_all(collections, embed_fn, text, topic_id, n):
    """Search every collection, merge, keep the n closest overall. Tutorial
    rows share text across topic_ids, so duplicate texts are dropped."""
    if not isinstance(collections, (list, tuple)):
        collections = [collections]          # old callers passed one collection
    emb = embed_fn(text)
    hits = []
    for coll in collections:
        hits += _query(coll, emb, topic_id, n)
    hits.sort(key=lambda h: h["distance"])
    seen, out = set(), []
    for h in hits:
        if h["text"] not in seen:
            seen.add(h["text"])
            out.append(h)
    return out[:n]


def fetch_grounding_chunks(collections, embed_fn: Callable[[str], list],
                           topic: dict, blend_topic: Optional[dict] = None,
                           n_main: int = 4, n_blend: int = 2):
    """Grounding text for the main topic, plus (if the graph chose one) the
    blended prerequisite topic. Each result is labelled role='main' or 'blend'
    so the drafting prompt can tell the model which facts belong to which.

    `collections` is a list from get_grounding_collections() (a single
    collection still works)."""
    out = [dict(c, role="main")
           for c in _query_all(collections, embed_fn, topic["name"], topic["id"], n_main)]
    if blend_topic:
        q = f'{blend_topic["name"]} (used within {topic["name"]})'
        out += [dict(c, role="blend")
                for c in _query_all(collections, embed_fn, q, blend_topic["id"], n_blend)]
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
