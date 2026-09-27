"""
retrieval.py -- the payoff of topic_id: ask Neo4j WHAT to ask about,
then ask ChromaDB for the book text about exactly that.
 
    from retrieval import fetch_grounding_chunks
    chunks = fetch_grounding_chunks(collection, embed_fn,
                                    topic={"id": "ALGO_06", "name": "Dynamic Programming"},
                                    blend_topic={"id": "SDF1_04", "name": "Pointers"})
 
`topic` / `blend_topic` come straight from the graph node (see integration.py:
current_topic_id and state["blend_topic"], which already carries the id).
"""

from typing import Callable, Optional

CHROMA_HOST , CHROMA_PORT , COLLECTION = "localhost" , 8080 , "book_content"

def _query(collection , embed_fn , text , topic_id , n):
    res = collection.query(
        query_embeddings = [embed_fn(text)],
        n_results=n,
        where = {"$and": [{"topic_id": topic_id}, {"needs_review": False}]},
        include = ["documents" , "metadatas", "distances"],
    )
    return [{"text": d, "topic_id": m["topic_id"], "subject": m["subject"],
             "section": m["section_title"], "pages": (m["page_start"], m["page_end"]),
             "is_prereq_content": m["is_prereq_content"], "distance": dist}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

def fetch_grounding_chunks(collection, embed_fn: Callable[[str], list],
                           topic: dict, blend_topic: Optional[dict] = None,
                           n_main: int = 4, n_blend: int = 2):
    """Book text for the main topic, plus (if the graph chose one) the blended
    prerequisite topic. Each result is labelled role='main' or 'blend' so the
    drafting prompt can tell the model which facts belong to which."""
    out = [dict(c, role="main") for c in _query(collection, embed_fn, topic["name"], topic["id"], n_main)]
    if blend_topic:
        q = f'{blend_topic["name"]} (used within {topic["name"]})'
        out += [dict(c, role="blend") for c in _query(collection, embed_fn, q, blend_topic["id"], n_blend)]
    return out

if __name__ == "__main__":
    import chromadb
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    coll = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT).get_or_create_collection(COLLECTION)
    embed = lambda t: model.encode(t).tolist()

    for c in fetch_grounding_chunks(coll, embed,
                                    {"id": "ALGO_06", "name": "Dynamic Programming"},
                                    {"id": "SDF1_04", "name": "Pointers"}):
        print(c["role"], c["topic_id"], c["subject"], c["section"], c["pages"])
        print("   ", c["text"].replace("\n", " "))
