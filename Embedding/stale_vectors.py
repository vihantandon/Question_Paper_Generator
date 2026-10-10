"""
stale_vectors.py -- shared by the embed scripts: after upserting, delete the
vectors whose chunk no longer exists in the chunk files.

WHY
    upsert only adds or overwrites. Chunk ids are positional ("DS_035_01",
    "DS_tut_chunk_007__DS_02"), so after re-extracting or re-tagging, some
    old ids simply stop existing in the files. Their vectors stayed in
    Chroma, still tagged with their old topic_id, and retrieval kept
    serving them as grounding text.

A full run makes the collection match the files exactly (the same thing
check_chroma.py verifies). A partial run (embed_tutorial_chunks.py
--subject APS) passes where={"subject": "APS"} so only that subject's
rows are cleaned.
"""
PAGE = 1000


def stored_ids(collection, where=None):
    """Every id in the collection (or only the rows matching `where`), paged
    so a big collection doesn't time out."""
    ids, offset = [], 0
    while True:
        part = collection.get(where=where, include=[], limit=PAGE, offset=offset)
        if not part["ids"]:
            break
        ids += part["ids"]
        offset += len(part["ids"])
    return ids


def remove_stale_vectors(collection, keep_ids, where=None):
    """Delete every vector (matching `where`) whose id is not in keep_ids.
    Returns the deleted ids, sorted.

    An empty keep_ids deletes nothing: "no chunks found" is far more likely
    a missing file than a wish to empty the collection.
    """
    keep_ids = set(keep_ids)
    if not keep_ids:
        return []
    stale = sorted(i for i in stored_ids(collection, where) if i not in keep_ids)
    for start in range(0, len(stale), PAGE):
        collection.delete(ids=stale[start:start + PAGE])
    return stale
