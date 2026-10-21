"""
Step 2 -- retrieve (ChromaDB).

Fills the two things the LLM drafting step will need:
  relevant_chunks  facts to ground the question in (book + tutorial text),
                   tagged role "main" (the topic) or "blend" (the prerequisite)
  style_examples   similar past exam questions, to copy the exam's style

and sets grounding_warning when there is nothing to ground in (for example
SDF1_01 has no book or tutorial text). A later step can then skip that
topic instead of letting the LLM invent facts.

With a question bank (pipeline/dedup.py), the questions already made for
this topic are looked up too: the writer is shown them (earlier_questions,
"write a different problem"), and the main-topic text they were built from is
avoided, so the next question gets other material. Without it, every ALGO_06
run got the same 7 sources, three of them about rod cutting, and every draft
was rod cutting again -- which the duplicate check then rejected.

Which prerequisite to blend: graph_lookup lists the candidates the graph
allows (blend_candidates); when there is more than one, the one whose study
material is closest to the main topic is used (retrieval.blend_fit), and the
instruction to the writer names it. Hops alone made Dynamic Programming blend
Searching and Sorting, and every draft invented a forced link.

Reads:  current_topic, blend_topic, blend_candidates, target_difficulty (easy / medium: no
        advanced-chapter text), target_marks (past questions of about
        that many marks are preferred as style examples)
Writes: relevant_chunks, style_examples, grounding_warning, earlier_questions,
        blend_topic + prompt_instruction (when a better-fitting prerequisite is chosen),
        blend_scores
"""
from pipeline.dedup import earlier_questions
from pipeline.difficulty_blend import build_question_prompt
from retrieval import blend_fit, fetch_grounding_chunks, fetch_similar_pyqs


def choose_blend(collections, embed_fn, topic, candidates, default):
    """(the best-fitting candidate, [(id, fit)]). Candidates with no text never
    win; a tie keeps the graph's order. With 0 or 1 candidates: the default."""
    if len(candidates) < 2:
        return default, []
    scores = [(c, blend_fit(collections, embed_fn, topic, c)) for c in candidates]
    usable = [(c, s) for c, s in scores if s is not None]
    best = max(usable, key=lambda cs: cs[1])[0] if usable else default
    return best, [(c["id"], s) for c, s in scores]


def texts_of(collections, ids):
    """The stored text of these chunk ids (the ones still in the collections)."""
    out = set()
    for c in collections:
        if ids and callable(getattr(c, "get", None)):
            out |= {t for t in c.get(ids=list(ids), include=["documents"])["documents"] if t}
    return out


def make_retrieve_node(grounding_collections, pyq_collection, embed_fn,
                       n_main=5, n_blend=2, n_examples=3, bank_collection=None):

    def retrieve_node(state):
        topic = state["current_topic"]
        blend = state.get("blend_topic")
        chosen = {}
        best, scores = choose_blend(grounding_collections, embed_fn, topic,
                                    state.get("blend_candidates") or [], blend)
        if scores:
            chosen["blend_scores"] = scores
        if best is not None and (blend is None or best["id"] != blend["id"]):
            blend = best
            chosen["blend_topic"] = best
            chosen["prompt_instruction"] = build_question_prompt(
                topic, best, state.get("target_difficulty"), state.get("target_marks"))

        # easy / medium: syllabus-level text only (topic_map.yaml advanced_chapters)
        core_only = state.get("target_difficulty") in ("easy", "medium")
        earlier = earlier_questions(bank_collection, topic["id"])
        used = texts_of(grounding_collections, {i for q in earlier for i in q["source_ids"]})
        chunks = fetch_grounding_chunks(grounding_collections, embed_fn, topic, blend,
                                        n_main=n_main, n_blend=n_blend, core_only=core_only,
                                        avoid_texts=used)
        examples = fetch_similar_pyqs(pyq_collection, embed_fn, topic["name"], topic["id"],
                                      n=n_examples, marks=state.get("target_marks"))

        warning = None
        if not any(c["role"] == "main" for c in chunks):
            warning = (f"No grounding text for {topic['id']} ({topic['name']}): a question on it "
                       f"would not be grounded. Add study material for it, or skip this topic.")
        elif blend and not any(c["role"] == "blend" for c in chunks):
            warning = (f"No grounding text for the blend topic {blend['id']} ({blend['name']}): "
                       f"the blended part of the question would not be grounded.")

        return {
            **chosen,
            "relevant_chunks": chunks,
            "style_examples": examples,
            "grounding_warning": warning,
            "earlier_questions": earlier,
        }

    return retrieve_node
