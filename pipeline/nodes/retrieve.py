"""
Step 2 -- retrieve (ChromaDB).

Fills the two things the LLM drafting step will need:
  relevant_chunks  facts to ground the question in (book + tutorial text),
                   tagged role "main" (the topic) or "blend" (the prerequisite)
  style_examples   similar past exam questions, to copy the exam's style

and sets grounding_warning when there is nothing to ground in (for example
SDF1_01 has no book or tutorial text). A later step can then skip that
topic instead of letting the LLM invent facts.

Reads:  current_topic, blend_topic
Writes: relevant_chunks, style_examples, grounding_warning
"""
from retrieval import fetch_grounding_chunks, fetch_similar_pyqs


def make_retrieve_node(grounding_collections, pyq_collection, embed_fn,
                       n_main=4, n_blend=2, n_examples=3):

    def retrieve_node(state):
        topic = state["current_topic"]
        blend = state.get("blend_topic")

        chunks = fetch_grounding_chunks(grounding_collections, embed_fn, topic, blend,
                                        n_main=n_main, n_blend=n_blend)
        examples = fetch_similar_pyqs(pyq_collection, embed_fn, topic["name"], topic["id"],
                                      n=n_examples)

        warning = None
        if not any(c["role"] == "main" for c in chunks):
            warning = (f"No grounding text for {topic['id']} ({topic['name']}): a question on it "
                       f"would not be grounded. Add study material for it, or skip this topic.")
        elif blend and not any(c["role"] == "blend" for c in chunks):
            warning = (f"No grounding text for the blend topic {blend['id']} ({blend['name']}): "
                       f"the blended part of the question would not be grounded.")

        return {
            "relevant_chunks": chunks,
            "style_examples": examples,
            "grounding_warning": warning,
        }

    return retrieve_node
