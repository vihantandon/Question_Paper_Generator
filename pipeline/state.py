"""
GraphState: the one dictionary that flows through the pipeline.

Each node reads some keys and returns ONLY the keys it wants to add or
change; LangGraph merges that into the state for the next node. So this
file is also a map of which step produces what.

total=False means no key is required up front: you start the graph with
just the three input keys, and the nodes fill in the rest.
"""
from typing import Optional, TypedDict


class GraphState(TypedDict, total=False):
    # --- input: what to write (later this comes from the paper planner) ---
    current_topic_id: str          # e.g. "ALGO_06"
    target_difficulty: str         # "easy" | "medium" | "hard"
    target_marks: int

    # --- filled by graph_lookup (Neo4j) ---
    current_topic: dict            # {"id", "name", "semester"}
    allowed_prior_topics: list     # every prerequisite, with its hop distance
    blend_topic: Optional[dict]    # the prerequisite to blend in, or None
    topic_cos: list                # COs this topic contributes to
    prompt_instruction: str        # instruction text for the LLM drafting step

    # --- filled by retrieve (ChromaDB) ---
    relevant_chunks: list          # grounding text, each tagged role "main" | "blend"
    style_examples: list           # similar past exam questions (PYQs)
    grounding_warning: Optional[str]   # set when there is nothing to ground in
