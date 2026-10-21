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
    blend_candidates: list         # the prerequisites it could have been (retrieve picks the best fit)
    blend_scores: list             # [(topic id, fit)] -- how close each one's material is to the topic
    topic_cos: list                # COs this topic contributes to
    prompt_instruction: str        # instruction text for the LLM drafting step

    # --- filled by retrieve (ChromaDB) ---
    relevant_chunks: list          # grounding text, each tagged role "main" | "blend"
    style_examples: list           # similar past exam questions (PYQs)
    grounding_warning: Optional[str]   # set when there is nothing to ground in
    earlier_questions: list        # questions already in the bank for this topic (pipeline/dedup.py)

    # --- filled by draft (the language model) ---
    draft: Optional[dict]          # {"question", "parts", "marks", "answer_outline",
                                   #  "sources_used", "source_ids", "blend_used", "model"}
    draft_error: Optional[str]     # why there is no draft (no grounding text, or the model gave up)
    draft_attempts: int            # how many times draft has run for this question
    drafts: list                   # every draft made for this question, oldest first

    # --- filled by quality_check, read by draft ---
    revision_feedback: Optional[str]   # why the last draft was rejected; the next draft fixes it
    quality: Optional[dict]        # the last check: claims (with quotes), answerable,
                                   # single_reading, marks_fit, blend_needed, problems, passed
    quality_passed: bool           # the current draft passed the check
    quality_reports: list          # every check made for this question, oldest first

    # --- filled by dedup (pipeline/dedup.py) ---
    dedup: Optional[dict]          # the last duplicate check: passed, duplicates, closest
    dedup_passed: bool             # the current draft is not too close to an existing question
    dedup_reports: list            # every duplicate check made for this question

    # --- filled by classify ---
    classification: Optional[dict] # {"co", "co_text", "co_level", "bloom", "difficulty", the reasons,
                                   #  "notes", "model"}, or {"error"}; None when there was no draft
