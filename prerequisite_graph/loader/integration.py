"""
LangGraph integration for the graph-lookup node.

Fixes applied:
  - imports the helper functions from difficulty_blend (previously
    called but never imported -> NameError on first run)
  - credentials read from env vars instead of being hardcoded
  - driver is closed properly instead of leaking
  - CO lookup added, so the node returns the topic's COs alongside the
    prerequisite blend (the paper-level CO constraint check needs this)
"""

import atexit
import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from neo4j import GraphDatabase

from difficulty_blend import (
    get_prerequisites_with_distance,
    select_blend_topic,
    build_question_prompt,
)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "yourpassword")


class GraphState(TypedDict):
    current_topic_id: str
    target_difficulty: str
    target_marks: int

    allowed_prior_topics: list
    blend_topic: dict | None
    topic_cos: list

    relevant_chunks: list
    prompt_instruction: str


driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
atexit.register(driver.close)


def get_topic_cos(session, topic_id):
    """Return the COs this topic contributes to."""
    result = session.run(
        """
        MATCH (t:Topic {topic_id: $topic_id})-[:CONTRIBUTES_TO]->(c:CO)
        RETURN c.co_id AS co_id, c.short_id AS short_id, c.description AS description
        ORDER BY c.short_id
        """,
        topic_id=topic_id,
    )
    return [dict(r) for r in result]


def graph_lookup_node(state):
    topic_id = state["current_topic_id"]
    difficulty = state["target_difficulty"]

    with driver.session() as session:
        prereqs = get_prerequisites_with_distance(session, topic_id)

        topic_record = session.run(
            """
            MATCH (t:Topic {topic_id: $id})
            RETURN t.name AS name, t.semester AS semester
            """,
            id=topic_id,
        ).single()

        if topic_record is None:
            raise ValueError(
                f"Topic '{topic_id}' not found in the graph. "
                f"Has the loader been run? (docker compose --profile tools run --rm loader)"
            )

        cos = get_topic_cos(session, topic_id)

    topic = {
        "name": topic_record["name"],
        "semester": topic_record["semester"],
    }

    blend = select_blend_topic(prereqs, difficulty, topic["semester"])

    prompt = build_question_prompt(
        topic,
        blend,
        difficulty,
        state["target_marks"],
    )

    return {
        "allowed_prior_topics": prereqs,
        "blend_topic": blend,
        "topic_cos": cos,
        "prompt_instruction": prompt,
    }


builder = StateGraph(GraphState)
builder.add_node("graph_lookup", graph_lookup_node)
builder.add_edge(START, "graph_lookup")
builder.add_edge("graph_lookup", END)

graph = builder.compile()


if __name__ == "__main__":
    # Quick manual check against a live Neo4j instance
    result = graph.invoke({
        "current_topic_id": "ALGO_06",      # Dynamic Programming
        "target_difficulty": "hard",
        "target_marks": 10,
    })
    print("Blend topic:", result["blend_topic"])
    print("COs:", result["topic_cos"])
    print("\nPrompt instruction:\n", result["prompt_instruction"])