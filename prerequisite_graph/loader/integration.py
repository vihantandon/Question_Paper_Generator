from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from neo4j import GraphDatabase


class GraphState(TypedDict):
    current_topic_id: str
    target_difficulty: str
    target_marks: int

    allowed_prior_topics: list
    blend_topic: dict | None

    relevant_chunks: list
    prompt_instruction: str


driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "yourpassword"))


def graph_lookup_node(state):

    topic_id = state["current_topic_id"]
    difficulty = state["target_difficulty"]

    with driver.session() as session:

        prereqs = get_prerequisites_with_distance(
            session,
            topic_id)

        topic_record = session.run(
            """
            MATCH (t:Topic {topic_id: $id})
            RETURN t.name AS name, t.semester AS semester
            """,
            id=topic_id
        ).single()

    topic = {
        "name": topic_record["name"],
        "semester": topic_record["semester"]
    }

    blend = select_blend_topic(
        prereqs,
        difficulty,
        topic["semester"]
    )

    prompt = build_question_prompt(
        topic,
        blend,
        difficulty,
        state["target_marks"]
    )

    return {
        "allowed_prior_topics": prereqs,
        "blend_topic": blend,
        "prompt_instruction": prompt
    }


builder = StateGraph(GraphState)

builder.add_node(
    "graph_lookup",
    graph_lookup_node
)

builder.add_edge(START, "graph_lookup")
builder.add_edge("graph_lookup", END)

graph = builder.compile()
