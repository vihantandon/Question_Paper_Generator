"""
Step 1 -- graph_lookup (Neo4j).

Moved here from prerequisite_graph/loader/integration.py. The logic is the
same; two changes:
  - the Neo4j driver is passed in (make_graph_lookup_node(driver)) instead
    of being opened when the file is imported;
  - it also returns current_topic {"id", "name", "semester"}, because the
    retrieve step needs the topic's name to search ChromaDB.

Reads:  current_topic_id, target_difficulty, target_marks
Writes: current_topic, allowed_prior_topics, blend_topic, topic_cos,
        prompt_instruction
"""
from pipeline.difficulty_blend import (
    build_question_prompt,
    get_prerequisites_with_distance,
    select_blend_topic,
)


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


def make_graph_lookup_node(driver):
    """Build the node function. It remembers `driver` (a closure), so
    LangGraph can call it with just the state."""

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
                    f"Topic '{topic_id}' not found in the graph. Has the loader been run? "
                    f"(docker compose --profile tools run --rm loader --reset)"
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
            "current_topic": {"id": topic_id, **topic},
            "allowed_prior_topics": prereqs,
            "blend_topic": blend,
            "topic_cos": cos,
            "prompt_instruction": prompt,
        }

    return graph_lookup_node
