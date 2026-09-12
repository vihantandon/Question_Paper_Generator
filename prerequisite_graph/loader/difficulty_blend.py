"""
Difficulty-aware topic blending.

Given a target topic and a desired difficulty, picks which prerequisite
topic(s) (if any) should be blended into the LLM prompt to create a
"tricky" cross-topic question, based on how many hops back that
prerequisite sits in the graph.
"""

from neo4j import GraphDatabase


def get_prerequisites_with_distance(session, topic_id, max_depth=4):
    """
    Returns every prerequisite of topic_id, tagged with its MINIMUM hop
    distance (not just "reachable within N hops" -- the actual shortest
    distance), plus which semester it's from.

    hop=1  -> direct prerequisite (normal difficulty)
    hop=2+ -> prerequisite-of-a-prerequisite (candidate for "tricky")
    """
    result = session.run(
        """
        MATCH path = (t:Topic {topic_id: $topic_id})-[:REQUIRES*1..%d]->(prereq:Topic)
        WITH prereq, min(length(path)) AS hops, t.semester AS current_semester
        RETURN prereq.topic_id AS id, prereq.name AS name,
               prereq.semester AS semester, hops, current_semester
        ORDER BY hops
        """ % max_depth,
        topic_id=topic_id,
    )
    return [dict(r) for r in result]


def select_blend_topic(prereqs, difficulty, current_semester):
    """
    Picks which (if any) prerequisite topic to blend into the question,
    based on difficulty level.

    easy   -> no blending. Question stays scoped to the topic alone.
    medium -> blend a direct (hop=1) prerequisite. Normal "apply what
              you just learned plus one building block" difficulty.
    hard   -> blend a hop>=2 prerequisite, preferring one from an
              earlier SEMESTER (a genuine callback), not just an earlier
              module in the same course. Falls back to hop=2 same-semester
              if no cross-semester candidate exists.
    """
    if difficulty == "easy" or not prereqs:
        return None

    if difficulty == "medium":
        direct = [p for p in prereqs if p["hops"] == 1]
        return direct[0] if direct else None

    if difficulty == "hard":
        deep = [p for p in prereqs if p["hops"] >= 2]
        if not deep:
            # No deep prerequisite exists (topic is close to the "root"
            # of its own subject) -- fall back to a direct one rather
            # than blending nothing.
            direct = [p for p in prereqs if p["hops"] == 1]
            return direct[0] if direct else None

        cross_semester = [p for p in deep if p["semester"] < current_semester]
        if cross_semester:
            # Prefer the one from the EARLIEST semester -- the biggest
            # real callback -- then shallowest hop among ties, so it's
            # a genuine reach-back rather than an obscure 4-hop trivia pick.
            cross_semester.sort(key=lambda p: (p["semester"], p["hops"]))
            return cross_semester[0]

        # No cross-semester option -- take the shallowest deep prereq
        # within the same subject instead.
        deep.sort(key=lambda p: p["hops"])
        return deep[0]

    raise ValueError(f"Unknown difficulty: {difficulty}")


def build_question_prompt(topic, blend_topic, difficulty, marks):
    """
    Constructs the instruction block that goes to the LLM drafting node.
    """
    base = f"Write a {marks}-mark question on: {topic['name']}."

    if blend_topic is None:
        return base + " The question should test this topic on its own."

    if difficulty == "medium":
        return (
            base +
            f" The question should require applying \"{blend_topic['name']}\" "
            f"as a building block to answer it -- a natural, expected connection."
        )

    # hard
    return (
        base +
        f" To make this genuinely challenging, the question should require the "
        f"student to recall and apply \"{blend_topic['name']}\" "
        f"(from semester {blend_topic['semester']}) even though it isn't the "
        f"main subject of this question. This should feel like a deliberate "
        f"callback, not a random detour -- the connection must be real and "
        f"answerable, not forced."
    )


# ---------------------------------------------------------------------------
# Example: wiring this into the LangGraph node from your pipeline
# ---------------------------------------------------------------------------
def graph_lookup_node(state, driver):
    """
    Replaces the plain 'allowed_prior_topics' lookup with a
    difficulty-aware blend decision for the current topic being drafted.
    """
    topic_id = state["current_topic_id"]
    difficulty = state["target_difficulty"]  # "easy" | "medium" | "hard"

    with driver.session() as session:
        prereqs = get_prerequisites_with_distance(session, topic_id)
        # also fetch the topic's own record for its name/semester
        topic_record = session.run(
            "MATCH (t:Topic {topic_id: $id}) RETURN t.name AS name, t.semester AS semester",
            id=topic_id,
        ).single()

    topic = {"name": topic_record["name"], "semester": topic_record["semester"]}
    blend = select_blend_topic(prereqs, difficulty, topic["semester"])

    state["allowed_prior_topics"] = prereqs
    state["blend_topic"] = blend
    state["prompt_instruction"] = build_question_prompt(
        topic, blend, difficulty, state["target_marks"]
    )
    return state


if __name__ == "__main__":
    # Quick manual check against a live Neo4j instance
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "yourpassword"))
    with driver.session() as session:
        prereqs = get_prerequisites_with_distance(session, "ALGO_06")  # Dynamic Programming
        for p in prereqs:
            print(p)
        blend = select_blend_topic(prereqs, "hard", current_semester=4)
        print("\nChosen blend topic for HARD difficulty:", blend)
    driver.close()
