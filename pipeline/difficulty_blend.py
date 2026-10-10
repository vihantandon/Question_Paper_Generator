"""
Difficulty-aware topic blending.

Given a target topic and a desired difficulty, picks which prerequisite
topic(s) (if any) should be blended into the LLM prompt to create a
"tricky" cross-topic question, based on how many hops back that
prerequisite sits in the graph.
"""

from neo4j import GraphDatabase

DIFFICULTIES = ("easy", "medium", "hard")


def get_prerequisites_with_distance(session, topic_id, max_depth=4):
    """
    Returns every prerequisite of topic_id, tagged with its MINIMUM hop
    distance (not just "reachable within N hops" -- the actual shortest
    distance), which semester it's from, and how strong the link is.

    hop=1  -> direct prerequisite (normal difficulty)
    hop=2+ -> prerequisite-of-a-prerequisite (candidate for "tricky")

    strength="hard" -> a shortest route to it uses only hard REQUIRES links
    strength="soft" -> every shortest route to it goes through a soft link

    Rows come back sorted by hops, then topic id, so the same graph always
    gives the same list.
    """
    result = session.run(
        """
        MATCH path = (t:Topic {topic_id: $topic_id})-[:REQUIRES*1..%d]->(prereq:Topic)
        WITH prereq, t.semester AS current_semester, length(path) AS hops,
             all(r IN relationships(path) WHERE r.strength = 'hard') AS all_hard
        WITH prereq, current_semester, min(hops) AS hops,
             min(CASE WHEN all_hard THEN hops END) AS hard_hops
        RETURN prereq.topic_id AS id, prereq.name AS name,
               prereq.semester AS semester, hops,
               CASE WHEN hard_hops = hops THEN 'hard' ELSE 'soft' END AS strength,
               current_semester
        ORDER BY hops, id
        """ % max_depth,
        topic_id=topic_id,
    )
    return [dict(r) for r in result]


def _pick(candidates):
    """
    The one prerequisite to blend out of several that qualify: a hard link
    before a soft one (a soft link is only used when there is no hard one),
    then the fewest hops, then the topic id. Nothing is left to the order
    the rows arrived in, so the same graph always gives the same pick.
    """
    if not candidates:
        return None
    return min(candidates, key=lambda p: (p.get("strength") == "soft", p["hops"], p["id"]))


def select_blend_topic(prereqs, difficulty, current_semester):
    """
    Picks which (if any) prerequisite topic to blend into the question,
    based on difficulty level.

    easy   -> no blending. Question stays scoped to the topic alone.
    medium -> blend a direct (hop=1) prerequisite. Normal "apply what
              you just learned plus one building block" difficulty.
    hard   -> blend a hop>=2 prerequisite: the NEAREST one from an
              earlier SEMESTER (a genuine callback), not just an earlier
              module in the same course. Falls back to the nearest hop>=2
              one in the same semester if no cross-semester candidate
              exists, and to a direct one if nothing is 2+ hops back.

    Ties are settled by _pick (hard link first, then hops, then topic id).
    """
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty: {difficulty!r} (expected one of: {', '.join(DIFFICULTIES)})")

    if difficulty == "easy" or not prereqs:
        return None

    direct = [p for p in prereqs if p["hops"] == 1]
    if difficulty == "medium":
        return _pick(direct)

    deep = [p for p in prereqs if p["hops"] >= 2]
    # Nearest first, not earliest semester first: reaching for semester 1
    # every time made NP-completeness blend "Arrays", 4 hops away, when
    # "Graphs" (semester 3) is 2 hops back.
    cross_semester = [p for p in deep if p["semester"] < current_semester]
    # No cross-semester option -- take the nearest deep prereq within the
    # same semester instead. No deep prerequisite at all (topic is close to
    # the "root" of its own subject) -- fall back to a direct one rather
    # than blending nothing.
    return _pick(cross_semester or deep or direct)


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
