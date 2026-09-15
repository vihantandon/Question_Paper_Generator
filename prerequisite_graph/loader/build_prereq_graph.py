"""
Generic prerequisite-graph loader.

Drop any number of subject YAML files into subjects/ (see subjects/*.yaml
for the schema) and this script loads them all into Neo4j. Adding a new
subject means adding a new YAML file -- no code changes needed.

Graph model:
    (:Topic)-[:REQUIRES {strength}]->(:Topic)
    (:Topic)-[:CONTRIBUTES_TO]->(:CO)
    (:CO {co_id, subject, description})

Run: python build_prereq_graph.py [--subjects-dir DIR]
Requires: pip install neo4j pyyaml
"""

import argparse
import glob
import os
import sys

import yaml
from neo4j import GraphDatabase

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "yourpassword")


def load_subject_files(subjects_dir):
    """Read every *.yaml file in subjects_dir and return
    (all_topics, all_prereqs, all_cos, all_topic_co_links)."""
    all_topics = []          # id, name, subject, semester
    all_prereqs = []         # topic, requires, strength
    all_cos = []             # co_id, subject, description
    all_topic_co_links = []  # topic_id, co_id

    files = sorted(glob.glob(os.path.join(subjects_dir, "*.yaml")) +
                   glob.glob(os.path.join(subjects_dir, "*.yml")))

    if not files:
        print(f"No YAML files found in {subjects_dir}", file=sys.stderr)
        sys.exit(1)

    for path in files:
        with open(path) as f:
            data = yaml.safe_load(f)

        subject = data["subject"]
        semester = data["semester"]

        # Course outcomes. CO ids are only unique WITHIN a subject
        # (every subject has its own CO1), so they're namespaced as
        # "DS:CO1" in the graph to keep them distinct across subjects.
        for co_id, description in data.get("course_outcomes", {}).items():
            all_cos.append({
                "co_id": f"{subject}:{co_id}",
                "short_id": co_id,
                "subject": subject,
                "description": description,
            })

        for t in data.get("topics", []):
            all_topics.append({
                "id": t["id"],
                "name": t["name"],
                "subject": subject,
                "semester": semester,
            })
            for co_id in t.get("co", []):
                all_topic_co_links.append({
                    "topic_id": t["id"],
                    "co_id": f"{subject}:{co_id}",
                })

        for p in data.get("prerequisites", []):
            all_prereqs.append({
                "topic": p["topic"],
                "requires": p["requires"],
                "strength": p.get("strength", "hard"),
            })

        print(f"Read {path}: {len(data.get('topics', []))} topics, "
              f"{len(data.get('prerequisites', []))} prerequisite edges, "
              f"{len(data.get('course_outcomes', {}))} COs")

    return all_topics, all_prereqs, all_cos, all_topic_co_links


def validate(all_topics, all_prereqs, all_cos, all_topic_co_links):
    """Catch typos: prereqs and CO links must reference things that exist."""
    known_topic_ids = {t["id"] for t in all_topics}
    known_co_ids = {c["co_id"] for c in all_cos}
    problems = []

    for p in all_prereqs:
        if p["topic"] not in known_topic_ids:
            problems.append(f"  '{p['topic']}' (in a prerequisite) is not a defined topic id")
        if p["requires"] not in known_topic_ids:
            problems.append(f"  '{p['requires']}' (required by '{p['topic']}') is not a defined topic id")

    for link in all_topic_co_links:
        if link["co_id"] not in known_co_ids:
            problems.append(
                f"  topic '{link['topic_id']}' maps to '{link['co_id']}', "
                f"which is not declared in that subject's course_outcomes block"
            )

    topic_id_list = [x["id"] for x in all_topics]
    dupes = {t["id"] for t in all_topics if topic_id_list.count(t["id"]) > 1}
    if dupes:
        problems.append(f"  duplicate topic ids across files: {sorted(dupes)}")

    # Warn (don't fail) on topics with no CO -- useful during the
    # transition while COs are still being filled in.
    linked_topics = {l["topic_id"] for l in all_topic_co_links}
    unlinked = sorted(known_topic_ids - linked_topics)
    if unlinked:
        print(f"\nWarning: {len(unlinked)} topic(s) have no CO mapping: {unlinked}",
              file=sys.stderr)

    # Warn on COs that no topic contributes to -- usually a typo or an
    # unused CO, and it would silently break CO coverage targets later.
    used_cos = {l["co_id"] for l in all_topic_co_links}
    orphan_cos = sorted(known_co_ids - used_cos)
    if orphan_cos:
        print(f"Warning: {len(orphan_cos)} CO(s) have no topic mapped to them: {orphan_cos}",
              file=sys.stderr)

    if problems:
        print("Validation failed:", file=sys.stderr)
        for p in problems:
            print(p, file=sys.stderr)
        sys.exit(1)


def add_topic(tx, topic_id, name, subject, semester):
    tx.run(
        """
        MERGE (t:Topic {topic_id: $topic_id})
        SET t.name = $name, t.subject = $subject, t.semester = $semester
        """,
        topic_id=topic_id, name=name, subject=subject, semester=semester,
    )


def add_co(tx, co_id, short_id, subject, description):
    tx.run(
        """
        MERGE (c:CO {co_id: $co_id})
        SET c.short_id = $short_id, c.subject = $subject, c.description = $description
        """,
        co_id=co_id, short_id=short_id, subject=subject, description=description,
    )


def add_prerequisite(tx, topic_id, prereq_topic_id, strength):
    tx.run(
        """
        MATCH (a:Topic {topic_id: $topic_id})
        MATCH (b:Topic {topic_id: $prereq_topic_id})
        MERGE (a)-[r:REQUIRES]->(b)
        SET r.strength = $strength
        """,
        topic_id=topic_id, prereq_topic_id=prereq_topic_id, strength=strength,
    )


def add_topic_co_link(tx, topic_id, co_id):
    tx.run(
        """
        MATCH (t:Topic {topic_id: $topic_id})
        MATCH (c:CO {co_id: $co_id})
        MERGE (t)-[:CONTRIBUTES_TO]->(c)
        """,
        topic_id=topic_id, co_id=co_id,
    )


def build_graph(subjects_dir):
    all_topics, all_prereqs, all_cos, all_topic_co_links = load_subject_files(subjects_dir)
    validate(all_topics, all_prereqs, all_cos, all_topic_co_links)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            # Pass 1: create every node (topics + COs) from every file first.
            for t in all_topics:
                session.execute_write(add_topic, t["id"], t["name"], t["subject"], t["semester"])
            for c in all_cos:
                session.execute_write(add_co, c["co_id"], c["short_id"], c["subject"], c["description"])

            # Pass 2: now that all nodes exist, edges can cross subject files
            # in any order (e.g. ALGO's file can reference DS's topic ids).
            for p in all_prereqs:
                session.execute_write(add_prerequisite, p["topic"], p["requires"], p["strength"])
            for link in all_topic_co_links:
                session.execute_write(add_topic_co_link, link["topic_id"], link["co_id"])
    finally:
        driver.close()

    print(f"\nDone. Loaded {len(all_topics)} topics, {len(all_prereqs)} prerequisite edges, "
          f"{len(all_cos)} COs, {len(all_topic_co_links)} topic->CO links.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subjects-dir",
        default=os.environ.get("SUBJECTS_DIR", "subjects"),
        help="Directory containing subject YAML files (default: ./subjects)",
    )
    args = parser.parse_args()
    build_graph(args.subjects_dir)