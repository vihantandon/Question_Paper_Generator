"""
Generic prerequisite-graph loader.

Drop any number of subject YAML files into subjects/ (see subjects/*.yaml
for the schema) and this script loads them all into Neo4j. Adding a new
subject means adding a new YAML file -- no code changes needed.

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
    """Read every *.yaml file in subjects_dir and return (all_topics, all_prereqs)."""
    all_topics = []      # list of dicts: id, name, subject, semester
    all_prereqs = []      # list of dicts: topic, requires, strength
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

        for t in data.get("topics", []):
            all_topics.append({
                "id": t["id"],
                "name": t["name"],
                "subject": subject,
                "semester": semester,
            })

        for p in data.get("prerequisites", []):
            all_prereqs.append({
                "topic": p["topic"],
                "requires": p["requires"],
                "strength": p.get("strength", "hard"),
            })

        print(f"Read {path}: {len(data.get('topics', []))} topics, "
              f"{len(data.get('prerequisites', []))} prerequisite edges")

    return all_topics, all_prereqs


def validate(all_topics, all_prereqs):
    """Catch typos: every prereq must reference a topic_id that actually exists."""
    known_ids = {t["id"] for t in all_topics}
    problems = []
    for p in all_prereqs:
        if p["topic"] not in known_ids:
            problems.append(f"  '{p['topic']}' (in a prerequisite) is not a defined topic id")
        if p["requires"] not in known_ids:
            problems.append(f"  '{p['requires']}' (required by '{p['topic']}') is not a defined topic id")

    dupes = {t["id"] for t in all_topics if [x["id"] for x in all_topics].count(t["id"]) > 1}
    if dupes:
        problems.append(f"  duplicate topic ids across files: {sorted(dupes)}")

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


def build_graph(subjects_dir):
    all_topics, all_prereqs = load_subject_files(subjects_dir)
    validate(all_topics, all_prereqs)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            # Pass 1: create every topic node from every file first.
            for t in all_topics:
                session.execute_write(add_topic, t["id"], t["name"], t["subject"], t["semester"])
            # Pass 2: now that all nodes exist, edges can cross subject files
            # in any order (e.g. ALGO's file can reference DS's topic ids).
            for p in all_prereqs:
                session.execute_write(add_prerequisite, p["topic"], p["requires"], p["strength"])
    finally:
        driver.close()

    print(f"\nDone. Loaded {len(all_topics)} topics and {len(all_prereqs)} prerequisite edges.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subjects-dir",
        default=os.environ.get("SUBJECTS_DIR", "subjects"),
        help="Directory containing subject YAML files (default: ./subjects)",
    )
    args = parser.parse_args()
    build_graph(args.subjects_dir)
