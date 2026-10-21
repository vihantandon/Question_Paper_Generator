"""
pipeline/curriculum.py -- the subject YAML files, read straight from disk.

The pipeline gets a topic's prerequisites and COs from Neo4j (graph_lookup).
The classify step needs more: EVERY course outcome of the subject (a question
may assess a CO its topic is not mapped to) and the Bloom level the college
gives each CO (co_levels, which is not loaded into Neo4j). Neo4j is built
from these same files, so reading them here gives the same answer without a
database -- which also lets pipeline/eval_classify.py run with no Docker.

    from pipeline.curriculum import load_subjects, subject_of_topic
    subjects = load_subjects()
    ds = subject_of_topic(subjects, "DS_03")
    ds["cos"]["CO4"]        -> "Demonstrate and implement various operations ..."
    ds["co_levels"]["CO4"]  -> "Apply"
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SUBJECTS_DIR = ROOT / "prerequisite_graph" / "subjects"

BLOOM_LEVELS = ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create")
# word stems, checked in this order ("Applying", "Analyse", "Create level", ...)
_BLOOM_STEMS = (("remember", "Remember"), ("recall", "Remember"), ("knowledge", "Remember"),
                ("understand", "Understand"), ("comprehen", "Understand"),
                ("appl", "Apply"), ("analy", "Analyze"), ("evaluat", "Evaluate"),
                ("creat", "Create"), ("synthes", "Create"))


def normalize_bloom(text):
    """'Applying' / 'Analyse' / 'create level' -> the standard name; None if unknown."""
    low = str(text or "").strip().lower()
    for stem, level in _BLOOM_STEMS:
        if stem in low:
            return level
    return None


def bloom_rank(level):
    """Remember = 1 ... Create = 6."""
    return BLOOM_LEVELS.index(level) + 1


def load_subjects(folder=SUBJECTS_DIR):
    """{subject id: {"id", "name", "semester", "cos": {CO1: text},
    "co_levels": {CO1: "Understand"}, "topics": {topic id: {"name", "cos"}}}}"""
    out = {}
    for path in sorted(Path(folder).glob("*.y*ml")):
        y = yaml.safe_load(path.read_text(encoding="utf-8"))
        cos = {str(co): " ".join(str(text).split()) for co, text in (y.get("course_outcomes") or {}).items()}
        levels = {}
        for co, level in (y.get("co_levels") or {}).items():
            levels[str(co)] = normalize_bloom(level)
            if levels[str(co)] is None or str(co) not in cos:
                raise ValueError(f"{path.name}: co_levels {co}: {level!r} -- the CO must be in "
                                 f"course_outcomes and the level one of {', '.join(BLOOM_LEVELS)}")
        out[y["subject"]] = {
            "id": y["subject"],
            "name": y.get("subject_name") or y["subject"],
            "semester": y.get("semester"),
            "cos": cos,
            "co_levels": levels,
            "topics": {t["id"]: {"name": t["name"], "cos": [str(c) for c in t.get("co") or []]}
                       for t in y.get("topics") or []},
        }
    return out


def subject_of_topic(subjects, topic_id):
    """The subject a topic belongs to (DS_03 -> the DS subject)."""
    for subject in subjects.values():
        if topic_id in subject["topics"]:
            return subject
    raise KeyError(f"topic {topic_id!r} is in no subject YAML in {SUBJECTS_DIR}")
