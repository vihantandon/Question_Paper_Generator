"""
Which prerequisite gets blended into a medium / hard question
(pipeline/difficulty_blend.py).

    python3 -m unittest tests.test_difficulty_blend -v      (from the repo root)

No Neo4j: the rule tests use hand-written rows shaped like the query's
result, and the whole-syllabus tests work the same rows out from the subject
YAML files (prereq_rows below does in Python what the Cypher query does).
"""
import random
import sys
import unittest
from itertools import permutations
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import difficulty_blend as B  # noqa: E402


def row(id_, semester, hops, strength="hard"):
    return {"id": id_, "name": id_, "semester": semester, "hops": hops, "strength": strength}


def pick(rows, difficulty, current_semester):
    chosen = B.select_blend_topic(rows, difficulty, current_semester)
    return chosen["id"] if chosen else None


# Real rows of three topics, as the query returns them (see the subject YAMLs).
NP_COMPLETE = [                                            # ALGO_09, semester 4: only soft links
    row("ALGO_04", 4, 1, "soft"), row("ALGO_05", 4, 1, "soft"), row("ALGO_06", 4, 1, "soft"),
    row("ALGO_03", 4, 2, "soft"), row("DS_05", 3, 2, "soft"),
    row("ALGO_01", 4, 3, "soft"), row("DS_01", 3, 3, "soft"), row("DS_02", 3, 3, "soft"),
    row("DS_03", 3, 3, "soft"),
    row("SDF1_03", 1, 4, "soft"), row("SDF1_04", 1, 4, "soft"), row("SDF2_02", 2, 4, "soft")]
POLYMORPHISM = [                                           # SDF2_04, semester 2
    row("SDF2_03", 2, 1), row("SDF2_02", 2, 2),
    row("SDF1_05", 1, 3), row("SDF1_06", 1, 3), row("SDF2_01", 2, 3),
    row("SDF1_02", 1, 4), row("SDF1_03", 1, 4), row("SDF1_04", 1, 4)]
FILE_HANDLING = [                                          # SDF1_07, semester 1
    row("SDF1_04", 1, 1, "soft"), row("SDF1_06", 1, 1),
    row("SDF1_02", 1, 2, "soft"), row("SDF1_03", 1, 2)]


class TestSamePickWhateverTheRowOrder(unittest.TestCase):
    """Neo4j does not promise an order for rows that tie on `hops`; the code
    used to take whichever came first."""

    def test_medium_and_hard_do_not_depend_on_row_order(self):
        for rows, semester in ((FILE_HANDLING, 1), (POLYMORPHISM[:5], 2)):
            for difficulty in ("medium", "hard"):
                picks = {pick(list(order), difficulty, semester) for order in permutations(rows)}
                self.assertEqual(len(picks), 1, f"{difficulty}: {picks}")

    def test_ties_go_to_the_lowest_topic_id(self):
        tied = [row("SDF1_04", 1, 2), row("SDF1_03", 1, 2), row("DS_01", 3, 1), row("DS_00", 3, 1)]
        self.assertEqual(pick(tied, "medium", 3), "DS_00")
        self.assertEqual(pick(tied, "hard", 3), "SDF1_03")


class TestLinkStrength(unittest.TestCase):
    def test_medium_takes_the_hard_link_not_the_soft_one(self):
        # File Handling: Structures is a hard prerequisite, Pointers a soft one
        self.assertEqual(pick(FILE_HANDLING, "medium", 1), "SDF1_06")

    def test_soft_link_is_used_when_there_is_no_hard_one(self):
        self.assertEqual(pick(NP_COMPLETE, "medium", 4), "ALGO_04")

    def test_hard_takes_the_hard_link_even_if_a_soft_one_is_nearer(self):
        rows = [row("DS_05", 3, 2, "soft"), row("DS_04", 3, 3, "hard"), row("ALGO_03", 4, 1)]
        self.assertEqual(pick(rows, "hard", 4), "DS_04")


class TestHardDifficulty(unittest.TestCase):
    def test_nearest_earlier_semester_topic_not_the_earliest_semester(self):
        # used to be Arrays (semester 1, 4 hops back); Graphs is 2 hops back
        self.assertEqual(pick(NP_COMPLETE, "hard", 4), "DS_05")

    def test_earlier_semester_wins_over_a_nearer_topic_of_the_same_semester(self):
        # Classes & Objects is 2 hops back but the same course; Functions is a real callback
        self.assertEqual(pick(POLYMORPHISM, "hard", 2), "SDF1_05")

    def test_same_semester_when_nothing_earlier_exists(self):
        self.assertEqual(pick(FILE_HANDLING, "hard", 1), "SDF1_03")

    def test_direct_prerequisite_when_nothing_is_two_hops_back(self):
        self.assertEqual(pick([row("SDF1_02", 1, 1)], "hard", 1), "SDF1_02")

    def test_a_direct_prerequisite_is_never_the_hard_pick_if_a_deeper_one_exists(self):
        rows = [row("SDF1_04", 1, 1), row("DS_01", 3, 1), row("SDF1_02", 1, 2)]
        self.assertEqual(pick(rows, "hard", 3), "SDF1_02")


class TestEasyAndBadInput(unittest.TestCase):
    def test_easy_blends_nothing(self):
        self.assertIsNone(pick(POLYMORPHISM, "easy", 2))

    def test_topic_without_prerequisites_blends_nothing(self):
        for difficulty in B.DIFFICULTIES:
            self.assertIsNone(pick([], difficulty, 1))

    def test_unknown_difficulty_is_an_error_with_or_without_prerequisites(self):
        # with no prerequisites this used to return None, as if "easy" had been asked for
        for rows in ([], POLYMORPHISM):
            with self.assertRaises(ValueError):
                B.select_blend_topic(rows, "tricky", 2)


# ------------------------------------------------------- the real syllabus
SEMESTER, NAME, LINKS = {}, {}, {}
for _f in sorted((ROOT / "prerequisite_graph" / "subjects").glob("*.y*ml")):
    _y = yaml.safe_load(_f.read_text(encoding="utf-8"))
    for _t in _y["topics"]:
        SEMESTER[_t["id"]], NAME[_t["id"]] = _y["semester"], _t["name"]
    for _p in _y.get("prerequisites", []):
        LINKS.setdefault(_p["topic"], []).append((_p["requires"], _p.get("strength", "hard")))


def fewest_hops(topic_id, only_hard, max_depth=4):
    best, frontier = {}, [topic_id]
    for depth in range(1, max_depth + 1):
        reached = []
        for node in frontier:
            for prereq, strength in LINKS.get(node, []):
                if prereq not in best and not (only_hard and strength != "hard"):
                    best[prereq] = depth
                    reached.append(prereq)
        frontier = reached
    return best


def prereq_rows(topic_id):
    """What get_prerequisites_with_distance returns, worked out from the YAMLs."""
    hops, hard_hops = fewest_hops(topic_id, False), fewest_hops(topic_id, True)
    return [{"id": p, "name": NAME[p], "semester": SEMESTER[p], "hops": h,
             "strength": "hard" if hard_hops.get(p) == h else "soft",
             "current_semester": SEMESTER[topic_id]} for p, h in hops.items()]


class TestWholeSyllabus(unittest.TestCase):
    def test_hand_written_rows_above_match_the_yaml_files(self):
        for topic_id, rows in (("ALGO_09", NP_COMPLETE), ("SDF2_04", POLYMORPHISM),
                               ("SDF1_07", FILE_HANDLING)):
            real = {(r["id"], r["semester"], r["hops"], r["strength"]) for r in prereq_rows(topic_id)}
            mine = {(r["id"], r["semester"], r["hops"], r["strength"]) for r in rows}
            self.assertEqual(real, mine, topic_id)

    def test_every_topic_gets_the_same_pick_in_any_row_order(self):
        shuffler = random.Random(0)
        for topic_id in SEMESTER:
            rows = prereq_rows(topic_id)
            for difficulty in ("medium", "hard"):
                picks = set()
                for _ in range(25):
                    shuffler.shuffle(rows)
                    picks.add(pick(rows, difficulty, SEMESTER[topic_id]))
                self.assertEqual(len(picks), 1, f"{topic_id} {difficulty}: {picks}")

    def test_hard_never_reaches_past_a_nearer_earlier_semester_topic(self):
        for topic_id, semester in SEMESTER.items():
            rows = prereq_rows(topic_id)
            earlier = [r for r in rows if r["hops"] >= 2 and r["semester"] < semester]
            if not earlier:
                continue
            chosen = B.select_blend_topic(rows, "hard", semester)
            self.assertLess(chosen["semester"], semester, topic_id)
            nearest = min(r["hops"] for r in earlier if r["strength"] == chosen["strength"])
            self.assertEqual(chosen["hops"], nearest, topic_id)

    def test_medium_uses_a_soft_link_only_for_topics_with_no_hard_one(self):
        for topic_id, semester in SEMESTER.items():
            direct = [r for r in prereq_rows(topic_id) if r["hops"] == 1]
            chosen = B.select_blend_topic(direct, "medium", semester)
            if chosen and chosen["strength"] == "soft":
                self.assertTrue(all(r["strength"] == "soft" for r in direct), topic_id)


# ------------------------------------------------------- the query
class RecordingSession:
    def __init__(self, rows):
        self.rows, self.query, self.params = rows, None, None

    def run(self, query, **params):
        self.query, self.params = query, params
        return self.rows


class TestPrerequisiteQuery(unittest.TestCase):
    def test_asks_for_link_strength_and_a_fixed_row_order(self):
        session = RecordingSession([row("DS_01", 3, 1)])
        rows = B.get_prerequisites_with_distance(session, "DS_03")
        self.assertEqual(rows, [row("DS_01", 3, 1)])
        self.assertEqual(session.params, {"topic_id": "DS_03"})
        self.assertIn("r.strength = 'hard'", session.query)
        self.assertIn("AS strength", session.query)
        self.assertIn("ORDER BY hops, id", session.query)


if __name__ == "__main__":
    unittest.main()
