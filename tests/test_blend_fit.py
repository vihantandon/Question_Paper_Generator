"""
Which prerequisite to blend: the candidates the graph allows
(difficulty_blend.blend_candidates), and the one whose material fits the main
topic best (retrieval.blend_fit, chosen in pipeline/nodes/retrieve.py).

    python3 -m unittest tests.test_blend_fit -v      (from the repo root)
"""
import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import retrieval  # noqa: E402
import tests.test_phase1 as P  # noqa: E402
from pipeline import difficulty_blend as B  # noqa: E402
from pipeline.graph import print_result  # noqa: E402
from pipeline.nodes.retrieve import choose_blend, make_retrieve_node  # noqa: E402


def row(id_, name, semester, hops, strength="hard"):
    return {"id": id_, "name": name, "semester": semester, "hops": hops, "strength": strength,
            "current_semester": 4}


# ALGO_06 Dynamic Programming, from the subject YAMLs
DP_PREREQS = [row("ALGO_03", "Divide and Conquer", 4, 1), row("ALGO_04", "Greedy Algorithms", 4, 1, "soft"),
              row("ALGO_01", "Asymptotic Analysis", 4, 2), row("DS_02", "Searching and Sorting", 3, 2),
              row("DS_05", "Graphs", 3, 2), row("DS_01", "Linear Data Structures", 3, 3),
              row("DS_03", "Trees and Heaps", 3, 3), row("SDF1_03", "Arrays", 1, 4)]


class TestCandidates(unittest.TestCase):
    def test_hard_earlier_semester_nearest_two_hop_counts(self):
        got = [c["id"] for c in B.blend_candidates(DP_PREREQS, "hard", 4)]
        self.assertEqual(got, ["DS_02", "DS_05", "DS_01", "DS_03"])
        self.assertEqual(B.select_blend_topic(DP_PREREQS, "hard", 4)["id"], got[0])   # the old pick comes first

    def test_medium_direct_and_easy_none(self):
        self.assertEqual([c["id"] for c in B.blend_candidates(DP_PREREQS, "medium", 4)], ["ALGO_03", "ALGO_04"])
        self.assertEqual(B.blend_candidates(DP_PREREQS, "easy", 4), [])
        self.assertEqual(B.blend_candidates([], "hard", 4), [])
        with self.assertRaises(ValueError):
            B.blend_candidates(DP_PREREQS, "tricky", 4)


def chunk(id_, topic, text, vec):
    return P.chunk(id_, topic, text, vec)


# "dynamic programming" points to [1, 0]; graph text near it, sorting text away from it
BOOKS = P.FakeCollection("book_content", [
    chunk("dp1", "ALGO_06", "Dynamic programming solves overlapping subproblems.", [1.0, 0.0]),
    chunk("dp2", "ALGO_06", "Rod cutting by dynamic programming.", [0.95, 0.31]),
    chunk("g1", "DS_05", "Shortest paths in a DAG relax edges in topological order.", [0.9, 0.44]),
    chunk("g2", "DS_05", "Breadth-first search finds shortest paths by edge count.", [0.8, 0.6]),
    chunk("s1", "DS_02", "Counting sort sorts integers in a known range.", [0.2, 0.98]),
    chunk("s2", "DS_02", "Binary search halves the range at each step.", [0.0, 1.0]),
])


def embed(text):
    return [1.0, 0.0] if "dynamic programming" in text.lower() else [0.0, 1.0]


DP = {"id": "ALGO_06", "name": "Dynamic Programming", "semester": 4}


class TestFit(unittest.TestCase):
    def test_the_prerequisite_whose_material_is_closest_wins(self):
        graphs, sorting = row("DS_05", "Graphs", 3, 2), row("DS_02", "Searching and Sorting", 3, 2)
        self.assertGreater(retrieval.blend_fit([BOOKS], embed, DP, graphs), retrieval.blend_fit([BOOKS], embed, DP, sorting))
        best, scores = choose_blend([BOOKS], embed, DP, [sorting, graphs], sorting)
        self.assertEqual(best["id"], "DS_05")
        self.assertEqual([i for i, _ in scores], ["DS_02", "DS_05"])

    def test_a_prerequisite_with_no_text_never_wins_and_one_candidate_is_kept(self):
        none = row("DS_09", "No material", 3, 2)
        sorting = row("DS_02", "Searching and Sorting", 3, 2)
        self.assertIsNone(retrieval.blend_fit([BOOKS], embed, DP, none))
        best, scores = choose_blend([BOOKS], embed, DP, [none, sorting], none)
        self.assertEqual((best["id"], scores), ("DS_02", [("DS_09", None), ("DS_02", scores[1][1])]))
        self.assertEqual(choose_blend([BOOKS], embed, DP, [sorting], sorting), (sorting, []))
        self.assertEqual(choose_blend([BOOKS], embed, DP, [none, row("X_1", "x", 1, 2)], none)[0], none)

    def test_retrieve_swaps_the_blend_and_rewrites_the_instruction(self):
        sorting, graphs = row("DS_02", "Searching and Sorting", 3, 2), row("DS_05", "Graphs", 3, 2)
        node = make_retrieve_node([BOOKS], None, embed, n_main=2, n_blend=1)
        state = {"current_topic": DP, "blend_topic": sorting, "blend_candidates": [sorting, graphs],
                 "target_difficulty": "hard", "target_marks": 7,
                 "prompt_instruction": B.build_question_prompt(DP, sorting, "hard", 7)}
        out = node(state)
        self.assertEqual(out["blend_topic"]["id"], "DS_05")
        self.assertIn('apply "Graphs" (from semester 3)', out["prompt_instruction"])
        self.assertNotIn("Sorting", out["prompt_instruction"])
        self.assertEqual({c["topic_id"] for c in out["relevant_chunks"] if c["role"] == "blend"}, {"DS_05"})
        printed = io.StringIO()
        state.update(out, topic_cos=[], style_examples=[], grounding_warning=None)
        with contextlib.redirect_stdout(printed):
            print_result(dict(state, current_topic=DP))
        self.assertIn("Blend : DS_05 Graphs (semester 3, 2 hop(s) back)\n        best fit of: DS_02 ", printed.getvalue())

    def test_the_graphs_choice_stays_when_it_fits_best(self):
        graphs, sorting = row("DS_05", "Graphs", 3, 2), row("DS_02", "Searching and Sorting", 3, 2)
        node = make_retrieve_node([BOOKS], None, embed, n_main=2, n_blend=1)
        out = node({"current_topic": DP, "blend_topic": graphs, "blend_candidates": [graphs, sorting],
                    "target_difficulty": "hard", "target_marks": 7, "prompt_instruction": "keep me"})
        self.assertNotIn("blend_topic", out)
        self.assertNotIn("prompt_instruction", out)


if __name__ == "__main__":
    unittest.main()
