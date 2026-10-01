"""
Phase 1 unit tests. They use small fake objects instead of Neo4j / Chroma /
the embedding model, so they run in about a second without Docker:

    python3 -m unittest tests.test_phase1 -v      (from the repo root)

A fake object only needs the methods the code actually calls (for Chroma:
.query(); for the embedder: a function text -> vector). This is the main
trick for testing code that normally talks to a database.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Extracting" / "pyqs_processing"))

import resolve_pyq_labels as R  # noqa: E402
import retrieval  # noqa: E402


class FakeCollection:
    """Just enough of a Chroma collection for our code: .name and .query().
    rows: list of (id, text, metadata, vector). Distance = squared L2,
    like Chroma's default."""

    def __init__(self, name, rows):
        self.name, self.rows = name, rows

    @staticmethod
    def _match(meta, where):
        if "$and" in where:
            return all(FakeCollection._match(meta, w) for w in where["$and"])
        return all(meta.get(k) == v for k, v in where.items())

    def query(self, query_embeddings, n_results, where=None, include=None):
        q = query_embeddings[0]
        hits = [(sum((a - b) ** 2 for a, b in zip(q, vec)), i, t, m)
                for i, t, m, vec in self.rows if not where or self._match(m, where)]
        hits.sort(key=lambda h: h[0])
        hits = hits[:n_results]
        return {"ids": [[h[1] for h in hits]], "documents": [[h[2] for h in hits]],
                "metadatas": [[h[3] for h in hits]], "distances": [[h[0] for h in hits]]}


def fake_embed(text):
    """2-D 'embedding': texts mentioning 'tree' point one way, others the other."""
    return [1.0, 0.0] if "tree" in text.lower() else [0.0, 1.0]


# ---------------------------------------------------------------- step 1
class TestBloom(unittest.TestCase):
    def test_all_16_spellings_found_in_the_papers(self):
        seen = {
            "Analyse level": "Analyze", "Analysing": "Analyze", "Analyze Level": "Analyze",
            "Analyzing": "Analyze", "Apply": "Apply", "Apply L3": "Apply",
            "Apply Level": "Apply", "Apply level": "Apply", "Applying": "Apply",
            "Create Level": "Create", "Create level": "Create", "Creating": "Create",
            "Creation": "Create", "Understand": "Understand", "Understand L2": "Understand",
            "Understanding": "Understand",
        }
        for raw, want in seen.items():
            self.assertEqual(R.normalize_bloom(raw), want, raw)

    def test_number_only_and_unknown(self):
        self.assertEqual(R.normalize_bloom("L4"), "Analyze")
        self.assertIsNone(R.normalize_bloom("hard"))
        self.assertIsNone(R.normalize_bloom(None))


class TestDecideTopic(unittest.TestCase):
    def test_picks_best_co_topic(self):
        d = R.decide_topic(["A", "B"], {"A": 0.50, "B": 0.40, "C": 0.30})
        self.assertEqual((d["topic_id"], d["method"], d["mismatch"]), ("A", "co+content", False))
        self.assertAlmostEqual(d["confidence"], 0.10)

    def test_small_lead_of_outside_topic_does_not_override_co(self):
        d = R.decide_topic(["A"], {"A": 0.50, "C": 0.52})      # C wins by 0.02 < 0.05
        self.assertEqual(d["topic_id"], "A")
        self.assertFalse(d["mismatch"])

    def test_clear_outside_winner_overrides_and_flags(self):
        # the DS quicksort case: CO2 -> only "Linear DS", but content says "Sorting"
        d = R.decide_topic(["DS_01"], {"DS_01": 0.35, "DS_02": 0.55})
        self.assertEqual((d["topic_id"], d["method"], d["mismatch"]), ("DS_02", "content-override", True))
        self.assertTrue(d["reasons"])

    def test_low_confidence_is_flagged(self):
        d = R.decide_topic(["A", "B"], {"A": 0.500, "B": 0.495})
        self.assertEqual(d["topic_id"], "A")
        self.assertTrue(any("low confidence" in r for r in d["reasons"]))

    def test_no_co_uses_content_only(self):
        d = R.decide_topic([], {"A": 0.3, "B": 0.6})
        self.assertEqual((d["topic_id"], d["method"]), ("B", "content"))

    def test_single_co_topic_without_text_is_trusted(self):
        # e.g. SDF1_01 has no book text: we cannot score it, so keep the paper's CO
        d = R.decide_topic(["SDF1_01"], {"SDF1_02": 0.6})
        self.assertEqual((d["topic_id"], d["method"]), ("SDF1_01", "co-only"))


SUBJECT = {
    "cos": {"CO1": "one", "CO3": "three"},
    "topics": {"T1": {"name": "Topic one", "co": ["CO1"]},
               "T2": {"name": "Topic two", "co": ["CO3"]},
               "T3": {"name": "Topic three", "co": ["CO3"]}},
}


class TestResolveRecord(unittest.TestCase):
    def meta(self, **kw):
        base = {"type": "question", "subject": "X", "raw_co_code": "CO3", "course_outcome": "CO3",
                "bloom_level": "Apply L3", "topic_id": None, "needs_review": True,
                "review_reason": "CO 'CO3' maps to 2 topics"}
        base.update(kw)
        return base

    def test_full_clean(self):
        new = R.resolve_record(self.meta(), SUBJECT, {"T1": 0.2, "T2": 0.3, "T3": 0.6})
        self.assertEqual(new["bloom_level"], "Apply")
        self.assertEqual(new["bloom_level_raw"], "Apply L3")
        self.assertEqual(new["bloom_rank"], 3)
        self.assertEqual((new["topic_id"], new["topic_name"]), ("T3", "Topic three"))
        self.assertFalse(new["needs_review"])
        self.assertIsNone(new["review_reason"])

    def test_ocr_co_is_repaired(self):
        new = R.resolve_record(self.meta(raw_co_code="C03", course_outcome=None),
                               SUBJECT, {"T2": 0.3, "T3": 0.6})
        self.assertEqual(new["course_outcome"], "CO3")
        self.assertEqual(new["co_description"], "three")

    def test_manual_topic_is_kept(self):
        m = self.meta(topic_id="T1", review_reason="topic set manually from content")
        new = R.resolve_record(m, SUBJECT, {"T2": 0.9})
        self.assertEqual((new["topic_id"], new["tag_method"]), ("T1", "manual"))
        self.assertEqual(new["bloom_level"], "Apply")          # bloom still cleaned

    def test_running_twice_gives_same_result(self):
        once = R.resolve_record(self.meta(), SUBJECT, {"T2": 0.3, "T3": 0.6})
        twice = R.resolve_record(once, SUBJECT, {"T2": 0.3, "T3": 0.6})
        self.assertEqual(once, twice)

    def test_input_is_not_modified(self):
        m = self.meta()
        R.resolve_record(m, SUBJECT, {"T2": 0.3, "T3": 0.6})
        self.assertEqual(m["bloom_level"], "Apply L3")

    def test_no_vector_yet(self):
        new = R.resolve_record(self.meta(), SUBJECT, None)
        self.assertTrue(new["needs_review"])
        self.assertIn("embed_pyqs", new["review_reason"])


# ---------------------------------------------------------------- step 2
def pyq(id_, topic, text, vec):
    return (id_, text, {"topic_id": topic, "subject": topic.split("_")[0], "marks": 5,
                        "bloom_level": "Apply", "course_outcome": "CO1",
                        "year": 2024, "exam": "test1"}, vec)


PYQS = FakeCollection("pyq_bank", [
    pyq("DS_q1", "DS_03", "Insert into a binary search tree", [1.0, 0.0]),
    pyq("DS_q2", "DS_03", "Delete a node from an AVL tree", [0.9, 0.1]),
    pyq("DS_q3", "DS_01", "Reverse a linked list", [0.0, 1.0]),
    pyq("ALGO_q1", "ALGO_06", "Knapsack with dynamic programming", [0.5, 0.5]),
])


class TestFetchSimilarPyqs(unittest.TestCase):
    def test_same_topic_first_then_same_subject(self):
        hits = retrieval.fetch_similar_pyqs(PYQS, fake_embed, "Trees", "DS_03", n=3)
        self.assertEqual([h["id"] for h in hits], ["DS_q1", "DS_q2", "DS_q3"])
        self.assertEqual([h["match"] for h in hits], ["topic", "topic", "subject"])

    def test_never_leaves_the_subject(self):
        hits = retrieval.fetch_similar_pyqs(PYQS, fake_embed, "Trees", "DS_03", n=10)
        self.assertTrue(all(h["subject"] == "DS" for h in hits))
        self.assertEqual(len(hits), 3)

    def test_exclude_ids(self):
        hits = retrieval.fetch_similar_pyqs(PYQS, fake_embed, "Trees", "DS_03", n=2,
                                            exclude_ids=["DS_q1"])
        self.assertEqual([h["id"] for h in hits], ["DS_q2", "DS_q3"])

    def test_no_collection_gives_empty_list(self):
        self.assertEqual(retrieval.fetch_similar_pyqs(None, fake_embed, "x", "DS_03"), [])


# ---------------------------------------------------------------- step 3
from pipeline.graph import build_graph            # noqa: E402
from pipeline.resources import Resources          # noqa: E402


class FakeResult(list):
    def single(self):
        return self[0] if self else None


class FakeSession:
    """Answers the three Cypher queries graph_lookup sends, from a dict."""

    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        tid = params.get("topic_id") or params.get("id")
        if "REQUIRES*" in query:
            return FakeResult(self.data["prereqs"].get(tid, []))
        if "CONTRIBUTES_TO" in query:
            return FakeResult(self.data["cos"].get(tid, []))
        if "RETURN t.name" in query:
            t = self.data["topics"].get(tid)
            return FakeResult([t] if t else [])
        raise AssertionError(f"unexpected query: {query}")


class FakeDriver:
    def __init__(self, data):
        self.data = data

    def session(self):
        return FakeSession(self.data)

    def close(self):
        pass


GRAPH = {
    "topics": {"DS_03": {"name": "Trees", "semester": 3},
               "DS_07": {"name": "Topic with no study material", "semester": 3}},
    "prereqs": {"DS_03": [
        {"id": "DS_01", "name": "Linked list", "semester": 3, "hops": 1, "current_semester": 3},
        {"id": "SDF1_04", "name": "Pointers", "semester": 1, "hops": 2, "current_semester": 3}]},
    "cos": {"DS_03": [{"co_id": "DS:CO4", "short_id": "CO4", "description": "trees"}]},
}


def chunk(id_, topic, text, vec):
    return (id_, text, {"topic_id": topic, "needs_review": False, "subject": "DS", "type": "book",
                        "section_title": "sec", "page_start": 1, "page_end": 2,
                        "is_prereq_content": False}, vec)


BOOK = FakeCollection("book_content", [
    chunk("b1", "DS_03", "A binary tree node has left and right children", [1.0, 0.0]),
    chunk("b2", "DS_03", "Tree traversal: inorder, preorder", [0.9, 0.1]),
    chunk("b3", "SDF1_04", "A pointer stores an address", [0.0, 1.0]),
])


def run_pipeline(topic_id, difficulty, marks=5):
    res = Resources(driver=FakeDriver(GRAPH), grounding_collections=[BOOK],
                    pyq_collection=PYQS, embed_fn=fake_embed)
    return build_graph(res).invoke({"current_topic_id": topic_id,
                                    "target_difficulty": difficulty, "target_marks": marks})


class TestPipeline(unittest.TestCase):
    def test_hard_blends_earlier_semester_and_retrieves_both(self):
        r = run_pipeline("DS_03", "hard", 10)
        self.assertEqual(r["current_topic"], {"id": "DS_03", "name": "Trees", "semester": 3})
        self.assertEqual(r["blend_topic"]["id"], "SDF1_04")          # 2 hops, semester 1
        self.assertIn("semester 1", r["prompt_instruction"])
        roles = {c["role"]: c["topic_id"] for c in r["relevant_chunks"]}
        self.assertEqual(roles, {"main": "DS_03", "blend": "SDF1_04"})
        self.assertEqual(r["style_examples"][0]["id"], "DS_q1")
        self.assertIsNone(r["grounding_warning"])

    def test_easy_has_no_blend(self):
        r = run_pipeline("DS_03", "easy")
        self.assertIsNone(r["blend_topic"])
        self.assertTrue(all(c["role"] == "main" for c in r["relevant_chunks"]))

    def test_topic_without_text_gets_a_warning(self):
        r = run_pipeline("DS_07", "easy")
        self.assertEqual(r["relevant_chunks"], [])
        self.assertIn("No grounding text for DS_07", r["grounding_warning"])

    def test_unknown_topic_fails_clearly(self):
        with self.assertRaises(ValueError):
            run_pipeline("NOPE_01", "easy")


class TestDotenv(unittest.TestCase):
    def test_reads_file_but_terminal_wins(self):
        import os
        import tempfile
        from pipeline.resources import load_dotenv_file
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / ".env"
            env.write_text('# comment\nQPG_TEST_A=from_file\nexport QPG_TEST_B="quoted"\n'
                           'QPG_TEST_C=from_file\nbad line\n', encoding="utf-8")
            os.environ["QPG_TEST_C"] = "from_terminal"
            try:
                load_dotenv_file(env)
                self.assertEqual(os.environ["QPG_TEST_A"], "from_file")
                self.assertEqual(os.environ["QPG_TEST_B"], "quoted")
                self.assertEqual(os.environ["QPG_TEST_C"], "from_terminal")
            finally:
                for k in ("QPG_TEST_A", "QPG_TEST_B", "QPG_TEST_C"):
                    os.environ.pop(k, None)


if __name__ == "__main__":
    unittest.main()