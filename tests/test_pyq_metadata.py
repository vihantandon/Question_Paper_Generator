"""
PYQ labels in pyq_bank (Embedding/embed_pyqs.py + resolve_pyq_labels.py):
empty values, hand-fixed ("manual") records, and topic scoring.

    python3 -m unittest tests.test_pyq_metadata -v      (from the repo root)

No Docker and no embedding model. FakeCollection keeps rows in memory and
treats metadata the way Chroma 1.5 does: upsert() / update() only overwrite
the keys they are given, and a key given as None is deleted.
"""
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Embedding"))
sys.path.insert(0, str(ROOT / "Extracting" / "pyqs_processing"))

import embed_pyqs  # noqa: E402
import resolve_pyq_labels as R  # noqa: E402


class FakeCollection:
    """rows: {id: {"document", "metadata", "embedding"}}"""

    def __init__(self, name, rows=None):
        self.name, self.rows = name, dict(rows or {})

    def _merge(self, id_, metadata):
        row = self.rows.setdefault(id_, {"document": "", "metadata": {}, "embedding": [0.0, 0.0]})
        for key, value in (metadata or {}).items():
            if value is None:
                row["metadata"].pop(key, None)
            else:
                row["metadata"][key] = value
        return row

    def upsert(self, ids, embeddings, documents, metadatas):
        for id_, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            self._merge(id_, meta).update(document=doc, embedding=emb)

    def update(self, ids, metadatas):
        for id_, meta in zip(ids, metadatas):
            self._merge(id_, meta)

    def get(self, ids=None, include=None):
        ids = [i for i in (ids if ids is not None else self.rows) if i in self.rows]
        return {"ids": ids,
                "metadatas": [dict(self.rows[i]["metadata"]) for i in ids],
                "embeddings": [self.rows[i]["embedding"] for i in ids]}

    def query(self, query_embeddings, n_results, where=None, include=None):
        wanted = {k: v for cond in where["$and"] for k, v in cond.items()}
        hits = sorted((sum((a - b) ** 2 for a, b in zip(query_embeddings[0], r["embedding"])), i)
                      for i, r in self.rows.items()
                      if all(r["metadata"].get(k) == v for k, v in wanted.items()))[:n_results]
        return {"ids": [[i for _, i in hits]],
                "documents": [[self.rows[i]["document"] for _, i in hits]],
                "metadatas": [[self.rows[i]["metadata"] for _, i in hits]],
                "distances": [[d for d, _ in hits]]}

    def count(self):
        return len(self.rows)


class FakeClient:
    def __init__(self, *collections):
        self.collections = {c.name: c for c in collections}

    def heartbeat(self):
        return 1

    def list_collections(self):
        return list(self.collections)

    def get_collection(self, name):
        return self.collections[name]

    def get_or_create_collection(self, name):
        return self.collections.setdefault(name, FakeCollection(name))


class FakeVectors(list):
    def tolist(self):
        return list(self)


class FakeModel:
    def __init__(self, name):
        pass

    def encode(self, texts, show_progress_bar=False):
        return FakeVectors([[1.0, 0.0] for _ in texts])


def fake_modules(client):
    return {"chromadb": types.SimpleNamespace(HttpClient=lambda host, port: client),
            "sentence_transformers": types.SimpleNamespace(SentenceTransformer=FakeModel),
            "tqdm": types.SimpleNamespace(tqdm=lambda items, **kwargs: items)}


def quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        fn()


def chunk(id_, topic, text, vec, needs_review=False):
    return id_, {"document": text, "embedding": vec,
                 "metadata": {"topic_id": topic, "needs_review": needs_review}}


QUESTION = "Q1. Insert the keys 5, 3 and 8 into an empty binary search tree and draw it."


def question(id_, **metadata):
    base = {"type": "question", "subject": "X", "year": 2024, "exam": "test1", "question_number": 1,
            "raw_co_code": "CO1", "course_outcome": "CO1", "bloom_level": "Apply", "marks": 5,
            "marks_breakdown": None, "topic_id": "T1", "topic_name": "Topic one",
            "needs_review": False, "review_reason": None}
    base.update(metadata)
    return {"id": id_, "document": QUESTION, "metadata": base}


SUBJECT = {"cos": {"CO1": "one", "CO3": "three"},
           "topics": {"T1": {"name": "Topic one", "co": ["CO1"]},
                      "T2": {"name": "Topic two", "co": ["CO3"]},
                      "T3": {"name": "Topic three", "co": ["CO3"]}}}


# ---------------------------------------------------------------- empty values
class TestSanitizeMetadata(unittest.TestCase):
    def test_an_empty_value_is_left_out(self):
        # it used to become the text "NONE" (or -1 for marks)
        clean = embed_pyqs.sanitize_metadata({"marks": None, "bloom_level": None,
                                              "topic_confidence": None, "year": 2024})
        self.assertEqual(clean, {"year": 2024})

    def test_values_keep_their_type_and_lists_become_json_text(self):
        clean = embed_pyqs.sanitize_metadata({"marks": 5, "topic_confidence": 0.25, "needs_review": False,
                                              "marks_breakdown": [{"marks": 3}, {"marks": 2}]})
        self.assertEqual(clean, {"marks": 5, "topic_confidence": 0.25, "needs_review": False,
                                 "marks_breakdown": '[{"marks": 3}, {"marks": 2}]'})

    def test_a_value_chroma_still_holds_is_sent_as_none_so_it_gets_deleted(self):
        clean = embed_pyqs.sanitize_metadata(
            {"marks": 5, "topic_confidence": None},
            stored={"marks": 4, "topic_confidence": "NONE", "topic_scores": "[]"})
        self.assertEqual(clean, {"marks": 5,
                                 "topic_confidence": None,       # empty in the record now
                                 "topic_scores": None})          # gone from the record


OLD_ROW = {"type": "question", "subject": "X", "marks": -1, "topic_id": "T1",
           "topic_confidence": "NONE", "review_reason": "NONE", "marks_breakdown": "NONE",
           "co_topic_mismatch": True, "topic_scores": "[[\"T1\", 0.5]]"}


class TestEmbedPyqs(unittest.TestCase):
    def test_re_embedding_leaves_no_placeholder_text_and_no_old_keys(self):
        bank = FakeCollection("pyq_bank", {"X_q1": {"document": QUESTION, "embedding": [1.0, 0.0],
                                                    "metadata": dict(OLD_ROW)}})
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "X_2024_test1_pyq.json").write_text(
                json.dumps([question("X_q1", topic_confidence=None), question("X_q2", marks=None)]),
                encoding="utf-8")
            with mock.patch.dict(sys.modules, fake_modules(FakeClient(bank))), \
                    mock.patch.object(sys, "argv", ["embed_pyqs.py", "--dir", d]):
                quiet(embed_pyqs.main)
        old, new = bank.rows["X_q1"]["metadata"], bank.rows["X_q2"]["metadata"]
        for meta in (old, new):
            self.assertNotIn("NONE", meta.values())
            self.assertNotIn(-1, meta.values())
        self.assertEqual(old["marks"], 5)
        for gone in ("topic_confidence", "review_reason", "marks_breakdown",     # empty in the file
                     "co_topic_mismatch", "topic_scores"):                       # not in the file at all
            self.assertNotIn(gone, old)
        self.assertNotIn("marks", new)                   # a question with no marks has no marks value


class TestResolveWrite(unittest.TestCase):
    def run_write(self, records, bank):
        book = FakeCollection("book_content", dict([
            chunk("b1", "T1", "binary search tree insertion", [1.0, 0.0]),
            chunk("b2", "T2", "hash table collisions", [0.0, 1.0])]))
        client = FakeClient(bank, book, FakeCollection("tut_content"))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "X_2024_test1_pyq.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            with mock.patch.dict(sys.modules, {"chromadb": fake_modules(client)["chromadb"]}), \
                    mock.patch.object(R, "load_subjects", lambda: {"X": SUBJECT}), \
                    mock.patch.object(R, "load_pyq_files", lambda: {path: json.loads(path.read_text())}), \
                    mock.patch.object(sys, "argv", ["resolve_pyq_labels.py", "--write"]):
                quiet(R.main)
            return json.loads(path.read_text(encoding="utf-8"))

    def test_write_leaves_no_placeholder_text_and_no_old_keys(self):
        manual = question("X_q2", tag_method="manual", topic_id="T2", topic_name="Topic two",
                          review_reason="manual: topic set to T2 after reading the question")
        bank = FakeCollection("pyq_bank", {
            "X_q1": {"document": QUESTION, "embedding": [1.0, 0.0], "metadata": dict(OLD_ROW)},
            "X_q2": {"document": QUESTION, "embedding": [1.0, 0.0],
                     "metadata": dict(OLD_ROW, topic_confidence=0.02)}})
        self.run_write([question("X_q1"), manual], bank)

        auto, hand = bank.rows["X_q1"]["metadata"], bank.rows["X_q2"]["metadata"]
        for meta in (auto, hand):
            self.assertNotIn("NONE", meta.values())
        # T1 is CO1's only topic, so there is no runner-up and no confidence number:
        # the row has no value, not a text where other rows have a number
        self.assertEqual((auto["topic_id"], auto["tag_method"]), ("T1", "co+content"))
        self.assertNotIn("topic_confidence", auto)
        self.assertNotIn("review_reason", auto)
        # the hand-fixed record has no score fields in the file, so none are left in Chroma
        self.assertEqual((hand["topic_id"], hand["tag_method"]), ("T2", "manual"))
        for gone in ("topic_confidence", "co_topic_mismatch", "topic_scores"):
            self.assertNotIn(gone, hand)


# ---------------------------------------------------------------- manual records
class TestManualRecords(unittest.TestCase):
    def manual(self, **kw):
        base = {"type": "question", "subject": "X", "raw_co_code": "CO3", "course_outcome": "CO3",
                "bloom_level": "Apply L3", "topic_id": "T1", "tag_method": "manual",
                "needs_review": True, "review_reason": "manual: topic set to T1; paper tags CO3"}
        base.update(kw)
        return base

    def test_hand_fixed_record_is_no_longer_flagged(self):
        # the flag used to stay as it was before the human looked at the question
        new = R.resolve_record(self.manual(), SUBJECT, {"T2": 0.9})
        self.assertFalse(new["needs_review"])
        self.assertEqual(new["topic_id"], "T1")
        self.assertEqual(new["review_reason"], "manual: topic set to T1; paper tags CO3")

    def test_a_label_problem_still_flags_it_and_says_why(self):
        once = R.resolve_record(self.manual(bloom_level="Appy"), SUBJECT, None)
        self.assertTrue(once["needs_review"])
        self.assertEqual(once["review_reason"],
                         "manual: topic set to T1; paper tags CO3; unknown Bloom level 'Appy'")
        self.assertEqual(R.resolve_record(once, SUBJECT, None), once)       # not added twice

    def test_splitters_choice_is_not_made_up_from_the_hand_set_topic(self):
        # topic_id_from_co means "what the splitter chose"; here that is no longer known
        self.assertNotIn("topic_id_from_co", R.resolve_record(self.manual(), SUBJECT, None))
        kept = R.resolve_record(self.manual(topic_id_from_co="T2"), SUBJECT, None)
        self.assertEqual(kept["topic_id_from_co"], "T2")

    def test_automatic_record_still_remembers_the_splitters_choice(self):
        meta = self.manual(tag_method=None, review_reason=None, topic_id="T2")
        new = R.resolve_record(meta, SUBJECT, {"T2": 0.3, "T3": 0.6})
        self.assertEqual((new["topic_id"], new["topic_id_from_co"]), ("T3", "T2"))


# ---------------------------------------------------------------- topic scores
class TestTopicScores(unittest.TestCase):
    def test_copies_of_one_text_count_once(self):
        # topic A: one matching page stored three times, the rest of the topic is unrelated
        # topic B: three different chunks that all match fairly well
        book = FakeCollection("book_content", dict([
            chunk("a1", "A", "the same page", [1.0, 0.0]),
            chunk("a2", "A", "the same page", [1.0, 0.0]),
            chunk("a3", "A", "the same page", [1.0, 0.0]),
            chunk("a4", "A", "unrelated one", [0.0, 1.0]),
            chunk("a5", "A", "unrelated two", [0.0, 1.0]),
            chunk("b1", "B", "good match one", [0.8, 0.6]),
            chunk("b2", "B", "good match two", [0.8, 0.6]),
            chunk("b3", "B", "good match three", [0.8, 0.6])]))
        scores = R.topic_scores([1.0, 0.0], ["A", "B", "EMPTY"], [book, FakeCollection("tut_content")])
        self.assertAlmostEqual(scores["A"], 1 / 3)          # was 1.0: three copies of one page
        self.assertAlmostEqual(scores["B"], 0.8)
        self.assertNotIn("EMPTY", scores)                   # a topic with no text has no score

    def test_chunks_flagged_for_review_are_not_used(self):
        book = FakeCollection("book_content", dict([
            chunk("a1", "A", "doubtful chunk", [1.0, 0.0], needs_review=True),
            chunk("a2", "A", "trusted chunk", [0.0, 1.0])]))
        self.assertAlmostEqual(R.topic_scores([1.0, 0.0], ["A"], [book])["A"], 0.0)


if __name__ == "__main__":
    unittest.main()
