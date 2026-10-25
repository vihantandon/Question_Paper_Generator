"""
The duplicate check (pipeline/dedup.py) and its place in the graph.

    python3 -m unittest tests.test_dedup -v      (from the repo root)

No Docker, no embedding model: the collections are FakeCollection from
tests/test_phase1.py with 2-D vectors.
"""
import contextlib
import io
import json
import sys
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tests.test_draft as T  # noqa: E402
import tests.test_phase1 as P  # noqa: E402
import tests.test_quality_check as QT  # noqa: E402
from pipeline import dedup as D  # noqa: E402
from pipeline.graph import accepted, build_graph, print_checks  # noqa: E402
from pipeline.resources import Resources  # noqa: E402


def embed(text):
    """Texts about postorder point one way (like the past paper below); the
    rest as in tests/test_phase1.py."""
    return [0.6, 0.8] if "postorder" in text.lower() else P.fake_embed(text)


def pyq(id_, subject, text, vec, type_="question"):
    return (id_, text, {"subject": subject, "type": type_, "topic_id": "DS_03", "marks": 5,
                        "bloom_level": "Apply"}, vec)


PAST = P.FakeCollection("pyq_bank", [
    pyq("DS_2024_endsem_q3", "DS", "Write the postorder traversal of the given binary tree.", [0.6, 0.8]),
    pyq("DS_2024_header", "DS", "Course outcomes ... postorder", [0.6, 0.8], type_="header"),
    pyq("ALGO_2025_q1", "ALGO", "A postorder walk of a recursion tree.", [0.6, 0.8]),
    pyq("DS_2024_test1_q5", "DS", "Reverse a linked list in place.", [-0.6, 0.8]),
])
TUTORIALS = P.FakeCollection("tut_content", [
    ("t1__DS_03", "Q2. Insert 4, 2, 6 into an empty BST and draw the tree.", {"subject": "DS"}, [-1.0, 0.0]),
    ("t1__DS_01", "Q2. Insert 4, 2, 6 into an empty BST and draw the tree.", {"subject": "DS"}, [-1.0, 0.0]),
])


class FakeBank(P.FakeCollection):
    """The question bank: also upsert / get / delete / count."""

    def __init__(self, rows=()):
        super().__init__("generated_questions", list(rows))

    def upsert(self, ids, documents, metadatas, embeddings):
        for row in zip(ids, documents, metadatas, embeddings):
            self.rows = [r for r in self.rows if r[0] != row[0]] + [row]

    def count(self):
        return len(self.rows)

    def delete(self, ids):
        self.rows = [r for r in self.rows if r[0] not in ids]

    def update(self, ids, metadatas):
        new = dict(zip(ids, metadatas))
        self.rows = [(i, t, new.get(i, m), v) for i, t, m, v in self.rows]


# ---------------------------------------------------------------- the measures
class TestTheLimit(unittest.TestCase):
    def test_the_default_sits_between_real_different_questions_and_a_rewritten_one(self):
        # --calibrate, 21 Oct: different real questions at most 0.72; rod cutting with new numbers 0.81
        self.assertTrue(0.72 < D.DEFAULT_SIMILARITY < 0.81)


class TestMeasures(unittest.TestCase):
    def test_wording_overlap_is_the_share_of_3_word_phrases_copied(self):
        old = "Insert the keys 4, 2, 6 into an empty binary search tree and draw it."
        self.assertEqual(D.overlap(old, old), 1.0)
        self.assertGreater(D.overlap("Insert the keys 4, 2, 6 into an empty binary search tree.", old), 0.8)
        self.assertLess(D.overlap("Delete the root of the AVL tree and rebalance it.", old), 0.2)
        self.assertEqual(D.overlap("two words", old), 0.0)

    def test_distance_to_similarity_and_subject(self):
        self.assertEqual((D.similarity(0.0), D.similarity(2.0), D.similarity(0.24)), (1.0, 0.0, 0.88))
        self.assertEqual((D.subject_of("ALGO_06"), D.subject_of("SDF1_02")), ("ALGO", "SDF1"))


# ---------------------------------------------------------------- the check
def look(bank=None):
    return D.places(PAST, TUTORIALS, bank)


class TestCheckDuplicates(unittest.TestCase):
    def test_the_same_question_in_other_words_is_too_close_in_meaning(self):
        r = D.check_duplicates("Give the postorder sequence of this tree.", "DS", embed, look())
        self.assertFalse(r["passed"])
        self.assertEqual([(d["place"], d["id"], d["similarity"]) for d in r["duplicates"]],
                         [("past paper", "DS_2024_endsem_q3", 1.0)])
        self.assertIn("meaning 1.00", r["duplicates"][0]["why"])
        self.assertEqual(r["looked_in"], ["past paper", "tutorial"])

    def test_only_the_same_subject_and_only_questions_count(self):
        r = D.check_duplicates("Give the postorder sequence of this tree.", "DS", embed, look())
        ids = {d["id"] for d in r["duplicates"]}
        self.assertNotIn("ALGO_2025_q1", ids)         # another subject
        self.assertNotIn("DS_2024_header", ids)       # a paper's header, not a question

    def test_copied_wording_is_too_close_even_when_the_meaning_score_is_not(self):
        text = "Insert 4, 2, 6 into an empty BST and draw the tree."
        r = D.check_duplicates(text, "DS", lambda t: [0.0, -1.0], look())
        self.assertEqual([d["id"] for d in r["duplicates"]], ["t1__DS_03"])     # the same text once
        self.assertEqual(r["duplicates"][0]["why"], "100% of the wording")

    def test_a_new_question_passes_and_its_closest_matches_are_kept(self):
        r = D.check_duplicates("Write a function that counts the leaves.", "DS", lambda t: [0.8, -0.6], look())
        self.assertTrue(r["passed"])
        self.assertEqual([(c["place"], c["id"]) for c in r["closest"]],
                         [("past paper", "DS_2024_endsem_q3"), ("tutorial", "t1__DS_03")])
        self.assertLess(max(c["similarity"] for c in r["closest"]), D.DEFAULT_SIMILARITY)

    def test_an_empty_bank_is_not_queried(self):
        bank = FakeBank()
        bank.query = mock.Mock(side_effect=AssertionError("queried an empty collection"))
        D.check_duplicates("anything", "DS", embed, look(bank))

    def test_the_writer_is_told_what_it_is_too_close_to(self):
        r = D.check_duplicates("Give the postorder sequence of this tree.", "DS", embed, look())
        text = D.feedback(r)
        self.assertIn('too close to a past paper question (DS_2024_endsem_q3; same meaning 1.00): '
                      '"Write the postorder traversal of the given binary tree."', text)
        self.assertIn("not the same question in other words or with new numbers", text)


# ---------------------------------------------------------------- the question bank
def accepted_result(question="Question two about inorder."):
    draft = T.reply(question=question)
    draft.update(marks=5, model="fake-model", answer_outline=["inorder: 2 4 6"])
    return {"draft": draft, "current_topic": {"id": "DS_03", "name": "Trees", "semester": 3},
            "target_difficulty": "hard", "classification": {"co": "CO4", "bloom": "Apply"}}


class TestQuestionBank(unittest.TestCase):
    def test_an_accepted_question_is_saved_and_the_next_one_is_checked_against_it(self):
        bank = FakeBank()
        when = datetime(2026, 10, 21, 13, 30, tzinfo=timezone.utc)
        saved = D.save_question(bank, embed, accepted_result("Write the postorder of the tree."), now=when)
        self.assertEqual(saved, "DS_03_20261021T133000")
        (id_, text, meta, _), = bank.rows
        self.assertTrue(text.startswith("Write the postorder of the tree.\n(a) Draw the tree"))
        self.assertEqual({k: meta[k] for k in ("subject", "topic_id", "marks", "difficulty", "co", "bloom")},
                         {"subject": "DS", "topic_id": "DS_03", "marks": 5, "difficulty": "hard", "co": "CO4",
                          "bloom": "Apply"})
        self.assertEqual(json.loads(meta["answer_key"]), ["inorder: 2 4 6"])
        r = D.check_duplicates("Postorder of this tree, please.", "DS", embed, D.places(None, None, bank))
        self.assertEqual([(d["place"], d["id"]) for d in r["duplicates"]], [("earlier question", saved)])
        r = D.check_duplicates("Postorder of this tree, please.", "ALGO", embed, D.places(None, None, bank))
        self.assertTrue(r["passed"])                     # another subject's bank entries do not count


# ---------------------------------------------------------------- in the graph
LOOP_GOOD = QT.LOOP_GOOD
DRAFT_POSTORDER = T.reply(sources_used=["S1", "S3"], question="Write the postorder of the tree below.")
DRAFT_INORDER = T.reply(sources_used=["S1", "S3"], question="Question two about inorder.")


def run(llm, bank=None, **options):
    res = Resources(driver=P.FakeDriver(P.GRAPH), grounding_collections=[P.BOOK, TUTORIALS],
                    pyq_collection=PAST, embed_fn=embed, llm=llm, bank_collection=bank)
    options.setdefault("classify", False)
    return build_graph(res, similarity_limit=0.88, overlap_limit=0.5, **options).invoke(
        {"current_topic_id": "DS_03", "target_difficulty": "hard", "target_marks": 5})


class TestDedupInTheGraph(unittest.TestCase):
    def test_a_duplicate_is_rewritten_and_the_new_question_accepted(self):
        client, chat, _ = T.make_client(DRAFT_POSTORDER, LOOP_GOOD, DRAFT_INORDER, LOOP_GOOD)
        r = run(client)
        self.assertTrue(accepted(r))
        self.assertEqual(r["draft"]["question"], "Question two about inorder.")
        self.assertEqual([d["passed"] for d in r["dedup_reports"]], [False, True])
        self.assertEqual([d["check"] for d in r["dedup_reports"]], [1, 2])
        rewrite = chat.sent[2]["messages"][1]["content"]
        self.assertIn("YOUR PREVIOUS DRAFT WAS REJECTED", rewrite)
        self.assertIn("too close to a past paper question (DS_2024_endsem_q3", rewrite)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            print_checks(r)
        out = printed.getvalue()
        self.assertIn("----- draft 1 (rejected) -----\n  Write the postorder of the tree below.", out)
        self.assertIn("Check 1 (fake-model): PASS", out)
        self.assertIn("  Duplicate check: TOO CLOSE\n    past paper DS_2024_endsem_q3 (same meaning 1.00)", out)
        self.assertIn("  Duplicate check: new  (closest: past paper", out)
        self.assertIn("ACCEPTED after 2 draft(s)", out)

    def test_duplicates_every_time_are_rejected_after_max_drafts(self):
        client, chat, _ = T.make_client(DRAFT_POSTORDER, LOOP_GOOD, DRAFT_POSTORDER, LOOP_GOOD)
        r = run(client, max_drafts=2)
        self.assertFalse(accepted(r))
        self.assertEqual((r["draft_attempts"], len(chat.sent), r["dedup_passed"]), (2, 4, False))

    def test_a_question_already_in_the_bank_is_sent_back(self):
        bank = FakeBank()
        D.save_question(bank, embed, accepted_result())                      # "... inorder" saved earlier
        client, chat, _ = T.make_client(DRAFT_INORDER, LOOP_GOOD)
        r = run(client, bank=bank, max_drafts=1)
        self.assertFalse(accepted(r))
        self.assertEqual(r["dedup"]["duplicates"][0]["place"], "earlier question")

    def test_the_duplicate_check_can_be_left_out(self):
        client, chat, _ = T.make_client(DRAFT_POSTORDER, LOOP_GOOD)
        r = run(client, dedup=False)
        self.assertTrue(accepted(r))
        self.assertNotIn("dedup", r)

    def test_classify_runs_only_after_a_new_question(self):
        from tests.test_classify import labels
        client, chat, _ = T.make_client(DRAFT_POSTORDER, LOOP_GOOD, DRAFT_INORDER, LOOP_GOOD, labels())
        r = run(client, classify=True)
        self.assertTrue(accepted(r))
        self.assertEqual((r["classification"]["co"], len(chat.sent)), ("CO4", 5))


class TestRejectedDraftsAreShown(unittest.TestCase):
    def test_a_checked_draft_is_shown_when_the_next_draft_never_came(self):
        # 21 Oct: draft 1 (with 7 wrong sums) was never printed, because draft 2 failed to come
        d1 = T.reply(question="Draft one with wrong sums.")
        r = {"drafts": [d1], "draft": None, "draft_attempts": 2,
             "quality_reports": [{"model": "m", "passed": False, "claims": [], "problems": ["sums wrong"]}]}
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            print_checks(r)
        self.assertIn("----- draft 1 (rejected) -----\n  Draft one with wrong sums.", printed.getvalue())
        self.assertIn("REJECTED after 2 draft(s)", printed.getvalue())


# ---------------------------------------------------------------- the next question on the same topic
class TestTheNextQuestionGetsOtherMaterial(unittest.TestCase):
    """21 Oct: every ALGO_06 run got the same 7 sources (three about rod
    cutting), so every draft was rod cutting again and the duplicate check
    rejected all three. Now the bank remembers what the writer was given."""

    def test_the_bank_keeps_the_main_sources_the_writer_was_given(self):
        result = accepted_result()
        result["relevant_chunks"] = [{"id": "b1", "role": "main"}, {"id": "b2", "role": "main"},
                                     {"id": "b3", "role": "blend"}]
        _, _, meta = D.question_record(result)
        self.assertEqual(json.loads(meta["source_ids"]), ["b1", "b2"])

    def test_earlier_questions_on_the_topic_newest_first(self):
        bank = FakeBank()
        for when, question in ((datetime(2026, 10, 21, 9, tzinfo=timezone.utc), "Old rod cutting."),
                               (datetime(2026, 10, 21, 10, tzinfo=timezone.utc), "New rod cutting.")):
            result = dict(accepted_result(question), relevant_chunks=[{"id": "b1", "role": "main"}])
            D.save_question(bank, embed, result, now=when)
        other = dict(accepted_result("Another topic."), current_topic={"id": "DS_01", "name": "L", "semester": 3})
        D.save_question(bank, embed, other)
        got = D.earlier_questions(bank, "DS_03")
        self.assertEqual([q["text"].split("\n")[0] for q in got], ["New rod cutting.", "Old rod cutting."])
        self.assertEqual(got[0]["source_ids"], ["b1"])
        self.assertEqual(D.earlier_questions(FakeBank(), "DS_03"), [])
        self.assertEqual(D.earlier_questions(None, "DS_03"), [])

    def test_retrieval_takes_other_text_first_and_old_text_only_to_fill_up(self):
        import retrieval
        topic = {"id": "DS_03", "name": "Trees"}
        b1 = "A binary tree node has left and right children"
        got = retrieval.fetch_grounding_chunks([P.BOOK], P.fake_embed, topic, n_main=1, min_tutorial=0)
        self.assertEqual([c["id"] for c in got], ["b1"])
        got = retrieval.fetch_grounding_chunks([P.BOOK], P.fake_embed, topic, n_main=1, min_tutorial=0,
                                               avoid_texts={b1})
        self.assertEqual([c["id"] for c in got], ["b2"])
        got = retrieval.fetch_grounding_chunks([P.BOOK], P.fake_embed, topic, n_main=2, min_tutorial=0,
                                               avoid_texts={b1})
        self.assertEqual([c["id"] for c in got], ["b2", "b1"])               # nothing else left: b1 fills up

    def test_the_writer_sees_the_earlier_question_and_gets_other_sources(self):
        bank = FakeBank()
        earlier = dict(accepted_result("Write the inorder of the tree."),
                       relevant_chunks=[{"id": "b1", "role": "main"}])
        D.save_question(bank, embed, earlier)
        client, chat, _ = T.make_client(DRAFT_INORDER, LOOP_GOOD)
        r = run(client, bank=bank, max_drafts=1)
        main = [c["id"] for c in r["relevant_chunks"] if c["role"] == "main"]
        self.assertEqual(main[0], "b2")                                      # b1 was used last time
        self.assertEqual([q["id"] for q in r["earlier_questions"]], [bank.rows[0][0]])
        prompt = chat.sent[0]["messages"][1]["content"]
        self.assertIn("QUESTIONS ALREADY SET ON THIS TOPIC -- write a DIFFERENT problem", prompt)
        self.assertIn("- Write the inorder of the tree. (a) Draw the tree", prompt)

    def test_without_a_bank_nothing_changes(self):
        client, chat, _ = T.make_client(DRAFT_INORDER, LOOP_GOOD)
        r = run(client)
        self.assertEqual(r["earlier_questions"], [])
        self.assertNotIn("ALREADY SET", chat.sent[0]["messages"][1]["content"])


# ---------------------------------------------------------------- calibration and the command line
class TestCalibrate(unittest.TestCase):
    def test_pairs_of_real_questions_are_measured(self):
        past = FakeBank(PAST.rows)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            pyq_pairs, tut_pairs = D.calibrate(past, TUTORIALS)
        self.assertEqual([(round(p[0], 2), p[2], p[3]) for p in pyq_pairs],
                         [(0.28, "DS_2024_endsem_q3", "DS_2024_test1_q5")])    # each pair once
        self.assertEqual(len(tut_pairs), 3)                                    # every question, its closest
        self.assertIn("past paper <-> its closest other past paper (1 pairs)", out.getvalue())
        self.assertIn("0.80:  0", out.getvalue())


class TestCommandLine(unittest.TestCase):
    def client(self, bank):
        return types.SimpleNamespace(list_collections=lambda: ["generated_questions"],
                                     get_or_create_collection=lambda name: bank,
                                     delete_collection=lambda name: bank.rows.clear())

    def run_cli(self, bank, *args):
        chroma = types.SimpleNamespace(HttpClient=lambda host, port: self.client(bank))
        with mock.patch.dict(sys.modules, {"chromadb": chroma}), contextlib.redirect_stdout(io.StringIO()) as out:
            D.main(list(args))
        return out.getvalue()

    def test_list_remove_clear(self):
        bank = FakeBank()
        saved = D.save_question(bank, embed, accepted_result())
        self.assertIn(f"{saved}", self.run_cli(bank, "--list"))
        self.assertIn("1 question(s) in the bank", self.run_cli(bank, "--list", "--subject", "DS"))
        self.assertIn("0 question(s) in the bank", self.run_cli(bank, "--list", "--subject", "ALGO"))
        self.assertIn("removed 1 question(s); 0 left", self.run_cli(bank, "--remove", saved))
        D.save_question(bank, embed, accepted_result())
        self.assertIn("the question bank is empty", self.run_cli(bank, "--clear"))
        self.assertEqual(bank.rows, [])


if __name__ == "__main__":
    unittest.main()
