"""
Phase 2: a whole paper from a teacher's blueprint (pipeline/blueprint.py,
pipeline/paper.py).

    python3 -m unittest tests.test_paper -v      (from the repo root)

No Docker, no API key: fake Neo4j / Chroma from tests/test_phase1.py, the
fake model from tests/test_draft.py, the fake bank from tests/test_dedup.py.
"""
import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import tests.test_dedup as TD
import tests.test_draft as T
import tests.test_phase1 as P
import tests.test_quality_check as QT
from pipeline import blueprint as B
from tests.test_classify import labels as classify_reply
from pipeline import paper as PP
from pipeline.curriculum import load_subjects
from pipeline.graph import build_graph
from pipeline.resources import Resources

SUBJECTS = load_subjects()


def blueprint(*questions, total=None, **changes):
    qs = list(questions) or [{"topic": "DS_03", "marks": 5, "difficulty": "hard"},
                             {"topic": "DS_01", "marks": 5, "difficulty": "easy"}]
    bp = {"subject": "DS", "exam": "T1", "total_marks": total or sum(q["marks"] for q in qs), "time": "1 hour",
          "questions": qs}
    bp.update(changes)
    return bp


# ---------------------------------------------------------------- the past papers
class TestDefaultsFromThePapers(unittest.TestCase):
    def test_tests_are_20_marks_in_an_hour_the_end_semester_35_in_two(self):
        t1, t3 = B.exam_defaults("DS", "T1"), B.exam_defaults("DS", "T3")
        self.assertEqual((t1["total_marks"], t1["time"], t1["questions"]), (20, "1 hour", 6))
        self.assertEqual(sum(t1["marks"]), 20)
        self.assertEqual((t3["total_marks"], t3["time"]), (35, "2 hours"))
        self.assertEqual(B.exam_defaults("ALGO", "T3")["total_marks"], 35)        # no ALGO end-sem: all subjects'

    def test_without_papers_the_usual_values(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(B.exam_defaults("DS", "T2", d), dict(B.FALLBACK["T2"], marks=[]))
            self.assertEqual(B.topics_in_past_papers("DS", "T1", d), set())

    def test_topics_asked_in_past_papers_of_the_kind(self):
        self.assertTrue({"DS_01", "DS_02"} <= B.topics_in_past_papers("DS", "T1"))


# ---------------------------------------------------------------- step 2: the checks
class TestValidate(unittest.TestCase):
    def test_a_good_blueprint(self):
        errors, warnings, rows = B.validate(blueprint(), SUBJECTS)
        self.assertEqual(errors, [])
        self.assertEqual([(r["topic"], r["cos"]) for r in rows], [("DS_03", ["CO1", "CO4"]), ("DS_01", ["CO1", "CO2"])])
        self.assertEqual(warnings, ["none of the chosen topics is mapped to CO3, CO5: no question can test it"])

    def test_errors_stop_the_paper(self):
        bad = blueprint({"topic": "DS_99", "marks": 5, "difficulty": "tricky"},
                        {"topic": "DS_01", "marks": 0, "difficulty": "easy"}, total=20, exam="T4")
        errors, _, _ = B.validate(bad, SUBJECTS)
        text = "\n".join(errors)
        for words in ("exam must be one of T1, T2, T3", "Q1: topic 'DS_99' is not a DS topic",
                      "Q1: difficulty must be easy, medium or hard",
                      "Q2: marks must be a whole number above 0", "add up to 5 marks, not total_marks 20"):
            self.assertIn(words, text)
        self.assertEqual(B.validate({"subject": "XYZ"}, SUBJECTS)[0][0][:22], "subject must be one of")
        self.assertIn("questions: the paper has none", B.validate(blueprint(questions=[]), SUBJECTS)[0][0])

    def test_the_teacher_does_not_give_the_co(self):
        errors, warnings, _ = B.validate(blueprint({"topic": "DS_03", "marks": 5, "co": "CO3", "difficulty": "easy"}),
                                         SUBJECTS)
        self.assertEqual(errors, [])
        self.assertIn("Q1: the co line is ignored -- classify decides the CO", warnings[0])

    def test_the_preview(self):
        errors, warnings, rows = B.validate(blueprint(), SUBJECTS)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            B.print_preview(blueprint(), errors, warnings, rows, reusable={1: "DS_03_x"}, subjects=SUBJECTS)
        text = out.getvalue()
        self.assertIn("Data Structures -- T1   10 marks, 1 hour, 2 question(s)", text)
        self.assertIn("<- reuses DS_03_x", text)
        self.assertIn("1 question(s) can reuse an unused question from the bank (no tokens); 1 to generate", text)
        self.assertIn("Q1  DS_03        5  hard       CO1 or CO4", text)
        self.assertIn("decided by classify once it is written", text)
        self.assertIn("The plan is fine", text)


# ---------------------------------------------------------------- step 1: the form
class TestTheForm(unittest.TestCase):
    def test_question_by_question(self):
        answers = iter(["DS", "",            # subject, exam T1
                        "2", "", "",         # 2 questions, 20 marks, 1 hour
                        "DS_03", "8", "hard",              # Q1
                        "99", "1", "", ""])                # Q2: 99 is not a choice; DS_01, 12 marks left, medium
        shown = []
        bp = B.new_blueprint(read=lambda prompt: next(answers), show=shown.append, subjects=SUBJECTS)
        self.assertEqual(bp, {"subject": "DS", "exam": "T1", "total_marks": 20, "time": "1 hour", "questions": [
            {"topic": "DS_03", "marks": 8, "difficulty": "hard"},
            {"topic": "DS_01", "marks": 12, "difficulty": "medium"}]})
        self.assertEqual(B.validate(bp, SUBJECTS)[0], [])
        self.assertTrue(any("past T1 papers: 20 marks, 1 hour" in s for s in shown))
        self.assertFalse(any("CO" in s for s in shown if s.startswith("  ") and "CO" in s[:8]))
        self.assertTrue(any("DS_01" in s and "(asked in past T1 papers)" in s for s in shown))
        self.assertIn("  please choose 1-7", shown)

    def test_saved_and_loaded_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            first = B.blueprint_path(blueprint(), d)
            B.save_blueprint(blueprint(), first)
            self.assertEqual(B.load_blueprint(first), blueprint())
            self.assertEqual((first.name, B.blueprint_path(blueprint(), d).name), ("ds_t1.yaml", "ds_t1_2.yaml"))

    def test_the_example_blueprint_is_valid(self):
        bp = B.load_blueprint(Path(B.PAPERS_DIR) / "example_ds_t1.yaml")
        self.assertEqual(B.validate(bp, SUBJECTS)[0], [])


# ---------------------------------------------------------------- step 3: the paper
WHEN = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
CO4 = classify_reply(co="CO4", bloom="Apply")          # what classify answers
CO1 = classify_reply(co="CO1", bloom="Understand")
FIRST = QT.DRAFT_1                                      # about a tree
SECOND = T.reply(sources_used=["S1", "S3"], question="Question two about pointers.",   # far from it
                 parts=[{"label": "a", "text": "Explain a pointer to a node.", "marks": 2},
                        {"label": "b", "text": "Write code that swaps two pointers.", "marks": 3}])


def embed(text):
    """As tests/test_phase1.py, but bank questions about reversing point a third way."""
    return [-1.0, 0.0] if "reverse" in text.lower() else P.fake_embed(text)


def unused_question(bank, topic="DS_03", marks=5, difficulty="hard", co="CO4", text="Reverse the list in place."):
    result = TD.accepted_result(text)
    result["draft"].update(marks=marks, parts=[])           # no "tree" in it: far from the drafts below
    result["current_topic"] = {"id": topic, "name": "x", "semester": 3}
    result["target_difficulty"] = difficulty
    result["classification"] = {"co": co, "bloom": "Apply"}
    return TD.D.save_question(bank, embed, result, now=WHEN)


class Paper:
    """A paper made with fakes: two DS_03 hard 5-mark questions."""

    def __init__(self, *replies, bank=None, spec=None):
        self.client, self.chat, _ = T.make_client(*replies)
        self.bank = TD.FakeBank() if bank is None else bank
        self.res = Resources(driver=P.FakeDriver(P.GRAPH), grounding_collections=[P.BOOK], pyq_collection=None,
                             embed_fn=embed, llm=self.client, bank_collection=self.bank)
        self.graph = build_graph(self.res, classify=True, dedup=True, similarity_limit=0.78, overlap_limit=0.5)
        q = spec or {"topic": "DS_03", "marks": 5, "difficulty": "hard"}
        self.blueprint = blueprint(dict(q), dict(q))
        self.rows = B.validate(self.blueprint, SUBJECTS)[2]
        self.dir = tempfile.TemporaryDirectory()
        self.progress = Path(self.dir.name) / "ds_t1.json"
        self.log = []

    def make(self, **options):
        return PP.make_paper(self.blueprint, self.rows, self.res, self.graph, "ds_t1", self.progress,
                             subjects=SUBJECTS, log=self.log.append, now=WHEN, **options)


class TestMakePaper(unittest.TestCase):
    def test_an_unused_bank_question_is_reused_and_the_other_is_generated(self):
        p = Paper(QT.DRAFT_1, QT.LOOP_GOOD, CO4)
        saved = unused_question(p.bank)
        slots, stopped = p.make()
        self.assertIsNone(stopped)
        self.assertEqual([(s["source"], s["status"]) for s in slots.values()], [("bank", "done"), ("generated", "done")])
        self.assertEqual(slots[1]["bank_id"], saved)
        self.assertEqual(slots[1]["question"], "Reverse the list in place.")
        self.assertEqual(len(p.chat.sent), 3)                         # draft, check, classify
        self.assertIn("COURSE OUTCOMES", p.chat.sent[2]["messages"][1]["content"])
        used = {i: m["used_in"] for i, _, m, _ in p.bank.rows}
        self.assertEqual(used[saved], "ds_t1")
        self.assertEqual(used[slots[2]["bank_id"]], "ds_t1")           # the new one is in the bank, used
        self.assertEqual((slots[2]["co"], slots[2]["bloom"], slots[2]["difficulty"]), ("CO4", "Apply", "hard"))
        progress = json.loads(p.progress.read_text())
        self.assertEqual([s["status"] for s in progress["slots"]], ["done", "done"])

    def test_a_used_question_is_never_reused_and_the_slot_must_fit(self):
        bank = TD.FakeBank()
        unused_question(bank, marks=6)                                 # other marks
        unused_question(bank, difficulty="easy")                       # other difficulty
        unused_question(bank, co="")                                   # no CO label
        p = Paper(FIRST, QT.LOOP_GOOD, CO4, SECOND, QT.LOOP_GOOD, CO1, bank=bank)
        slots, _ = p.make()
        self.assertEqual([s["source"] for s in slots.values()], ["generated", "generated"])
        self.assertIsNone(TD.D.find_unused(p.bank, "DS_03", 5, "hard"))          # both new ones are used now

    def test_a_stopped_run_goes_on_and_a_done_question_is_not_made_again(self):
        p = Paper(QT.DRAFT_1, QT.LOOP_GOOD, CO4, T.ApiError(401, "bad key"))
        slots, stopped = p.make()
        self.assertIn("API key was rejected", stopped)
        self.assertEqual(list(slots), [1])
        p.client, p.chat, _ = T.make_client(SECOND, QT.LOOP_GOOD, CO1)
        p.res.llm = p.client
        p.graph = build_graph(p.res, classify=True, dedup=True, similarity_limit=0.78, overlap_limit=0.5)
        slots, stopped = p.make()
        self.assertIsNone(stopped)
        self.assertEqual(len(p.chat.sent), 3)                          # only Q2: draft, check, classify
        self.assertIn("Q1: done before (generated)", p.log)

    def test_a_changed_question_is_made_again_and_fresh_starts_over(self):
        p = Paper(FIRST, QT.LOOP_GOOD, CO4, SECOND, QT.LOOP_GOOD, CO1)
        p.make()
        self.assertEqual(PP.load_progress(p.progress, p.rows).keys(), {1, 2})
        changed = [dict(p.rows[0], marks=6), p.rows[1]]
        self.assertEqual(PP.load_progress(p.progress, changed).keys(), {2})

    def test_a_question_that_fails_is_recorded_and_the_paper_goes_on(self):
        p = Paper(FIRST, QT.LOOP_BAD, SECOND, QT.LOOP_GOOD, CO1)
        p.graph = build_graph(p.res, classify=True, dedup=True, max_drafts=1)
        slots, stopped = p.make()
        self.assertEqual((slots[1]["status"], slots[2]["status"]), ("failed", "done"))
        self.assertIn("every draft was rejected: These facts are wrong", slots[1]["reason"])
        self.assertEqual(PP.load_progress(p.progress, p.rows).keys(), {2})        # Q1 is tried again next run

    def test_the_paper_and_its_answer_key(self):
        p = Paper(QT.DRAFT_1, QT.LOOP_GOOD, CO4)
        unused_question(p.bank)
        slots, _ = p.make()
        paper = PP.render_paper(p.blueprint, slots, SUBJECTS)
        self.assertIn("# Data Structures -- T1 Examination", paper)
        self.assertIn("**Maximum Marks:** 10    **Maximum Time:** 1 hour", paper)
        self.assertIn("| CO4 | Demonstrate and implement various operations", paper)
        self.assertIn("**Q1.** Reverse the list in place.  **[5 marks] [CO4, Apply]**", paper)
        self.assertIn("**Q2.** Question one about postorder.  **[5 marks] [CO4, Apply]**", paper)
        self.assertIn("(a) Draw the tree after inserting 4, 2, 6.  [2]", paper)
        key = PP.render_key(p.blueprint, slots, SUBJECTS)
        self.assertIn("## Q1 (5 marks, DS_03 ", key)
        self.assertIn("reused from the question bank", key)
        self.assertIn("- inorder: 2 4 6", key)

    def test_check_command(self):
        with tempfile.TemporaryDirectory() as d:
            good, bad = Path(d) / "good.yaml", Path(d) / "bad.yaml"
            B.save_blueprint(blueprint(), good)
            B.save_blueprint(blueprint(total=99), bad)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(PP.check_command(good, show_reuse=False), 0)
                self.assertEqual(PP.check_command(bad, show_reuse=False), 1)
        self.assertIn("ERROR: the questions add up to 10 marks, not total_marks 99", out.getvalue())


class TestClassifyGivesTheCO(unittest.TestCase):
    def test_the_co_comes_from_classify_and_the_topic_wins(self):
        # the fake graph maps DS_03 to CO4 only: classify's CO1 is replaced by CO4, with a note
        p = Paper(FIRST, QT.LOOP_GOOD, CO1, SECOND, QT.LOOP_GOOD, CO4)
        slots, _ = p.make()
        self.assertEqual([(s["co"], s["bloom"]) for s in slots.values()], [("CO4", "Apply"), ("CO4", "Apply")])
        self.assertEqual(slots[1]["co_free"], "CO1")
        self.assertIn("the classifier would pick CO1, but DS_03 is mapped to CO4", slots[1]["notes"][0])
        saved = {m["co"] for _, _, m, _ in p.bank.rows}
        self.assertEqual(saved, {"CO4"})
        self.assertIn("note: the classifier would pick CO1", PP.render_key(p.blueprint, slots, SUBJECTS))

    def test_when_classify_gives_nothing_the_topics_co_is_used_with_a_note(self):
        p = Paper(FIRST, QT.LOOP_GOOD, "no", "no", "no", SECOND, QT.LOOP_GOOD, CO4)
        slots, _ = p.make()
        self.assertEqual((slots[1]["status"], slots[1]["co"], slots[1]["bloom"]), ("done", "CO1", "Understand"))
        self.assertIn("no CO from classify", slots[1]["notes"][0])
        self.assertIn("CO1, the topic's first CO, is used -- check it", slots[1]["notes"][0])

    def test_marks_per_co_and_the_cos_no_question_tests(self):
        slots = {1: {"status": "done", "co": "CO4", "marks": 5}, 2: {"status": "done", "co": "CO4", "marks": 3},
                 3: {"status": "failed", "marks": 2}}
        self.assertEqual(PP.co_coverage(slots, SUBJECTS["DS"]), ({"CO4": 8}, ["CO1", "CO2", "CO3", "CO5"]))


if __name__ == "__main__":
    unittest.main()
