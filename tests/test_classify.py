"""
The classify step (pipeline/nodes/classify.py), the subject YAMLs it reads
(pipeline/curriculum.py) and its evaluation against the PYQ labels
(pipeline/eval_classify.py).

    python3 -m unittest tests.test_classify -v      (from the repo root)

No API key, no network, no Docker: the model is FakeChat from tests/test_draft.py.
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tests.test_draft as T  # noqa: E402
import tests.test_quality_check as QT  # noqa: E402
from pipeline import eval_classify as E  # noqa: E402
from pipeline import llm as L  # noqa: E402
from pipeline.curriculum import BLOOM_LEVELS, load_subjects, normalize_bloom, subject_of_topic  # noqa: E402
from pipeline.graph import accepted, print_classification  # noqa: E402
from pipeline.nodes import classify as C  # noqa: E402

SUBJECTS = load_subjects()
DS = SUBJECTS["DS"]


def labels(**changes):
    """What the classifying model sends back (as given: a CO4 question that reads as Apply)."""
    r = {"co": "CO4", "co_reason": "inserting into a BST is an operation on a non-linear DS",
         "bloom": "Apply", "bloom_reason": "the student carries out insertion on given keys"}
    r.update(changes)
    return r


# the labelled papers the examples come from
PYQS = [
    {"id": "p1", "subject": "DS", "text": "Explain abstract data types with an example.", "marks": 2,
     "label_co": "CO1", "label_bloom": "Understand"},
    {"id": "p2", "subject": "DS", "text": "Insert 4, 2, 6 into an empty binary search tree.", "marks": 3,
     "label_co": "CO4", "label_bloom": "Apply"},
    {"id": "p3", "subject": "DS", "text": "Delete the root of the AVL tree and rebalance it.", "marks": 4,
     "label_co": "CO4", "label_bloom": "Apply"},
    {"id": "p4", "subject": "DS", "text": "Find the number of comparisons quick sort makes on this array.",
     "marks": 5, "label_co": "CO3", "label_bloom": "Analyze"},
    {"id": "p5", "subject": "SDF1", "text": "Insert a node into a linked list using pointers.", "marks": 5,
     "label_co": "CO5", "label_bloom": "Apply"},
]


# ---------------------------------------------------------------- the subject YAMLs
class TestCurriculum(unittest.TestCase):
    def test_every_subject_gives_every_co_a_bloom_level(self):
        self.assertEqual(set(SUBJECTS), {"SDF1", "SDF2", "DS", "ALGO"})
        for s in SUBJECTS.values():
            self.assertEqual(set(s["co_levels"]), set(s["cos"]), s["id"])
            self.assertTrue(set(s["co_levels"].values()) <= set(BLOOM_LEVELS), s["id"])
        self.assertEqual(DS["co_levels"]["CO3"], "Analyze")
        self.assertEqual(SUBJECTS["ALGO"]["co_levels"]["CO4"], "Create")

    def test_a_topic_finds_its_subject(self):
        self.assertEqual(subject_of_topic(SUBJECTS, "DS_03")["id"], "DS")
        self.assertEqual(subject_of_topic(SUBJECTS, "SDF1_04")["topics"]["SDF1_04"]["cos"], ["CO5"])
        with self.assertRaises(KeyError):
            subject_of_topic(SUBJECTS, "XX_01")

    def test_bloom_spellings(self):
        for raw, level in (("Applying", "Apply"), ("Analyse L4", "Analyze"), ("create level", "Create"),
                           ("Understand (L2)", "Understand"), ("recall", "Remember"), ("Evaluating", "Evaluate")):
            self.assertEqual(normalize_bloom(raw), level, raw)
        self.assertIsNone(normalize_bloom("Level 3"))

    def test_a_wrong_level_in_a_yaml_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "x.yaml").write_text("subject: X\ncourse_outcomes:\n  CO1: Explain things\n"
                                            "co_levels:\n  CO1: Memorise\ntopics: []\n", encoding="utf-8")
            with self.assertRaises(ValueError) as e:
                load_subjects(d)
        self.assertIn("x.yaml: co_levels CO1", str(e.exception))


# ---------------------------------------------------------------- examples from the papers
class TestExamples(unittest.TestCase):
    def test_one_example_per_co_the_closest_one_and_never_the_question_itself(self):
        got = C.pick_examples("Insert 7 into the binary search tree.", "DS", PYQS)
        self.assertEqual([e["id"] for e in got], ["p1", "p4", "p2"])       # CO1, CO3, CO4 (p2 beats p3)
        got = C.pick_examples("Insert 4, 2, 6 into an empty binary search tree.", "DS", PYQS, exclude_id="p2")
        self.assertEqual([e["id"] for e in got], ["p1", "p4", "p3"])       # p2 is the question: left out
        got = C.pick_examples("Insert 4, 2, 6 into an  empty binary search tree.", "DS", PYQS)
        self.assertNotIn("p2", [e["id"] for e in got])                     # the same text counts as itself

    def test_examples_come_only_from_the_same_subject(self):
        self.assertEqual([e["id"] for e in C.pick_examples("Insert a node using pointers.", "SDF1", PYQS)], ["p5"])
        self.assertEqual(C.pick_examples("anything", "ALGO", PYQS), [])

    def test_the_real_papers_give_examples_for_every_subject(self):
        pyqs = C.load_pyqs()
        for sid, subject in SUBJECTS.items():
            got = C.pick_examples("Write a program.", sid, pyqs)
            self.assertEqual({e["label_co"] for e in got}, set(subject["cos"]), sid)


# ---------------------------------------------------------------- the prompt and the reply
class TestPromptAndReply(unittest.TestCase):
    def test_the_prompt_gives_every_co_examples_and_nothing_that_gives_the_answer_away(self):
        examples = C.pick_examples("Insert 7 into the binary search tree.", "DS", PYQS)
        system, user = C.build_messages("Insert 4, 2, 6 into an empty BST.", 5, DS, examples)
        text = user["content"]
        self.assertIn("COURSE: Data Structures (semester 3)", text)
        for co, words in DS["cos"].items():
            self.assertIn(f"{co}: {words}", text)
        self.assertIn("WITH THE CO THE EXAMINERS GAVE THEM", text)
        self.assertIn("- [CO4] (3 marks) Insert 4, 2, 6 into an empty binary search tree.", text)
        self.assertIn("QUESTION TO LABEL (5 marks)\nInsert 4, 2, 6 into an empty BST.", text)
        for level in BLOOM_LEVELS:
            self.assertIn(f"   {level} -- ", text)
        for gone in ("difficult", "routine", "insight", "hard"):         # the teacher sets the difficulty
            self.assertNotIn(gone, text.lower())
        for leak in ("DS_03", "Trees", "hard question", "Create CO", "co_levels", "Analyze]", "Apply]"):
            self.assertNotIn(leak, text)
        self.assertIn("ONE JSON object", system["content"])
        self.assertNotIn("WITH THE CO", C.build_messages("Q", 5, DS)[1]["content"])

    def test_a_long_example_is_cut(self):
        long = dict(PYQS[0], text="word " * 200)
        text = C.build_messages("Q", 5, DS, [long])[1]["content"]
        line = next(x for x in text.split("\n") if x.startswith("- [CO1]"))
        self.assertLess(len(line), C.EXAMPLE_CHARS + 30)
        self.assertTrue(line.endswith(" ..."))

    def test_a_reply_is_read_into_a_fixed_shape(self):
        got = C.parse_classification(labels(co="DS:CO 4", bloom="Applying"), list(DS["cos"]))
        self.assertEqual((got["co"], got["bloom_read"]), ("CO4", "Apply"))
        self.assertEqual(set(got), {"co", "co_reason", "bloom_read", "bloom_reason"})

    def test_an_unknown_label_is_sent_back_with_the_choices(self):
        for bad, words in ((labels(co="CO9"), '"co" must be one of CO1, CO2, CO3, CO4, CO5'),
                           (labels(co=None), '"co" must be one of'),
                           (labels(bloom="Level 3"), '"bloom" must be one of Remember, Understand')):
            with self.assertRaises(ValueError) as e:
                C.parse_classification(bad, list(DS["cos"]))
            self.assertIn(words, str(e.exception))

    def test_the_model_is_asked_again_when_its_co_does_not_exist(self):
        client, chat, _ = T.make_client(labels(co="CO7"), labels())
        got = C.classify(client, "Insert 4 into the BST.", 5, DS)
        self.assertEqual((got["co"], got["co_level"], got["model"], len(chat.sent)),
                         ("CO4", "Apply", "fake-model", 2))
        self.assertIn("must be one of CO1", chat.sent[1]["messages"][-1]["content"])
        self.assertEqual(chat.sent[0]["temperature"], 0.0)
        self.assertTrue(got["co_text"].startswith("Demonstrate and implement various operations"))


# ---------------------------------------------------------------- the labels made from the answer
class TestBloomLabel(unittest.TestCase):
    def test_the_label_is_the_level_of_the_chosen_co_and_the_reading_is_kept(self):
        got = C.with_course_details({"co": "CO3", "bloom_read": "Apply"}, DS)
        self.assertEqual((got["bloom"], got["bloom_source"], got["bloom_read"]), ("Analyze", "co level", "Apply"))

    def test_without_a_co_level_the_reading_is_the_label(self):
        subject = dict(DS, co_levels={})
        got = C.with_course_details({"co": "CO3", "bloom_read": "Apply"}, subject)
        self.assertEqual((got["bloom"], got["bloom_source"], got["co_level"]), ("Apply", "question", None))


# ---------------------------------------------------------------- the node
def accepted_state(**changes):
    s = QT.draft_state(quality_passed=True)
    s.update(changes)
    return s


class TestClassifyNode(unittest.TestCase):
    def node(self, client):
        return C.make_classify_node(client, SUBJECTS, PYQS)

    def test_labels_and_notes(self):
        client, chat, _ = T.make_client(labels(bloom="Analyze"))
        out = self.node(client)(accepted_state())["classification"]
        self.assertEqual((out["co"], out["bloom"], out["bloom_read"], out["difficulty"], out["co_level"]),
                         ("CO4", "Apply", "Analyze", "hard", "Apply"))      # hard: as the teacher asked
        self.assertEqual((out["co_free"], out["co_source"]), ("CO4", "model"))
        self.assertEqual(out["notes"], [
            "the paper would print Apply (CO4's level), but the question itself reads as Analyze"])
        prompt = chat.sent[0]["messages"][1]["content"]
        self.assertIn("QUESTION TO LABEL (5 marks)\nGiven a BST stored with pointers, answer the following.\n"
                      "(a) Draw it.   [2 marks]", prompt)
        self.assertIn("- [CO4] (3 marks) Insert 4, 2, 6 into an empty binary search tree.", prompt)

    def test_labels_that_fit_give_no_notes(self):
        client, _, _ = T.make_client(labels())
        out = self.node(client)(accepted_state())["classification"]
        self.assertEqual((out["bloom"], out["difficulty"], out["notes"]), ("Apply", "hard", []))

    def test_no_draft_no_request(self):
        client, chat, _ = T.make_client()
        self.assertEqual(self.node(client)(accepted_state(draft=None)), {"classification": None})
        self.assertEqual(chat.sent, [])

    def test_a_model_that_gives_no_labels_does_not_lose_the_question(self):
        client, _, _ = T.make_client("no", "no", "no")
        out = self.node(client)(accepted_state())["classification"]
        self.assertIn("error", out)

    def test_wrong_settings_stop_the_run(self):
        client, _, _ = T.make_client(T.ApiError(401, "bad key"))
        with self.assertRaises(L.LLMFatalError):
            self.node(client)(accepted_state())

    def test_the_printout(self):
        client, _, _ = T.make_client(labels(bloom="Create"))
        r = accepted_state(**self.node(client)(accepted_state()))
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            print_classification(r)
        out = printed.getvalue()
        self.assertIn("Classification (fake-model):\n  CO          CO4  Demonstrate and implement", out)
        self.assertIn("  Bloom       Apply  (CO4's level, as the papers print it; the question itself reads "
                      "as Create)\n              why Create: the student carries out", out)
        self.assertIn("  Difficulty  hard  (as asked)", out)
        self.assertNotIn("classifier's answers", out)


class TestTheTopicDecidesTheCO(unittest.TestCase):
    """A generated question was written for a topic: when the model picks a
    CO the topic is not mapped to, the topic's CO is used."""

    def node(self, client):
        return C.make_classify_node(client, SUBJECTS, PYQS)

    def test_a_topic_with_one_co_gets_that_co(self):
        # the ALGO_06 run of 21 Oct: dynamic programming, the model said CO3
        client, _, _ = T.make_client(labels(co="CO3", bloom="Create"))
        state = accepted_state(current_topic={"id": "ALGO_06", "name": "Dynamic Programming", "semester": 4},
                               topic_cos=[{"co_id": "ALGO:CO4", "short_id": "CO4"}])
        out = self.node(client)(state)["classification"]
        self.assertEqual((out["co"], out["co_free"], out["co_source"], out["bloom"], out["co_level"]),
                         ("CO4", "CO3", "topic", "Create", "Create"))
        self.assertTrue(out["co_text"].startswith("Identify, formulate and design"))
        self.assertEqual(out["notes"][0], "the classifier would pick CO3, but ALGO_06 is mapped to CO4, so CO4 "
                                          "is used: check that the question kept to its topic")
        self.assertEqual(len(out["notes"]), 1)              # Create reads as Create; hard is hard
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            print_classification(dict(state, classification=out))
        self.assertIn("  CO          CO4  Identify, formulate and design", printed.getvalue())
        self.assertIn("              (the topic's CO; the classifier picked CO3)\n              why CO3: ",
                      printed.getvalue())

    def test_a_topic_with_several_cos_gets_the_one_whose_level_fits_the_question(self):
        cos = [{"co_id": "DS:CO1", "short_id": "CO1"}, {"co_id": "DS:CO4", "short_id": "CO4"}]
        for read, want in (("Understand", "CO1"), ("Apply", "CO4"), ("Create", "CO4"), ("Remember", "CO1")):
            client, _, _ = T.make_client(labels(co="CO5", bloom=read))
            out = self.node(client)(accepted_state(topic_cos=cos))["classification"]
            self.assertEqual((out["co"], out["co_free"]), (want, "CO5"), read)

    def test_a_pick_the_topic_is_mapped_to_is_kept(self):
        cos = [{"co_id": "DS:CO1", "short_id": "CO1"}, {"co_id": "DS:CO4", "short_id": "CO4"}]
        client, _, _ = T.make_client(labels(co="CO1", bloom="Apply"))
        out = self.node(client)(accepted_state(topic_cos=cos))["classification"]
        self.assertEqual((out["co"], out["co_source"]), ("CO1", "model"))

    def test_without_topic_cos_the_yaml_mapping_is_used(self):
        client, _, _ = T.make_client(labels(co="CO2"))
        out = self.node(client)(accepted_state(topic_cos=[]))["classification"]
        self.assertEqual((out["co"], out["topic_cos"]), ("CO4", ["CO1", "CO4"]))   # ds.yaml: DS_03 -> CO1, CO4


# ---------------------------------------------------------------- in the graph
class TestClassifyInTheGraph(unittest.TestCase):
    def test_an_accepted_question_is_classified_by_the_checking_model(self):
        writer, written, _ = T.make_client(QT.DRAFT_1)
        checker, checked, _ = T.make_client(QT.LOOP_GOOD, labels())
        r = QT.run(writer, checker, classify=True)
        self.assertTrue(accepted(r))
        self.assertEqual((len(written.sent), len(checked.sent)), (1, 2))
        self.assertEqual(r["classification"]["co"], "CO4")
        prompt = checked.sent[1]["messages"][1]["content"]
        self.assertIn("COURSE OUTCOMES", prompt)
        self.assertIn("WITH THE CO THE EXAMINERS GAVE THEM", prompt)       # the real papers

    def test_a_rejected_question_is_not_classified(self):
        client, chat, _ = T.make_client(QT.DRAFT_1, QT.LOOP_BAD)
        r = QT.run(client, classify=True, max_drafts=1)
        self.assertFalse(accepted(r))
        self.assertNotIn("classification", r)
        self.assertEqual(len(chat.sent), 2)

    def test_without_the_check_the_first_draft_is_classified(self):
        client, chat, _ = T.make_client(QT.DRAFT_1, labels())
        r = QT.run(client, check=False, classify=True)
        self.assertEqual((r["classification"]["bloom"], len(chat.sent)), ("Apply", 2))

    def test_classify_can_be_left_out(self):
        client, chat, _ = T.make_client(QT.DRAFT_1, QT.LOOP_GOOD)
        r = QT.run(client, classify=False)
        self.assertTrue(accepted(r))
        self.assertNotIn("classification", r)


# ---------------------------------------------------------------- the rule-based stand-in
class TestKeywordClassify(unittest.TestCase):
    def test_bloom_read_from_the_highest_verb(self):
        for text, level in (("Design an efficient algorithm to merge the files and analyse it.", "Create"),
                            ("Explain the difference between a stack and a queue.", "Understand"),
                            ("Write a program to reverse a linked list.", "Apply"),
                            ("Find the time complexity of the following code.", "Analyze"),
                            ("Define an abstract data type.", "Remember"),
                            ("Hello there.", "Apply")):
            self.assertEqual(C.keyword_classify(text, DS)["bloom_read"], level, text)

    def test_co_from_shared_words(self):
        got = C.keyword_classify("Analyse the performance of quick sort and binary searching.", DS)
        self.assertEqual((got["co"], got["bloom"], got["model"]), ("CO3", "Analyze", "keywords"))


# ---------------------------------------------------------------- the evaluation
def row(id_, label_co, co, label_bloom, bloom_read, subject="DS", **extra):
    level = SUBJECTS[subject]["co_levels"][co]
    return dict({"id": id_, "subject": subject, "marks": 5, "label_co": label_co, "co": co,
                 "label_bloom": label_bloom, "bloom": level, "bloom_read": bloom_read,
                 "text": f"text of {id_}", "prompt_version": C.PROMPT_VERSION}, **extra)


class TestScores(unittest.TestCase):
    def test_kappa(self):
        # 2x2 table: 20 agree-yes, 15 agree-no, 5 + 10 disagree -> po 0.7, pe 0.5 -> 0.4
        pairs = [("y", "y")] * 20 + [("y", "n")] * 5 + [("n", "y")] * 10 + [("n", "n")] * 15
        self.assertAlmostEqual(E.kappa(pairs), 0.4)
        self.assertEqual(E.kappa([("a", "a"), ("b", "b")]), 1.0)
        self.assertIsNone(E.kappa([("a", "a")] * 3))
        self.assertIsNone(E.kappa([]))

    def test_score(self):
        rows = [row("q1", "CO3", "CO3", "Analyze", "Analyze"),
                row("q2", "CO2", "CO5", "Apply", "Analyze", marks=8),
                row("q3", "CO1", "CO1", "Understand", "Create"),
                row("q4", "CO5", "CO5", "Create", "Create", subject="SDF2"),
                row("q5", "CO1", "CO1", "Apply", "Apply", error="no reply")]
        s = E.score(rows, SUBJECTS)
        self.assertEqual((s["n"], s["co_right"]), (4, 3))
        # label = chosen CO's level: q1 Analyze ok, q2 Create (CO5) wrong, q3 Understand ok, q4 Create ok
        self.assertEqual(s["bloom_right"], 3)
        self.assertEqual((s["read_right"], s["read_near"]), (2, 3))
        self.assertEqual((s["labels_are_co_level"], s["labels_with_co_level"]), (4, 4))
        self.assertEqual(s["by_subject"]["DS"], {"n": 3, "co_right": 2, "bloom_right": 2})
        self.assertEqual([r["id"] for r in s["wrong_co"]], ["q2"])
        self.assertEqual(s["confusion"][("Understand", "Create")], 1)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            E.print_report("fake-model", s, 5)
        out = printed.getvalue()
        self.assertRegex(out, r"\n  CO +3/4 +75% +0\.69\n  Bloom label +3/4 +75%")
        self.assertRegex(out, r"\n  Bloom read +2/4 +50%")
        self.assertNotIn("Difficulty", out)
        self.assertIn("On these papers, 4 of 4 Bloom labels are exactly their CO's level.", out)
        self.assertIn('q2                       paper CO2, got CO5   "text of q2"', out)


PYQ_FILE = [
    {"id": "DS_2024_test1_header", "document": "Course Title: Data Structures", "metadata": {"type": "header"}},
    {"id": "DS_2024_test1_q1", "document": "Q.1 Explain an ADT. [CO1]",
     "metadata": {"type": "question", "subject": "DS", "course_outcome": "CO1", "bloom_level": "Understand",
                  "marks": 2}},
    {"id": "DS_2024_test1_q2", "document": "**Q2.** Sort 3, 1, 2 with insertion sort.",
     "metadata": {"type": "question", "subject": "DS", "course_outcome": "CO3", "bloom_level": "Analyze",
                  "marks": 3}},
    {"id": "DS_2024_test1_q3", "document": "Q3) No labels on this one.",
     "metadata": {"type": "question", "subject": "DS", "course_outcome": None, "bloom_level": None}},
]


class TestEvaluationRun(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.folder = Path(self.dir.name)
        (self.folder / "DS_2024_test1_pyq.json").write_text(json.dumps(PYQ_FILE), encoding="utf-8")
        self.path = E.results_path("vendor/model:free", self.folder / "out")

    def test_questions_are_read_without_their_number_and_unlabelled_ones_are_skipped(self):
        qs = E.load_pyqs(self.folder)
        self.assertEqual([(q["id"], q["text"], q["marks"]) for q in qs],
                         [("DS_2024_test1_q1", "Explain an ADT. [CO1]", 2),
                          ("DS_2024_test1_q2", "Sort 3, 1, 2 with insertion sort.", 3)])
        self.assertEqual(C.clean_question("Question 4: Draw it."), "Draw it.")
        self.assertEqual(self.path.name, "classify_vendor-model-free.jsonl")

    def test_every_answer_is_saved_at_once_and_a_stopped_run_goes_on_later(self):
        qs = E.load_pyqs(self.folder)
        client, chat, _ = T.make_client(labels(co="CO1", bloom="Understand"),
                                        T.ApiError(429, "Rate limit exceeded: free-models-per-day"))

        def label(q):
            return C.classify(client, q["text"], q["marks"], DS)
        rows, stopped = E.label_all(qs, label, self.path, {}, log=lambda *a: None)
        self.assertEqual([r["id"] for r in rows], ["DS_2024_test1_q1"])
        self.assertIn("daily limit", stopped)
        saved = E.load_saved(self.path)
        self.assertEqual(list(saved), ["DS_2024_test1_q1"])

        client, chat, _ = T.make_client(labels(co="CO3", bloom="Analyze"))   # the next day: only q2 is asked
        rows, stopped = E.label_all(qs, label, self.path, saved, log=lambda *a: None)
        self.assertIsNone(stopped)
        self.assertEqual(len(chat.sent), 1)
        self.assertEqual(E.score(rows, SUBJECTS)["co_right"], 2)

    def test_a_failed_question_is_asked_again_next_time_and_old_prompts_are_not_reused(self):
        qs = E.load_pyqs(self.folder)[:1]
        client, _, _ = T.make_client("no", "no", "no")

        def label(q):
            return C.classify(client, q["text"], q["marks"], DS)
        rows, _ = E.label_all(qs, label, self.path, {}, log=lambda *a: None)
        self.assertEqual(rows, [])
        self.assertIn('"error"', self.path.read_text(encoding="utf-8"))
        self.assertEqual(E.load_saved(self.path), {})
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row("DS_2024_test1_q1", "CO1", "CO1", "Understand", "Understand",
                                   prompt_version=C.PROMPT_VERSION - 1)) + "\n")
        self.assertEqual(E.load_saved(self.path), {})

    def test_the_model_never_sees_the_question_it_is_scored_on_as_an_example(self):
        questions = E.load_pyqs()
        client, chat, _ = T.make_client(*[labels(co="CO1")] * 3)
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            original, E.RESULTS_DIR = E.RESULTS_DIR, Path(d)
            original_make, L.make_llm = L.make_llm, lambda **kw: client
            try:
                E.main(["--subject", "DS", "--limit", "3"])
            finally:
                E.RESULTS_DIR, L.make_llm = original, original_make
        self.assertEqual(len(chat.sent), 3)
        ds = [q for q in questions if q["subject"] == "DS"][:3]
        for sent, q in zip(chat.sent, ds):
            prompt = sent["messages"][1]["content"]
            examples = prompt.split("QUESTION TO LABEL")[0]
            self.assertIn("WITH THE CO THE EXAMINERS GAVE THEM", examples)
            self.assertNotIn(C._short(q["text"])[:120], examples, q["id"])

    def test_the_real_papers_load(self):
        qs = E.load_pyqs()
        self.assertGreaterEqual(len(qs), 50)
        self.assertTrue(all(q["subject"] in SUBJECTS and q["label_co"] in SUBJECTS[q["subject"]]["cos"]
                            for q in qs))
        self.assertFalse(any(q["text"].lower().startswith("q") and q["text"][1:2] in ".0123456789" for q in qs))


if __name__ == "__main__":
    unittest.main()
