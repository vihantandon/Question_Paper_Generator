"""
Splitting a transcribed paper into questions (split_pyq_questions.py).

    python3 -m unittest tests.test_pyq_split -v      (from the repo root)

The splitter script itself is never run. Its functions are called on small
made-up papers, and on the real page files only to READ them: output goes to
a temporary folder, never to Extracting/Processed/pyq_structured.
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
sys.path.insert(0, str(ROOT / "Extracting" / "pyqs_processing"))

import resolve_pyq_labels as R  # noqa: E402
import split_pyq_questions as S  # noqa: E402

YAMLS = S.load_subject_yamls(sorted(str(p) for p in S.SUBJECTS_DIR.glob("*.y*ml")))


def quiet(fn, *args, **kwargs):
    """Run fn; return (result, what it printed)."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        result = fn(*args, **kwargs)
    return result, printed.getvalue()


def numbers(text):
    """The question number each block starts with."""
    return [int(S.QUESTION_START_RE.match(b).group(1)) for b in S.split_into_questions(text)[1]]


# ---------------------------------------------------------------- question markers
class TestQuestionMarkers(unittest.TestCase):
    def test_a_question_label_inside_a_sentence_does_not_start_a_question(self):
        # this made 4 blocks with two "q1" and two "q2" ids, and embed_pyqs.py then stopped
        paper = ("Course Title: Data Structures\n"
                 "Q1. Build one stack from two queues Q1 and Q2 and show push on it.\n"
                 "Q2. Now do the same with Q1 only. Is Q2 needed?\n")
        header, blocks = S.split_into_questions(paper)
        self.assertEqual(header, "Course Title: Data Structures\n")
        self.assertEqual(len(blocks), 2)
        self.assertIn("two queues Q1 and Q2", blocks[0])

    def test_a_note_about_a_question_does_not_start_one(self):
        # from a real page: the model listed the student's handwritten notes
        paper = ("Q1. Design a class.\n"
                 "### Handwritten Notes on the Page:\n"
                 "*   On the left of Q3: **DMA**\n"
                 "*   Under Q2: operator overloading\n"
                 "### Q2.\nOverload the + operator.\n"
                 "### Q3.\nCreate a Product class.\n")
        self.assertEqual(numbers(paper), [1, 2, 3])

    def test_every_way_a_question_is_numbered(self):
        for marker in ("Q1.", "Q1:", "Q.1", "Q. 1", "**Q1:**", "**Q1.**", "**Q1**.", "Q1)", "Q1-",
                       "Q1 -", "Q1", "`Q.1", "> Q1.", "## Q1.", "  Q1."):
            self.assertEqual(numbers(f"Header\n{marker} Explain a stack.\nQ2. Explain a queue.\n"),
                             [1, 2], marker)
        self.assertEqual(numbers("Q1(a) Define a stack.\nQ2(a) Define a queue.\n"), [1, 2])

    def test_things_that_are_not_a_question_start(self):
        for line in ("Queue Q1 holds the odd numbers.", "Q1and Q2 are queues.", "Q2.5 is the ratio.",
                     "*   Q3: a bullet in a list of notes", "See Q4. It is similar."):
            self.assertEqual(numbers(f"Q1. First question.\n{line}\nQ9. Last question.\n"), [1, 9], line)

    def test_a_number_that_already_started_a_question_stays_in_its_block(self):
        parts = "Q1(a) Define a stack.\nQ1(b) Define a queue.\nQ2. Compare them.\n"
        self.assertEqual(numbers(parts), [1, 2])
        self.assertIn("Q1(b) Define a queue.", S.split_into_questions(parts)[1][0])
        wrapped = "Q1. Use two queues.\nQ2. Here the queues\nQ1 and Q2 hold the two halves.\n"
        self.assertEqual(numbers(wrapped), [1, 2])

    def test_heading_above_the_question_line_is_dropped_but_a_lone_heading_is_kept(self):
        repeated = "Header\n### Q.1\n`Q.1 Add code in line 4.`\n### Q.2\n`Q.2 Sort the array.`\n"
        blocks = S.split_into_questions(S.drop_redundant_headings(repeated))[1]
        self.assertEqual(blocks, ["Q.1 Add code in line 4.`", "Q.2 Sort the array.`"])
        # the heading is the ONLY marker here: removing it used to lose questions 2 and 3
        lone = "Q1. Design a class.\n### Q2.\nOverload +.\n### Q3.\nCreate a Product class.\n"
        self.assertEqual(numbers(S.drop_redundant_headings(lone)), [1, 2, 3])
        self.assertEqual(S.split_into_questions(S.drop_redundant_headings(lone))[1][1], "Q2.\nOverload +.")


# ---------------------------------------------------------------- subject and year
class TestSubjectAndYear(unittest.TestCase):
    def test_sdf_one_followed_by_a_word_starting_with_i_is_not_sdf_two(self):
        header = "Course Title: Software Development Fundamentals-I Instructions: attempt all questions"
        self.assertEqual(S.detect_subject(header, YAMLS, filename="T1 SDF 1_pages"), "SDF1")
        self.assertEqual(S.detect_subject("Software Development Fundamentals - II Maximum Time", YAMLS), "SDF2")
        self.assertEqual(S.detect_subject("Course Title: Data\nStructures", YAMLS), "DS")

    def test_file_name_is_used_when_the_header_does_not_say(self):
        self.assertEqual(S.detect_subject("Algorithms and Project Solving", YAMLS, filename="T2_Aps_pages"),
                         "ALGO")
        self.assertIsNone(S.detect_subject("no course title here", YAMLS, filename="paper_pages"))

    def test_year_of_the_exam(self):
        for header, year in (
                ("End-Term Examination, ODD 2023", 2023),
                ("Session 2023-24\nEnd Term Examination, May 2024", 2024),     # was 2023
                ("Academic Year 2023-2024, Even Semester", 2024),
                ("T1 Examination 2023-24 (Odd Semester)", 2023),
                ("Date: 2024-05-17", 2024),                                    # a date, not a session
                ("Paper code 9920103045, Test-1", None),                       # digits inside a number
                ("Test-1 Examination", None)):
            self.assertEqual(S.detect_year(header), year, header)


# ---------------------------------------------------------------- [CO, Bloom, marks] tags
CODES = {"CO1": "CO1", "CO2": "CO2", "CO3": "CO3", "C109.4": "CO4"}


def tag(text):
    parsed = S.parse_marks_block(text, CODES)
    return parsed["co_short"], parsed["bloom_level"], parsed["marks"]


class TestMarksTags(unittest.TestCase):
    def test_layouts_found_in_the_ten_papers_read_as_before(self):
        for text, want in (
                ("[CO2(Applying), 6 Marks]", ("CO2", "Applying", 6)),
                ("[C109.4 (Applying), 5 Marks]", ("CO4", "Applying", 5)),
                ("[CO1 (Analyse level); 2+3 Marks]", ("CO1", "Analyse level", 5)),
                ("[CO3 (Applying), 2+2+1+1+1=7 Marks]", ("CO3", "Applying", 7)),
                ("[C109.4 (Apply L3), 1 + 1.5 + 1.5 + 2 Marks]", ("CO4", "Apply L3", 6)),
                ("[CO 1(Understanding), 3 Marks]", ("CO1", "Understanding", 3)),
                ("[C03, 6 Marks, Understand]", ("CO3", "Understand", 6)),
                ("[CO2(Creating), 4M]", ("CO2", "Creating", 4)),
                ("[CO1(Analyzing), 1.5M]", ("CO1", "Analyzing", 1.5)),
                ("[CO2, 6 Marks]", ("CO2", None, 6))):
            self.assertEqual(tag("Q1. Explain. " + text), want, text)

    def test_layouts_that_used_to_give_no_co_and_no_marks(self):
        for text, want in (
                ("[CO2, Apply, 5 Marks]", ("CO2", "Apply", 5)),        # level between CO and marks
                ("[5 Marks, CO2]", ("CO2", None, 5)),                  # marks first
                ("(CO2, 5 Marks)", ("CO2", None, 5)),                  # round brackets
                ("(CO2 (Applying), 5 Marks)", ("CO2", "Applying", 5)),
                ("[CO-2, 5M]", ("CO2", None, 5)),                      # hyphen in the code
                ("[CO2 5 Marks]", ("CO2", None, 5)),                   # no comma: 5, not 2 + 5
                ("[5 Marks]", (None, None, 5))):                       # no CO printed
            self.assertEqual(tag("Q1. Explain. " + text), want, text)

    def test_marks_each_leaves_the_total_empty_and_says_so(self):
        parsed = S.parse_marks_block("Q1. (a) Define X. (b) Define Y. [CO2, 5 marks each]", CODES)
        self.assertEqual((parsed["co_short"], parsed["marks"], parsed["marks_each"]), ("CO2", None, True))
        self.assertNotIn("each", parsed["clean_text"])

    def test_tag_is_taken_out_of_the_question_text_and_parts_are_added_up(self):
        parsed = S.parse_marks_block("Q4. (a) Predict the output. [CO2, Apply, 3 Marks]\n"
                                     "(b) Fix the bug. (CO2, 2 Marks)", CODES)
        self.assertEqual(parsed["marks"], 5)
        self.assertEqual(len(parsed["marks_breakdown"]), 2)
        self.assertEqual(parsed["clean_text"], "Q4. (a) Predict the output. \n(b) Fix the bug.")

    def test_brackets_that_are_not_tags_are_left_alone(self):
        for text in ("Q1. Fill a[10] and b[5 M] with zeros.",
                     "Q1. (CO2 carries 5 marks in total) Explain.",
                     "Q1. Explain. [Hint: this part is worth 5 Marks]",
                     "Q1. Each part (a) and (b) is for (5 marks).",
                     "Q1. Call swap(c1, c2) and print(x, 5)."):
            parsed = S.parse_marks_block(text, CODES)
            self.assertEqual((parsed["raw_co_code"], parsed["marks"]), (None, None), text)
            self.assertEqual(parsed["clean_text"], text)


# ---------------------------------------------------------------- output files
def paper(pdf, title="Data Structures", year="2024", question="Explain how a stack is built on an array."):
    text = (f"Jaypee Institute of Information Technology, Noida\nTest-1 Examination, ODD {year}\n"
            f"Course Title: {title}\nCourse Code: 15B11CI311\nMaximum Marks: 20\n\n"
            f"Q1. {question} [CO2, 5 Marks]\nQ2. Compare a queue and a deque in a table. [CO2, 5 Marks]\n")
    return json.dumps({"source_file": pdf, "page_num": 1, "text": text}) + "\n"


class TestOutputFiles(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.pages, self.out = Path(tmp.name) / "pages", Path(tmp.name) / "out"
        self.pages.mkdir()
        (self.pages / "T1 Dsa_pages.jsonl").write_text(paper("PYQs/T1 Dsa.pdf"), encoding="utf-8")

    def split(self, **kwargs):
        return quiet(S.process_all, self.pages, self.out, **kwargs)[1]

    def records(self, name):
        return json.loads((self.out / name).read_text(encoding="utf-8"))

    def fix_by_hand(self, name, question_id, **changes):
        records = self.records(name)
        for r in records:
            if r["id"] == question_id:
                r["metadata"].update(changes)
        (self.out / name).write_text(json.dumps(records, indent=2), encoding="utf-8")

    def test_a_paper_that_is_already_split_is_not_touched(self):
        self.split()
        self.fix_by_hand("DS_2024_test1_pyq.json", "DS_2024_test1_q2", topic_id="DS_03")   # not marked manual
        before = (self.out / "DS_2024_test1_pyq.json").read_bytes()
        (self.pages / "T1 APS_pages.jsonl").write_text(
            paper("PYQs/T1 APS.pdf", title="Algorithms and Problem Solving"), encoding="utf-8")

        printed = self.split()                       # the run after a new paper was added
        self.assertIn("Skipping T1 Dsa_pages.jsonl", printed)
        self.assertEqual((self.out / "DS_2024_test1_pyq.json").read_bytes(), before)
        self.assertEqual(len(self.records("ALGO_2024_test1_pyq.json")), 3)        # header + 2 questions

    def test_all_splits_again_but_keeps_manual_records_exactly(self):
        self.split()
        self.fix_by_hand("DS_2024_test1_pyq.json", "DS_2024_test1_q1", topic_id="DS_02",
                         tag_method="manual", review_reason="manual: topic set to DS_02")
        self.fix_by_hand("DS_2024_test1_pyq.json", "DS_2024_test1_q2", topic_id="DS_03")
        manual_before = self.records("DS_2024_test1_pyq.json")[1]

        self.split(redo_all=True)
        records = self.records("DS_2024_test1_pyq.json")
        self.assertEqual(records[1], manual_before)                      # text, labels and note
        self.assertEqual(records[2]["metadata"]["topic_id"], "DS_01")     # rebuilt from the paper's CO
        self.assertEqual([r["id"] for r in records],
                         ["DS_2024_test1_header", "DS_2024_test1_q1", "DS_2024_test1_q2"])

    def test_a_manual_record_is_kept_even_if_the_new_split_has_no_such_question(self):
        self.split()
        self.fix_by_hand("DS_2024_test1_pyq.json", "DS_2024_test1_q2", tag_method="manual")
        (self.pages / "T1 Dsa_pages.jsonl").write_text(          # re-extracted: question 2 is gone
            paper("PYQs/T1 Dsa.pdf").replace("Q2. Compare", "Then compare"), encoding="utf-8")
        printed = self.split(redo_all=True)
        self.assertIn("DS_2024_test1_q2 is a manual record", printed)
        self.assertIn("DS_2024_test1_q2", [r["id"] for r in self.records("DS_2024_test1_pyq.json")])

    def test_two_papers_with_the_same_subject_year_and_exam_do_not_overwrite_each_other(self):
        (self.pages / "T1 Dsa makeup_pages.jsonl").write_text(
            paper("PYQs/T1 Dsa makeup.pdf", question="Explain how a queue is built on a linked list."),
            encoding="utf-8")
        printed = self.split()
        self.assertIn("this one is stored as DS_2024_test1_2", printed)
        first, second = self.records("DS_2024_test1_pyq.json"), self.records("DS_2024_test1_2_pyq.json")
        self.assertEqual({r[0]["metadata"]["source"] for r in (first, second)},      # both papers are there
                         {"PYQs/T1 Dsa.pdf", "PYQs/T1 Dsa makeup.pdf"})
        self.assertNotEqual(first[1]["document"], second[1]["document"])
        ids = [r["id"] for r in first + second]
        self.assertEqual(len(ids), len(set(ids)))                # embed_pyqs.py refuses duplicate ids
        # and a second run changes nothing: both are now "already split"
        self.assertEqual(self.split().count("Skipping"), 2)


# ---------------------------------------------------------------- the real papers (read only)
class TestRealPapers(unittest.TestCase):
    def test_a_normal_run_would_skip_all_committed_papers(self):
        done = quiet(S.papers_already_split, S.OUT_DIR)[0]
        for pages_file in sorted(S.PAGES_DIR.glob("*_pages.jsonl")):
            self.assertIn(S.paper_source(pages_file), done, pages_file.name)

    def test_real_page_files_split_into_the_committed_questions(self):
        done = quiet(S.papers_already_split, S.OUT_DIR)[0]
        for pages_file in sorted(S.PAGES_DIR.glob("*_pages.jsonl")):
            new = quiet(S.process_pages_file, pages_file, YAMLS)[0]
            committed = json.loads(done[S.paper_source(pages_file)].read_text(encoding="utf-8"))
            self.assertEqual([r["id"] for r in new], [r["id"] for r in committed], pages_file.name)
            for a, b in zip(new[1:], committed[1:]):
                self.assertEqual(a["metadata"]["marks"], b["metadata"]["marks"], a["id"])
                self.assertEqual(R.normalize_bloom(a["metadata"]["bloom_level"]),
                                 b["metadata"]["bloom_level"], a["id"])


# ---------------------------------------------------------------- label check
class TestQuestionWithoutMarksIsFlagged(unittest.TestCase):
    SUBJECT = {"cos": {"CO1": "one"}, "topics": {"T1": {"name": "Topic one", "co": ["CO1"]}}}

    def meta(self, **kw):
        return {"type": "question", "subject": "X", "raw_co_code": "CO1", "course_outcome": "CO1",
                "bloom_level": "Apply", "topic_id": "T1", "needs_review": False, "review_reason": None, **kw}

    def test_empty_marks_need_a_human(self):
        new = R.resolve_record(self.meta(marks=None), self.SUBJECT, {"T1": 0.5})
        self.assertTrue(new["needs_review"])
        self.assertIn("no marks", new["review_reason"])

    def test_marks_present_or_not_recorded_at_all_is_fine(self):
        self.assertFalse(R.resolve_record(self.meta(marks=5), self.SUBJECT, {"T1": 0.5})["needs_review"])
        self.assertFalse(R.resolve_record(self.meta(), self.SUBJECT, {"T1": 0.5})["needs_review"])


if __name__ == "__main__":
    unittest.main()
