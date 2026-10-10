"""
The candidate's own name / enrollment number (handwritten on the scanned
papers) must never be stored: not in the page files, not in the PYQ records.

    python3 -m unittest tests.test_pyq_pii -v      (from the repo root)

No Docker, no embedding model, no PyMuPDF and no API key needed: the PDF
library and the vision client are faked. The names and numbers below are
made up.
"""
import contextlib
import io
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "Extracting" / "Processed"
sys.path.insert(0, str(ROOT / "Extracting" / "pyqs_processing"))

from split_pyq_questions import redact_pii  # noqa: E402

PAPER = "Jaypee Institute of Information Technology, Noida\nT1 Examination, Odd 2024"
QUESTION = "Q1. Reverse a singly linked list. [CO2, 5 Marks]"


def page(*candidate_lines):
    """A first page: printed header, the candidate's lines, then a question."""
    return "\n".join([PAPER, *candidate_lines, "Max Marks: 20", QUESTION])


CLEAN = page()


class TestRedactPii(unittest.TestCase):
    def test_every_layout_found_in_the_papers(self):
        layouts = [
            ["Name Asha Rao Enrollment No. 21103045"],
            ["Name: Asha Rao                       Enrollment No. 21103045"],
            ["Name Asha Rao  ", "Enrollment No. 21103045"],
            ["Name: Asha (handwritten)  ", "Enrollment No.: 2110 3045 (handwritten)"],
            ["Name $\\underline{\\text{ Asha Rao }}$ Enrollment No. $\\underline{\\text{ 21103045 }}$"],
            ["Name Asha Rao Enrollment No. NJG132512"],      # OCR'd number with letters
        ]
        for lines in layouts:
            self.assertEqual(redact_pii(page(*lines)), CLEAN, lines)

    def test_other_labels(self):
        for line in ["Roll No.: 17", "Enrolment Number 21103045", "**Name:** Asha Rao",
                     "Student Name: Asha Rao", "| Name: Asha Rao | Enrollment No.: 21103045 |"]:
            self.assertEqual(redact_pii(page(line)), CLEAN, line)

    def test_blank_form_fields_are_kept(self):
        # nothing personal in them, and the old rule's side effect of
        # deleting any line that merely mentions "enrollment" is gone
        text = page("Name ________   Enrollment No. ________",
                    "Write your enrollment number on every sheet.")
        self.assertEqual(redact_pii(text), text)

    def test_question_text_is_never_dropped(self):
        body = "\n".join([
            "Q1. Create a class student having data members as",
            "name (string) and rollno (int). Every student has a",
            "Name and roll number of each student are stored in a file.",
            "Name the data structure used by breadth first search.",
            "Name Mangling",                       # a term, not a person (below Q1)
            "-Name(String)",                       # UML attribute
            "Roll No 1 to 10 are placed in group A.",
            "Enrollment number and marks are read from the keyboard.",
            "roll_no = 5;",
            "rollno = 101;",
            "| Name | Artist | Year |",
            "eid:123 eName: Rohit annualSalary: 1000000",
            "[CO5, 6 Marks]",
        ])
        text = PAPER + "\n" + body
        self.assertEqual(redact_pii(text), text)

    def test_header_lines_about_the_course_are_kept(self):
        text = "\n".join([PAPER, "Course Name: Data Structures", "Name of Course: Data Structures",
                          "Course Code: 15B11CI311", QUESTION])
        self.assertEqual(redact_pii(text), text)

    def test_number_on_its_own_line_is_removed_anywhere(self):
        # an id needs no context: a filled-in "Enrollment No." line is personal on any page
        self.assertEqual(redact_pii(QUESTION + "\nEnrollment No. 21103045\n(b) Explain."),
                         QUESTION + "\n(b) Explain.")

    def test_running_twice_changes_nothing_more(self):
        once = redact_pii(page("Name Asha Rao Enrollment No. 21103045"))
        self.assertEqual(redact_pii(once), once)


class TestCommittedDataIsClean(unittest.TestCase):
    """The data files in the repo, checked with the same rule. Fails if a
    paper is added (or re-extracted) with the candidate's details in it."""

    def test_page_files(self):
        files = sorted((PROCESSED / "pyq_pages").glob("*_pages.jsonl"))
        self.assertTrue(files, "no page files found")
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.assertEqual(redact_pii(row["text"]), row["text"],
                                     f"{f.name} page {row['page_num']}")

    def test_pyq_records(self):
        files = sorted((PROCESSED / "pyq_structured").glob("*_pyq.json"))
        self.assertTrue(files, "no PYQ files found")
        for f in files:
            for rec in json.loads(f.read_text(encoding="utf-8")):
                # headers must be clean; questions must not lose a line to the rule
                self.assertEqual(redact_pii(rec["document"]), rec["document"], rec["id"])


class FakePage:
    """Just enough of a PyMuPDF page for the text-layer path (no vision call)."""

    def __init__(self, text):
        self.text = text

    def get_text(self):
        return self.text

    def get_images(self, full=True):
        return []


class TestExtractorRedactsPages(unittest.TestCase):
    def test_page_text_is_redacted_before_it_is_stored(self):
        first = page("Name Asha Rao Enrollment No. 21103045")
        second = "Q2. Name the two traversal orders of a graph and compare them. [CO4, 5 Marks]"
        fitz = types.ModuleType("fitz")
        fitz.open = lambda path: [FakePage(first), FakePage(second)]
        openai = types.ModuleType("openai")
        openai.OpenAI = lambda **kwargs: None
        pil = types.ModuleType("PIL")
        pil.Image = None
        fakes = {"fitz": fitz, "openai": openai, "PIL": pil}

        with mock.patch.dict(sys.modules, fakes), \
                mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "not-a-real-key"}), \
                contextlib.redirect_stdout(io.StringIO()):
            sys.modules.pop("pdf_extract_full", None)
            import pdf_extract_full
            records = pdf_extract_full.process_pdf(str(ROOT / "PYQs" / "T1 Fake.pdf"))

        self.assertEqual([r["extraction_method"] for r in records], ["text_layer", "text_layer"])
        self.assertEqual(records[0]["text"], CLEAN)
        self.assertEqual(records[1]["text"], second)


if __name__ == "__main__":
    unittest.main()
