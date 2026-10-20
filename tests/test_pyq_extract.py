"""
PYQ PDFs -> page texts (Extracting/pyqs_processing/pdf_extract_full.py).

    python3 -m unittest tests.test_pyq_extract -v      (from the repo root)

The script is never run and no model is called: PyMuPDF, the OpenRouter
client and PIL are stand-ins, and pages are made-up text. Output goes to a
temporary folder.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Extracting" / "pyqs_processing"))

import split_pyq_questions as S  # noqa: E402

PAGES_DIR = ROOT / "Extracting" / "Processed" / "pyq_pages"
GOOD = "Q1. Explain how a stack is built on an array, and show push and pop on it. [CO2, 5 Marks]"


def load_extractor():
    """Import pdf_extract_full with stand-ins for the libraries it loads."""
    fakes = {"fitz": types.ModuleType("fitz"), "openai": types.ModuleType("openai"),
             "PIL": types.ModuleType("PIL")}
    fakes["openai"].OpenAI = lambda **kwargs: None
    fakes["PIL"].Image = None
    with mock.patch.dict(sys.modules, fakes), \
            mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "not-a-real-key"}):
        sys.modules.pop("pdf_extract_full", None)
        import pdf_extract_full
    sys.modules.pop("pdf_extract_full", None)
    pdf_extract_full.page_to_pil_image = lambda page, dpi=200: None
    pdf_extract_full.pil_image_to_data_url = lambda img: "data:image/png;base64,AAAA"
    pdf_extract_full.DELAY_BETWEEN_CALLS = 0
    return pdf_extract_full


class ScannedPage:
    """A page with no text layer, so it goes to the vision model."""

    def get_text(self):
        return ""

    def get_images(self, full=True):
        return [1]


class FakeModel:
    """Stands in for the OpenRouter client: gives the next answer each call.
    An answer that is an Exception is raised instead."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **request):
        self.calls += 1
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        message = types.SimpleNamespace(content=answer)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def quiet(fn, *args, **kwargs):
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        result = fn(*args, **kwargs)
    return result, printed.getvalue()


# ---------------------------------------------------------------- what the model adds
CHATTY_PAGE = ("Here is the complete transcription of the page, preserving the printed text.\n\n---\n\n"
               "### Handwritten Notes on the Page:\n"
               "*   On the left of Q3: **DMA** (with an arrow)\n"
               "*   Under Q2: operator overloading\n\n---\n\n"
               "### Q2.\nOverload the + operator for a 2D vector.\n")


class TestModelChatter(unittest.TestCase):
    def test_opening_sentence_and_handwritten_notes_are_removed(self):
        self.assertEqual(S.strip_model_chatter(CHATTY_PAGE),
                         "### Q2.\nOverload the + operator for a 2D vector.\n")

    def test_a_clean_page_is_not_changed(self):
        for page in (GOOD, "POSSESSION OF MOBILES IN EXAM IS UFM PRACTICE\n\nQ1. Draw a flowchart.",
                     "Q3. Handwritten digit recognition uses which classifier? Explain.\n* k-NN\n* SVM",
                     "Q1. Here is the transcription of a DNA strand: ATGC. Find its complement.", ""):
            self.assertEqual(S.strip_model_chatter(page), page)

    def test_notes_about_a_question_no_longer_end_up_in_the_question_above(self):
        pages = [{"source_file": "PYQs/T1 X.pdf", "page_num": 1,
                  "text": "Course Title: Data Structures\nTest-1 Examination, 2024\n"
                          "Q1. Design a stack class. [CO2, 5 Marks]\n"},
                 {"source_file": "PYQs/T1 X.pdf", "page_num": 2,
                  "text": CHATTY_PAGE.replace("2D vector.", "2D vector. [CO2, 5 Marks]")}]
        yamls = S.load_subject_yamls(sorted(str(p) for p in S.SUBJECTS_DIR.glob("*.y*ml")))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "T1 X_pages.jsonl"
            path.write_text("".join(json.dumps(p) + "\n" for p in pages), encoding="utf-8")
            records = quiet(S.process_pages_file, path, yamls)[0]
        self.assertEqual([r["document"] for r in records[1:]],
                         ["Q1. Design a stack class.", "Q2.\nOverload the + operator for a 2D vector."])

    def test_verdicts_and_refusals_are_not_page_text(self):
        for answer in ("User Safety: safe", "  user safety : unsafe\nS1", "I'm sorry, but I can't help with that.",
                       "I cannot assist with this request.", "I can’t transcribe this image."):
            self.assertTrue(S.is_junk_page(answer), answer)
        for answer in (GOOD, "I. Answer all questions.", "Users of a safety system ..."):
            self.assertFalse(S.is_junk_page(answer), answer)


# ---------------------------------------------------------------- loading the file
class TestLoadsOnEveryPythonVersion(unittest.TestCase):
    """The tests below replace PIL with a stand-in whose Image is None. On
    Python 3.10-3.13 the hint "-> Image.Image" used to be read while the file
    loaded, so every test that loads the extractor crashed with
    "'NoneType' object has no attribute 'Image'" (Python 3.14 did not)."""

    def test_the_extractor_loads_with_a_stand_in_for_pil(self):
        X = load_extractor()
        self.assertTrue(callable(X.transcribe_with_vision))

    def test_type_hints_are_not_read_when_the_file_loads(self):
        path = ROOT / "Extracting" / "pyqs_processing" / "pdf_extract_full.py"
        self.assertIn("from __future__ import annotations", path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- one page
class TestTranscribeOnePage(unittest.TestCase):
    def setUp(self):
        self.X = load_extractor()
        patcher = mock.patch.object(self.X.time, "sleep", lambda seconds: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def transcribe(self, *answers):
        self.X.client = FakeModel(*answers)
        return quiet(self.X.transcribe_with_vision, ScannedPage())[0]

    def test_an_empty_answer_or_a_verdict_is_tried_again(self):
        # None crashed on .strip(); "User Safety: safe" was stored as the page's text for good
        self.assertEqual(self.transcribe(None, "User Safety: safe", GOOD), GOOD)
        self.assertEqual(self.X.client.calls, 3)

    def test_it_gives_up_with_an_error_instead_of_storing_junk(self):
        with self.assertRaises(ValueError) as stop:
            self.transcribe("User Safety: safe", "", None)
        self.assertIn("no transcription", str(stop.exception))

    def test_the_models_own_opening_line_is_not_stored(self):
        self.assertEqual(self.transcribe("Here is the transcription of the page:\n\n" + GOOD), GOOD)


# ---------------------------------------------------------------- one paper
class TestWholePaper(unittest.TestCase):
    def setUp(self):
        self.X = load_extractor()
        patcher = mock.patch.object(self.X.time, "sleep", lambda seconds: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.pdfs, self.out = Path(tmp.name) / "PYQs", Path(tmp.name) / "pages"
        self.pdfs.mkdir()
        (self.pdfs / "T1 Dsa.pdf").write_bytes(b"%PDF")
        self.X.fitz.open = lambda path: [ScannedPage(), ScannedPage(), ScannedPage()]

    def run_extractor(self, *answers):
        self.X.client = FakeModel(*answers)
        return quiet(self.X.process_all_pdfs, self.pdfs, self.out)[1]

    def rows(self, name):
        return [json.loads(line) for line in (self.out / name).read_text(encoding="utf-8").splitlines()]

    def test_record_names_the_paper_not_a_subject(self):
        self.run_extractor("page one " + GOOD, "page two " + GOOD, "page three " + GOOD)
        rows = self.rows("T1 Dsa_pages.jsonl")
        self.assertEqual([r["page_id"] for r in rows], ["T1 Dsa_p001", "T1 Dsa_p002", "T1 Dsa_p003"])
        self.assertEqual({r["paper"] for r in rows}, {"T1 Dsa"})      # was "subject", and it never was one
        self.assertNotIn("subject", rows[0])
        self.assertEqual([r["page_num"] for r in rows], [1, 2, 3])

    def test_a_failure_on_the_last_page_keeps_the_pages_before_it(self):
        down = RuntimeError("503 provider unavailable")
        printed = self.run_extractor("page one " + GOOD, "page two " + GOOD, down, down, down)
        self.assertIn("FAILED on T1 Dsa.pdf", printed)
        self.assertIn("2 page(s) already transcribed are kept", printed)
        self.assertFalse((self.out / "T1 Dsa_pages.jsonl").exists())        # the splitter sees no half paper
        self.assertEqual([r["page_num"] for r in self.rows("T1 Dsa_pages.jsonl.partial")], [1, 2])

        # next run: only page 3 is sent to the model (it used to start again from page 1)
        printed = self.run_extractor("page three " + GOOD)
        self.assertEqual(self.X.client.calls, 1)
        self.assertEqual(printed.count("kept from the last run"), 2)
        rows = self.rows("T1 Dsa_pages.jsonl")
        self.assertEqual([r["text"].split()[1] for r in rows], ["one", "two", "three"])
        self.assertFalse((self.out / "T1 Dsa_pages.jsonl.partial").exists())

    def test_a_page_that_only_ever_gives_a_verdict_leaves_the_paper_unfinished(self):
        printed = self.run_extractor("page one " + GOOD, "User Safety: safe", "User Safety: safe",
                                     "User Safety: safe")
        self.assertIn("FAILED on T1 Dsa.pdf", printed)
        self.assertFalse((self.out / "T1 Dsa_pages.jsonl").exists())
        self.assertEqual(len(self.rows("T1 Dsa_pages.jsonl.partial")), 1)

    def test_a_finished_paper_is_skipped(self):
        self.run_extractor("page one " + GOOD, "page two " + GOOD, "page three " + GOOD)
        printed = self.run_extractor()
        self.assertIn("Skipping T1 Dsa.pdf", printed)
        self.assertEqual(self.X.client.calls, 0)


# ---------------------------------------------------------------- the committed page files
class TestCommittedPageFiles(unittest.TestCase):
    def test_every_row_has_the_new_fields(self):
        files = sorted(PAGES_DIR.glob("*_pages.jsonl"))
        self.assertTrue(files)
        for path in files:
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.assertEqual(list(row), ["page_id", "paper", "source_file", "page_num",
                                             "extraction_method", "text"], path.name)
                self.assertEqual(row["paper"], Path(row["source_file"]).stem)
                self.assertEqual(row["page_id"], f"{row['paper']}_p{row['page_num']:03d}")

    def test_no_stored_page_is_a_verdict_or_a_refusal(self):
        for path in sorted(PAGES_DIR.glob("*_pages.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.assertFalse(S.is_junk_page(row["text"]), f"{path.name} page {row['page_num']}")
                self.assertGreater(len(row["text"].strip()), S.MIN_PAGE_TEXT_LEN)


if __name__ == "__main__":
    unittest.main()
