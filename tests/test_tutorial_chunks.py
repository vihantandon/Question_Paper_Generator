"""
Tutorial sheets -> question chunks (Extracting/tutorial_chunks.py).

    python3 -m unittest tests.test_tutorial_chunks -v      (from the repo root)

The script itself is never run and Gemini is never called: its functions are
given made-up lines, a made-up PDF object and a stand-in Gemini client.
"""
import io
import json
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Extracting"))

for _name in ("pymupdf", "dotenv"):                 # only needed to open real PDFs / read .env
    try:
        __import__(_name)
    except ImportError:
        sys.modules[_name] = types.SimpleNamespace(load_dotenv=lambda *a, **k: None)

import tutorial_chunks as T  # noqa: E402

PAGE_HEIGHT = 800.0
SUBJECTS_DIR = ROOT / "prerequisite_graph" / "subjects"
TOPIC_MAP = ROOT / "Extracting" / "topic_map.yaml"


def setUpModule():
    logging.disable(logging.CRITICAL)


def tearDownModule():
    logging.disable(logging.NOTSET)


def line(text, page=0, y=300.0, x=50.0):
    return T.Line(page=page, x0=x, y0=y, y1=y + 12, text=text)


def question_lines(lines):
    """The text each detected question starts with."""
    markers = T.detect_question_markers(lines, T.body_left_margin(lines))
    return [lines[i].text for i in markers]


# ---------------------------------------------------------------- running headers
class TestRunningLines(unittest.TestCase):
    def pages(self):
        """Two pages: a header in the top margin, a footer in the bottom one,
        and a question with the same lines of code on both pages."""
        lines = []
        for page in (0, 1):
            lines += [line("JIIT Noida   Tutorial Sheet", page, y=30),
                      line(f"Q{page + 1}. Predict the output:", page, y=200),
                      line("int main()", page, y=220), line("{", page, y=240),
                      line("return 0;", page, y=260), line("}", page, y=280),
                      line("Odd Semester 2024", page, y=770)]
        return lines

    def kept(self, lines):
        running = T.find_running_lines(lines, 2, PAGE_HEIGHT)
        return [ln.text for ln in lines if not T.is_boilerplate_line(ln, PAGE_HEIGHT, running)]

    def test_code_repeated_on_every_page_is_kept(self):
        # "{", "}", "return 0;" ... were deleted from every question of a multi-page sheet
        kept = self.kept(self.pages())
        for code in ("int main()", "{", "return 0;", "}"):
            self.assertEqual(kept.count(code), 2, code)

    def test_header_and_footer_are_removed_even_with_capital_letters(self):
        # the raw text was looked up among lower-cased keys, so these were never removed
        kept = self.kept(self.pages())
        self.assertNotIn("JIIT Noida   Tutorial Sheet", kept)
        self.assertNotIn("Odd Semester 2024", kept)
        self.assertEqual(kept[0], "Q1. Predict the output:")

    def test_a_body_line_that_equals_the_header_text_stays(self):
        lines = self.pages() + [line("jiit noida tutorial sheet", 1, y=400)]
        self.assertIn("jiit noida tutorial sheet", self.kept(lines))

    def test_single_page_sheet_has_no_running_lines(self):
        self.assertEqual(T.find_running_lines(self.pages()[:7], 1, PAGE_HEIGHT), set())


# ---------------------------------------------------------------- question numbering
class TestQuestionNumbering(unittest.TestCase):
    def test_plain_numbering(self):
        lines = [line("Tutorial 5"), line("Q1. First"), line("some text"), line("Q2. Second"),
                 line("Q3. Third")]
        self.assertEqual(question_lines(lines), ["Q1. First", "Q2. Second", "Q3. Third"])

    def test_a_stray_big_number_does_not_swallow_the_questions_after_it(self):
        # "12." was accepted, and then Q4 .. Q10 were all rejected and glued into one question
        lines = [line(f"Q{n}. Question {n}") for n in (1, 2, 3)]
        lines.append(line("12. marks will be deducted for late submission"))
        lines += [line(f"Q{n}. Question {n}") for n in range(4, 11)]
        self.assertEqual(question_lines(lines), [f"Q{n}. Question {n}" for n in range(1, 11)])

    def test_numbering_that_starts_again_is_a_second_set_of_questions(self):
        # Tutorial Q1-2, then Assignment Q1-3: the assignment was glued onto tutorial Q2
        lines = [line("Q1. Tutorial one"), line("Q2. Tutorial two"), line("Assignment"),
                 line("Q1. Assignment one"), line("Q2. Assignment two"), line("Q3. Assignment three")]
        self.assertEqual(question_lines(lines),
                         ["Q1. Tutorial one", "Q2. Tutorial two", "Q1. Assignment one",
                          "Q2. Assignment two", "Q3. Assignment three"])

    def test_a_wrapped_reference_to_an_earlier_question_is_still_rejected(self):
        # "... (Fig. 1 of\nQ1). Write a program ..." -- the reason the old rule existed
        lines = [line("Q1. Draw the tree."), line("Q2. Insert 5."), line("Q3. Use the tree (Fig. 1 of"),
                 line("Q1). Write a program for it."), line("Q4. Delete 5."), line("Q5. Print it.")]
        self.assertEqual(question_lines(lines),
                         ["Q1. Draw the tree.", "Q2. Insert 5.", "Q3. Use the tree (Fig. 1 of",
                          "Q4. Delete 5.", "Q5. Print it."])
        first = [line("Q1. Use the tree (Fig. 1 of"), line("Q1). Write a program for it."),
                 line("Q2. Delete 5.")]
        self.assertEqual(question_lines(first), ["Q1. Use the tree (Fig. 1 of", "Q2. Delete 5."])

    def test_a_missed_number_does_not_stop_the_later_ones(self):
        lines = [line("Q1. a"), line("Q2. b"), line("Q4. d"), line("Q5. e")]
        self.assertEqual(len(question_lines(lines)), 4)

    def test_bare_numbers_count_only_at_the_left_margin(self):
        lines = [line("1) First question"), line("1) an indented sub-point", x=90),
                 line("2) second sub-point", x=90), line("2) Second question"), line("3) Third question")]
        self.assertEqual(question_lines(lines),
                         ["1) First question", "2) Second question", "3) Third question"])

    def test_no_candidates_means_no_questions(self):
        self.assertEqual(T.choose_question_markers([]), [])
        self.assertEqual(question_lines([line("just a paragraph")]), [])


# ---------------------------------------------------------------- figures
def figure(page, y, name="fig.png"):
    return T.ImageRef(page=page, x0=50.0, y0=y, path=Path(name), sha1="x")


class TestFigures(unittest.TestCase):
    def units(self):
        return [T.Unit("Q1. Draw it.", True, page=0, y0=200.0), T.Unit("Q2. Insert 5.", True, page=0, y0=500.0)]

    def test_a_figure_above_the_first_question_is_not_given_to_question_one(self):
        units = self.units()
        T.associate_images_to_units(units, [figure(0, 40.0, "logo.png")])       # the sheet's header
        self.assertEqual([len(u.images) for u in units], [0, 0])

    def test_a_figure_goes_to_the_question_above_it(self):
        units = self.units()
        T.associate_images_to_units(units, [figure(0, 350.0, "a.png"), figure(0, 600.0, "b.png"),
                                            figure(1, 100.0, "c.png")])
        self.assertEqual([[i.path.name for i in u.images] for u in units], [["a.png"], ["b.png", "c.png"]])

    def test_an_image_type_gemini_cannot_read_is_saved_as_png(self):
        class Page:
            def get_image_info(self, xrefs=False):
                return [{"xref": 7, "width": 300, "height": 200, "bbox": (50, 300, 350, 500)}]

        class Doc:
            def __len__(self):
                return 1

            def __getitem__(self, i):
                return Page()

            def extract_image(self, xref):
                return {"image": b"jpeg-2000 bytes", "ext": "jpx"}

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(T, "image_as_png", lambda doc, xref: b"png bytes"):
            refs = T.extract_images(Doc(), "DS", "DS_TUT_1", Path(d))
            self.assertEqual([r.path.suffix for r in refs], [".png"])
            self.assertEqual(refs[0].path.read_bytes(), b"png bytes")


# ---------------------------------------------------------------- Gemini client
class FakeGenai:
    """Stands in for google.genai: records what was sent, answers or fails as told."""

    def __init__(self, fail_with=None):
        self.fail_with, self.sent, self.calls = fail_with, [], 0
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls += 1
        self.sent.append((model, contents))
        if self.fail_with:
            raise RuntimeError(self.fail_with)
        return types.SimpleNamespace(text="A -> B")


GENAI_TYPES = types.SimpleNamespace(
    GenerateContentConfig=lambda **kw: types.SimpleNamespace(**kw),
    Part=types.SimpleNamespace(from_bytes=lambda data, mime_type: ("image", mime_type)))


def client_with(fake, models=("m1", "m2", "m3")):
    client = T.GeminiClient("key", primary_model=models[0], fallback_models=models[1:])
    client._client = fake
    return client


def no_sleeping(seconds):
    raise AssertionError(f"slept {seconds}s")


class TestGeminiClient(unittest.TestCase):
    def setUp(self):
        google = types.ModuleType("google")
        google.genai = types.SimpleNamespace(types=GENAI_TYPES)
        patcher = mock.patch.dict(sys.modules, {"google": google, "google.genai": google.genai,
                                                "google.genai.types": GENAI_TYPES})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_every_model_unavailable_fails_at_once_instead_of_waiting_a_day(self):
        fake = FakeGenai(fail_with="404 NOT_FOUND: model is no longer available")
        with mock.patch.object(T.time, "sleep", no_sleeping), self.assertRaises(T.GeminiError) as stop:
            client_with(fake)._generate_raw("prompt", json_mode=False)
        self.assertIn("no Gemini model is available", str(stop.exception))
        self.assertEqual(fake.calls, 3)                     # each model tried once

    def test_a_long_rate_limit_stops_instead_of_waiting(self):
        fake = FakeGenai(fail_with="429 RESOURCE_EXHAUSTED quota. Please retry in 40000s")
        with mock.patch.object(T.time, "sleep", no_sleeping), self.assertRaises(T.GeminiError) as stop:
            client_with(fake)._generate_raw("prompt", json_mode=False)
        self.assertIn("rate-limited", str(stop.exception))

    def test_a_short_rate_limit_is_waited_out(self):
        client = client_with(FakeGenai(), models=("m1",))
        client._cooldown["m1"] = T.time.time() + 20
        slept = []
        with mock.patch.object(T.time, "sleep", slept.append):
            self.assertEqual(client._pick_model(), "m1")
        self.assertEqual(len(slept), 1)
        self.assertTrue(15 < slept[0] <= 20)

    def test_a_model_that_works_is_used(self):
        fake = FakeGenai()
        self.assertEqual(client_with(fake)._generate_raw("prompt", json_mode=False), "A -> B")
        self.assertEqual(fake.calls, 1)

    def test_figure_is_sent_with_its_real_type(self):
        with tempfile.TemporaryDirectory() as d:
            for name, mime in (("a.png", "image/png"), ("b.jpg", "image/jpeg"), ("c.jpeg", "image/jpeg"),
                               ("d.webp", "image/webp")):
                fake = FakeGenai()
                path = Path(d) / name
                path.write_bytes(name.encode())
                self.assertEqual(client_with(fake).transcribe_diagram(path), "A -> B")
                self.assertEqual(fake.sent[0][1][0], ("image", mime), name)
            # every non-PNG used to be labelled image/jpeg
            fake, odd = FakeGenai(), Path(d) / "e.jpx"
            odd.write_bytes(b"jpx")
            with self.assertRaises(T.GeminiError):
                client_with(fake).transcribe_diagram(odd)
            self.assertEqual(fake.calls, 0)


class TestOneFailedFigureDoesNotStopTheRun(unittest.TestCase):
    def test_question_text_is_kept_and_the_figure_gets_a_note(self):
        unit = T.Unit("Q1. Traverse the graph below.", True, page=0, y0=200.0,
                      images=[figure(0, 300.0, "good.png"), figure(0, 400.0, "bad.png")])

        def transcribe(path):
            if path.name == "bad.png":
                raise T.GeminiError("all models failed")
            return "A -> B"

        failed = []
        text = T.build_description([unit], transcribe, failed)
        self.assertEqual(text, "Q1. Traverse the graph below.\n"
                               "[Diagram Context / Description: A -> B]\n"
                               "[Diagram Context / Description: figure bad.png could not be transcribed]")
        self.assertEqual(failed, ["bad.png"])


# ---------------------------------------------------------------- chunk size
# The embedding model reads only 256 tokens of a chunk. Several questions used
# to be packed into chunks of up to 300 WORDS (and a longer question was kept
# whole), so 16 of the 19 committed tutorial chunks ran past that window.
# T.estimate_tokens is the measure, the same one the book chunks are held to.
def long_question(number=7, code_lines=60):
    """A question with a long program and a figure description: far more
    than one chunk's worth of tokens."""
    code = "\n".join(f"for (int i{k} = 0; i{k} < n; i{k}++) a[i{k}] = b[i{k}] + c[i{k}];"
                     for k in range(code_lines))
    return (f"Q{number}. Predict the output of the program below and explain each line.\n"
            f"{code}\n[Diagram Context / Description: A -> B, B -> C, C -> A]")


def lines_without_notes(pieces):
    """Every non-blank line of the pieces, minus the "(Q7 continued)" notes."""
    lines = [ln.strip() for piece in pieces for ln in piece.split("\n")]
    return [ln for ln in lines if ln and not (ln.startswith("(") and ln.endswith("continued)"))]


def keyword_tagger(subject_name, text, topics):
    return T.keyword_classify(text, topics)


def no_cos(topic_ids):
    return [], []


class TestChunksFitTheEmbeddingModel(unittest.TestCase):
    def assert_all_fit(self, pieces):
        for piece in pieces:
            self.assertLessEqual(T.estimate_tokens(piece), T.MAX_CHUNK_TOKENS, piece[:60])

    def test_a_question_that_fits_is_one_chunk_exactly_as_written(self):
        text = "Q2. Construct a Min Binary Heap using 10, 22, 30, 40.\n[Diagram Context / Description: A -> B]"
        self.assertEqual(T.split_question(text), [text])

    def test_a_long_question_is_cut_into_pieces_that_fit(self):
        text = long_question()
        self.assertGreater(T.estimate_tokens(text), 3 * T.MAX_CHUNK_TOKENS)
        pieces = T.split_question(text)
        self.assertGreater(len(pieces), 3)
        self.assert_all_fit(pieces)

    def test_no_line_is_lost_repeated_or_reordered(self):
        text = long_question()
        self.assertEqual(lines_without_notes(T.split_question(text)), text.split("\n"))

    def test_every_later_piece_says_which_question_it_continues(self):
        pieces = T.split_question(long_question(number=7))
        self.assertTrue(pieces[0].startswith("Q7. Predict the output"))
        for piece in pieces[1:]:
            self.assertTrue(piece.startswith("(Q7 continued)\n"), piece[:40])

    def test_question_label(self):
        for text, label in (("Q5. Sort it.", "Q5"), ("Q.12) Sort it.", "Q12"), ("Question 3: Sort", "Q3"),
                            ("3) Sort it.", "Q3"), ("  Q1 . Sort", "Q1"), ("Sort the array.", "")):
            self.assertEqual(T.question_label(text), label, text)
        pieces = T.split_question(long_question().replace("Q7. ", "", 1))
        self.assertTrue(all(p.startswith("(continued)\n") for p in pieces[1:]))

    def test_every_question_is_its_own_chunk(self):
        units = [T.Unit(f"Q{n}. Question {n} about stacks.", True, page=0, y0=100.0 * n) for n in (1, 2, 3)]
        chunks = T.chunks_for_questions(units, "Data Structures", [], lambda p: "", keyword_tagger, no_cos)
        self.assertEqual([c.description for c in chunks],
                         ["Q1. Question 1 about stacks.", "Q2. Question 2 about stacks.",
                          "Q3. Question 3 about stacks."])

    def test_topics_are_chosen_once_per_question_and_every_piece_gets_them(self):
        units = [T.Unit(long_question(number=1), True, page=0, y0=100.0),
                 T.Unit("Q2. Insert 5 into the heap.", True, page=1, y0=100.0)]
        asked = []

        def classify(subject_name, text, topics):
            asked.append(text)
            tid = "DS_01" if text.startswith("Q1.") else "DS_03"
            return [{"topic": f"topic {tid}", "topic_id": tid}]

        def cos_for(topic_ids):
            return ([f"CO-{t}" for t in topic_ids],
                    [{"id": f"CO-{t}", "label": f"CO-{t}"} for t in topic_ids])

        chunks = T.chunks_for_questions(units, "Data Structures", [], lambda p: "", classify, cos_for)
        self.assertEqual(asked, [units[0].text, units[1].text])      # whole questions, once each
        self.assertGreater(len(chunks), 2)
        *first, last = chunks
        for c in first:
            self.assertEqual((c.topic_ids, c.topics, c.cos), (["DS_01"], ["topic DS_01"], ["CO-DS_01"]))
        self.assertEqual((last.description, last.topic_ids), ("Q2. Insert 5 into the heap.", ["DS_03"]))
        self.assert_all_fit(c.description for c in chunks)

    def test_a_failed_classification_falls_back_to_keywords(self):
        def broken(subject_name, text, topics):
            raise T.GeminiError("all models failed")

        topics = [{"id": "DS_07", "name": "Hashing (Hash Tables, Collision Resolution)"}]
        units = [T.Unit("Q1. Insert the keys into a hash table using linear probing.", True, page=0, y0=1.0)]
        chunks = T.chunks_for_questions(units, "Data Structures", topics, lambda p: "", broken, no_cos)
        self.assertEqual(chunks[0].topic_ids, ["DS_07"])

    def test_figures_are_described_inside_their_own_question(self):
        units = [T.Unit("Q1. Traverse the graph.", True, page=0, y0=100.0, images=[figure(0, 150.0, "g.png")]),
                 T.Unit("Q2. Insert 5.", True, page=0, y0=300.0)]
        chunks = T.chunks_for_questions(units, "Data Structures", [], lambda p: "A -> B", keyword_tagger, no_cos)
        self.assertEqual([c.description for c in chunks],
                         ["Q1. Traverse the graph.\n[Diagram Context / Description: A -> B]", "Q2. Insert 5."])

    def test_verify_reports_a_chunk_that_is_too_long(self):
        def row(chunk_id, text):
            return {"chunk_id": chunk_id, "description": text, "topics": ["Hashing"],
                    "topic_ids": ["DS_07"], "cos": ["CO5"], "co_details": [{"id": "CO5", "label": "CO5"}]}

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "DS_tutorial_chunks.jsonl"
            path.write_text(json.dumps(row("DS_tut_chunk_001", "Q1. Short.")) + "\n", encoding="utf-8")
            with mock.patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(T.verify_outputs(d), 0)
            path.write_text(path.read_text() + json.dumps(row("DS_tut_chunk_002", long_question())) + "\n",
                            encoding="utf-8")
            with mock.patch("sys.stdout", new=io.StringIO()) as out:
                self.assertEqual(T.verify_outputs(d), 1)
            self.assertIn("DS_tut_chunk_002", out.getvalue())

    def test_committed_tutorial_chunks_fit_the_embedding_model(self):
        # Fails until the tutorials are extracted again with this version:
        #   python Extracting/tutorial_chunks.py
        files = sorted((ROOT / "Extracting" / "Processed").glob("*_tutorial_chunks.jsonl"))
        self.assertTrue(files)
        too_long = [row["chunk_id"] for path in files
                    for row in map(json.loads, filter(str.strip, path.read_text(encoding="utf-8").splitlines()))
                    if T.estimate_tokens(row["description"]) > T.MAX_CHUNK_TOKENS]
        self.assertEqual(too_long, [], "re-run: python Extracting/tutorial_chunks.py")


# ---------------------------------------------------------------- subjects and COs
class TestTopicCatalog(unittest.TestCase):
    def test_every_subject_of_the_graph_has_a_catalog(self):
        # only DS and APS were known (a copy written into the script), so SDF tutorials were skipped
        catalog = T.load_topic_catalog(TOPIC_MAP, SUBJECTS_DIR)
        for folder, subject in (("DS", "DS"), ("APS", "ALGO"), ("SDF-1", "SDF1"), ("SDF-2", "SDF2"),
                                ("SDF1", "SDF1"), ("ALGO", "ALGO")):
            self.assertEqual(catalog[folder]["yaml_subject"], subject, folder)
            self.assertTrue(all(t["id"].startswith(subject + "_") for t in catalog[folder]["topics"]))
        self.assertEqual(catalog["APS"]["name"], "Algorithms and Problem Solving")

    def test_topics_come_from_the_subject_files(self):
        with tempfile.TemporaryDirectory() as d:
            subjects = Path(d) / "subjects"
            subjects.mkdir()
            (subjects / "x.yaml").write_text(
                "subject: X\nsubject_name: Subject X\nsemester: 1\n"
                "topics:\n  - id: X_01\n    name: A topic that was renamed\n", encoding="utf-8")
            topic_map = Path(d) / "topic_map.yaml"
            topic_map.write_text("books:\n  XBOOK:\n    yaml_subject: X\n", encoding="utf-8")
            catalog = T.load_topic_catalog(topic_map, subjects)
        self.assertEqual(catalog["XBOOK"], {"name": "Subject X", "yaml_subject": "X",
                                            "topics": [{"id": "X_01", "name": "A topic that was renamed"}]})
        self.assertEqual(catalog["X"], catalog["XBOOK"])

    def test_chunk_file_name_keeps_the_hyphen_of_the_folder(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(T.write_subject_jsonl("SDF-1", [], Path(d)).name, "SDF-1_tutorial_chunks.jsonl")


class TestCourseOutcomes(unittest.TestCase):
    graph = T.load_curriculum_graph(SUBJECTS_DIR)

    def test_default_is_the_chunks_own_topics_only(self):
        # the default was 1, which labelled a BST chunk (DS_04, CO4) with CO1 as well
        self.assertEqual(T.build_arg_parser().parse_args([]).prereq_depth, 0)
        self.assertEqual(T.collect_cos(["DS_04"], self.graph, 0, own_subject="DS")[0], ["CO4"])
        self.assertEqual(T.collect_cos(["DS_04"], self.graph, 1, own_subject="DS")[0], ["CO4", "CO1"])

    def test_committed_tutorial_chunks_hold_no_prerequisite_cos(self):
        catalog = T.load_topic_catalog(TOPIC_MAP, SUBJECTS_DIR)
        files = sorted((ROOT / "Extracting" / "Processed").glob("*_tutorial_chunks.jsonl"))
        self.assertTrue(files)
        for path in files:
            subject = catalog[path.name.removesuffix("_tutorial_chunks.jsonl")]["yaml_subject"]
            for row in map(json.loads, filter(str.strip, path.read_text(encoding="utf-8").splitlines())):
                cos, details = T.collect_cos(row["topic_ids"], self.graph, 0, own_subject=subject)
                self.assertEqual((row["cos"], row["co_details"]), (cos, details), row["chunk_id"])


# ---------------------------------------------------------------- a real sheet (read only)
class TestRealSheet(unittest.TestCase):
    def test_code_lines_survive_on_a_real_multi_page_sheet(self):
        pdf = ROOT / "Tutorials" / "DS" / "DS_TUT_2.pdf"
        if not pdf.exists() or not hasattr(T.pymupdf, "open"):
            self.skipTest("needs PyMuPDF and Tutorials/DS/DS_TUT_2.pdf")
        with tempfile.TemporaryDirectory() as d:            # figures go here, not into the repo
            units = T.process_pdf(pdf, "DS", ROOT / "Tutorials", Path(d), False)
        text = "\n".join(u.text for u in units)
        self.assertEqual(len(units), 4)
        for code in ("int main()", "using namespace std;", "return 0;"):
            self.assertIn(code, text)                       # all three used to be missing

    def test_every_real_sheet_gives_chunks_that_fit_and_keep_every_line(self):
        pdfs = sorted((ROOT / "Tutorials").rglob("*.pdf"))
        if not pdfs or not hasattr(T.pymupdf, "open"):
            self.skipTest("needs PyMuPDF and the PDFs in Tutorials/")
        # figure descriptions from Gemini's saved answers, so no call is made
        cache_file = ROOT / "Extracting" / ".cache" / "gemini_cache.json"
        cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}

        def transcribe(path):
            import hashlib
            key = T.GeminiClient._key("diagram", hashlib.sha1(path.read_bytes()).hexdigest())
            return cache.get(key, "A -> B")

        with tempfile.TemporaryDirectory() as d:            # figures go here, not into the repo
            for pdf in pdfs:
                units = T.process_pdf(pdf, pdf.parent.name, ROOT / "Tutorials", Path(d), False)
                chunks = T.chunks_for_questions(units, "", [], transcribe, keyword_tagger, no_cos)
                for c in chunks:
                    self.assertLessEqual(T.estimate_tokens(c.description), T.MAX_CHUNK_TOKENS,
                                         f"{pdf.name}: {c.description[:60]}")
                whole = [ln.strip() for u in units
                         for ln in T.build_description([u], transcribe).split("\n") if ln.strip()]
                self.assertEqual(lines_without_notes(c.description for c in chunks), whole, pdf.name)


if __name__ == "__main__":
    unittest.main()
