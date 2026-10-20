"""
Book text clean-up (Extracting/text_cleanup.py), used by extract.py.

    python3 -m unittest tests.test_text_cleanup -v      (from the repo root)

The rules are tried on made-up pages (tests/fake_pdf.py). When PyMuPDF and
the PDFs in Books/ are there, a few real pages are checked too (read only).
The last test checks the committed chunk files; it fails until the books are
extracted again with this version:  python Extracting/extract.py
"""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Extracting"))

import text_cleanup as C  # noqa: E402
from tests.fake_pdf import FakePage  # noqa: E402


def line_text(*spans):
    """The cleaned text of one line made of (font, text) spans."""
    return C.page_lines(FakePage(lines=[(list(spans), 300)]))[0].text


# ---------------------------------------------------------------- characters
class TestCharacters(unittest.TestCase):
    def test_ligatures_become_plain_letters(self):
        self.assertEqual(C.clean_span("CMR10", "eﬃcient ﬁrst ﬂow diﬀerent oﬄine"),
                         "efficient first flow different offline")
        self.assertEqual(C.clean_text("Suﬃx Trees and Suﬃx Arrays"), "Suffix Trees and Suffix Arrays")

    def test_clrs_maths_reads_as_maths(self):
        # how the APS book (CLRS) came out before: "n1 D q \x00 p C 1"
        self.assertEqual(line_text(("MT2MIT", "n1"), ("MT2SYT", " D "), ("MT2MIT", "q"),
                                   ("MT2SYT", " \x00 "), ("MT2MIT", "p"), ("MT2SYT", " C "),
                                   ("TimesNewRoman", "1")), "n1 = q − p + 1")
        self.assertEqual(line_text(("MT2MIT", "f .n/"), ("MT2SYT", " D "), ("MT2MIT", "‚.n"),
                                   ("MT2MIS", "2"), ("MT2MIT", "/")), "f (n) = Θ(n2)")
        self.assertEqual(line_text(("MT2MIT", "AŒ1 : : j "), ("MT2SYT", "\x00"), ("MT2MIT", " 1\x8d")),
                         "A[1 .. j − 1]")
        self.assertEqual(line_text(("MT2SYT", "h"), ("MT2MIT", "a1; a2; : : : ; an"), ("MT2SYT", "i")),
                         "⟨a1, a2, ... , an⟩")
        self.assertEqual(line_text(("MT2MIT", "x"), ("MT2SYT", " ¤ "), ("TimesNewRoman", "NIL and "),
                                   ("MT2MIT", "i"), ("MT2SYT", " \x14 "), ("MT2MIT", "n"),
                                   ("MT2SYT", " 6D "), ("MT2MIT", "j")), "x ≠ NIL and i ≤ n != j")
        self.assertEqual(line_text(("TimesNewRoman", "for "), ("MT2MIT", "j"), ("MT2SYT", " D "),
                                   ("TimesNewRoman", "2 "), ("TimesNewRoman,Bold", "to "),
                                   ("MT2MIT", "A:length")), "for j = 2 to A.length")

    def test_greek_letters_and_sets(self):
        self.assertEqual(C.clean_span("MT2MIT", "\x7f.n lg n/"), "Ω(n lg n)")
        self.assertEqual(C.clean_span("MT2MIT", "\x19.1/"), "π(1)")
        self.assertEqual(C.clean_span("MT2SYT", "f"), "{")
        self.assertEqual(C.clean_span("MT2HRBT", "Z"), "ℤ")
        self.assertEqual(C.clean_span("MT2EXA", "X"), "∑")
        self.assertEqual(C.clean_span("CMEX10", "\x02"), "[")      # the DS book's big brackets

    def test_the_same_characters_in_a_text_font_are_left_alone(self):
        for font in ("TimesNewRoman", "CMR10", "Palatino-Roman", "Courier"):
            self.assertEqual(C.clean_span(font, "Data. C/C++ (a; b) = D"), "Data. C/C++ (a; b) = D")

    def test_figure_glyphs_and_other_control_characters_go(self):
        self.assertEqual(C.clean_span("T7", "\x00\x01\x01\x00"), "")    # DS book: figures drawn as text
        self.assertEqual(C.clean_span("TimesNewRoman", "a\x00b\tc"), "ab\tc")


# ---------------------------------------------------------------- headers and footers
def handbook_page(n, body):
    return FakePage(lines=[("Handbook of Data Structures and Applications", 20), (f"3-{n}", 22)]
                    + [(text, 120 + 12 * i) for i, text in enumerate(body)]
                    + [("© 2005 by Chapman & Hall/CRC", 780)])


class TestHeadersAndFooters(unittest.TestCase):
    def test_running_headers_footers_and_page_numbers_go(self):
        book = [handbook_page(n, [f"Body line {n} about heaps.", "A second line."]) for n in range(1, 5)]
        pages = C.clean_book_pages(book)
        self.assertEqual(pages[0], "Body line 1 about heaps.\nA second line.\n")
        self.assertTrue(all("Handbook" not in p and "Chapman" not in p and "3-" not in p for p in pages))

    def test_the_same_words_in_the_body_stay(self):
        book = [handbook_page(n, ["Handbook of Data Structures and Applications"]) for n in range(1, 5)]
        self.assertEqual(C.clean_book_pages(book)[0], "Handbook of Data Structures and Applications\n")

    def test_a_margin_line_on_only_two_pages_stays(self):
        book = [FakePage(lines=[("Value", 790), ("Body text", 300)]) for _ in range(2)]
        book.append(FakePage(lines=[("Body text", 300)]))
        self.assertEqual(C.clean_book_pages(book)[0], "Value\nBody text\n")    # an ASCII table, not a footer

    def test_a_page_number_goes_from_the_margin_only(self):
        page = FakePage(lines=[("419", 15), ("Chapter 16", 200), ("419", 300), ("xii", 790)])
        self.assertEqual(C.clean_book_pages([page])[0], "Chapter 16\n419\n")

    def test_spaced_out_headers_of_the_cpp_book_go(self):
        header = "C + + : T h e C o m p l e t e R e f e r e n c e"
        book = [FakePage(lines=[(str(400 + n), 20), (header, 22), (f"line {n}", 300)]) for n in range(3)]
        self.assertEqual(C.clean_book_pages(book), ["line 0\n", "line 1\n", "line 2\n"])


# ---------------------------------------------------------------- figure labels
def body(*texts, font="Times-Roman"):
    return [(t, 120 + 10 * i, font) for i, t in enumerate(texts)]


class TestFigureLabels(unittest.TestCase):
    def clean(self, lines):
        return C.clean_book_pages([FakePage(lines=lines)])[0]

    def test_a_run_of_node_labels_goes(self):
        # DS_03's top search hit began with 25 lines like these (Figure 3.11)
        labels = ["20", "10", "15", "5", "8", "30", "25", "A", "B"]
        self.assertEqual(self.clean(body("FIGURE 3.11: Splitting a BST.", *labels, "3.7 Heaps")),
                         "FIGURE 3.11: Splitting a BST.\n3.7 Heaps\n")

    def test_labels_written_as_pairs_go_too(self):
        # interval heap nodes in DS Figure 8.10 came through as "4,10", "5,11", ...
        pairs = ["4,10", "5,11", "5,9", "4,7", "8,8", "A", "4,12"]
        self.assertEqual(self.clean(body("(a) min heap", *pairs, "FIGURE 8.10: Interval heap.")),
                         "(a) min heap\nFIGURE 8.10: Interval heap.\n")

    def test_a_few_short_lines_stay(self):
        self.assertEqual(self.clean(body("Section", "3.7", "Heaps", "1", "2", "end")),
                         "Section\n3.7\nHeaps\n1\n2\nend\n")

    def test_program_output_in_a_code_font_stays(self):
        out = ["1", "2", "3", "4", "5", "6"]
        self.assertEqual(self.clean(body(*out, font="Courier")), "1\n2\n3\n4\n5\n6\n")


class TestContentsLines(unittest.TestCase):
    def clean(self, lines):
        return C.clean_book_pages([FakePage(lines=lines)])[0]

    def test_a_chapter_opening_contents_box_goes(self):
        # the prerequisite source for Linear DS was this box and nothing else
        box = body("Stacks and Queues .................................", "2-12",
                   "Stack Implementation • Queue Implementation",
                   "Introduction. . . . . . . . . . . . . . . 1-1",
                   "2.1", "Introduction", "In this chapter, we review several basic structures.")
        self.assertEqual(self.clean(box), "2.1\nIntroduction\nIn this chapter, we review several basic structures.\n")

    def test_bullet_lists_problem_numbers_and_code_stay(self):
        lines = body("• queues and deques [5, 10, 11, 21]", "2-4", "Inversions",
                     "Wait while loading ......", "x = a + b", "....... 3")[:5]
        lines += [("printf(\"..........\");", 300, "Courier"), ("12", 312, "Courier")]
        self.assertEqual(self.clean(lines),
                         "• queues and deques [5, 10, 11, 21]\n2-4\nInversions\nx = a + b\n"
                         "printf(\"..........\");\n12\n")


# ---------------------------------------------------------------- end of a chapter
class TestBackMatter(unittest.TestCase):
    def test_reference_list_is_cut(self):
        text = ("A heap supports insert in O(log n).\nReferences\n[1]\n"
                "T. Cormen, C. Leiserson, and R. Rivest, Introduction to Algorithms, 1992.\n[2]\nD. Knuth\n")
        self.assertEqual(C.cut_back_matter(text), "A heap supports insert in O(log n).\n")

    def test_acknowledgments_and_what_follows_are_cut(self):
        text = ("The queue takes O(1) time.\nAcknowledgments\n"
                "The author would like to thank Kun Gao for proofreading this chapter.\nReferences\n[1]\n")
        self.assertEqual(C.cut_back_matter(text), "The queue takes O(1) time.\n")

    def test_clrs_chapter_notes_are_cut(self):
        text = ("Problem 2-4 Inversions\nChapter notes\n"
                "In 1968, Knuth published the first of three volumes with the general title\n")
        self.assertEqual(C.cut_back_matter(text), "Problem 2-4 Inversions\n")

    def test_the_cpp_chapter_on_references_stays(self):
        text = ("References\nC++ contains a feature that is related to the pointer called a reference. A\n"
                "reference is essentially an implicit pointer.\n")
        self.assertEqual(C.cut_back_matter(text), text)

    def test_a_table_of_contents_stays(self):
        text = "Contents\nBibliography\n1231\nIndex\n1251\nChapter notes\n36\n"
        self.assertEqual(C.cut_back_matter(text), text)

    def test_a_section_that_is_only_back_matter_becomes_empty(self):
        self.assertEqual(C.cut_back_matter("References\n[1]\nD. Knuth, The Art of Computer Programming, 1997.\n"),
                         "")


# ---------------------------------------------------------------- the real books (read only)
def real_book(name):
    pdf = ROOT / "Books" / f"{name}.pdf"
    try:
        import pymupdf
    except ImportError:
        return None
    return pymupdf.open(pdf) if pdf.exists() else None


class TestRealBooks(unittest.TestCase):
    def page(self, book, number):
        doc = real_book(book)
        if doc is None:
            self.skipTest(f"needs PyMuPDF and Books/{book}.pdf")
        return "\n".join(ln.text for ln in C.page_lines(doc[number - 1]))

    def test_clrs_insertion_sort_reads_correctly(self):
        text = self.page("APS", 39)
        for want in ("INSERTION-SORT(A)", "for j = 2 to A.length", "key = A[j ]",
                     "the sorted sequence A[1 .. j − 1]", "while i > 0 and A[i] > key"):
            self.assertIn(want, text)
        self.assertNotRegex(text, r"[\x00-\x08\x0b-\x1f\x7f-\x9fŒ]")

    def test_handbook_ligatures_are_gone(self):
        text = self.page("DS", 6)
        self.assertIn("defined", text)
        self.assertNotRegex(text, "[ﬀ-ﬆ]")


# ---------------------------------------------------------------- the committed chunk files
class TestCommittedChunkFiles(unittest.TestCase):
    def test_book_chunks_hold_clean_text(self):
        # Fails until the books are extracted again with this version:
        #   python Extracting/extract.py  &&  python tag_topics.py
        problems = []
        checks = {"ligature": re.compile("[ﬀ-ﬆ]"),
                  "control or MathTime character": re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9fŒ]"),
                  "running header/footer": re.compile(r"Handbook of Data Structures and Applications$|"
                                                      r"© 2005 by Chapman & Hall/CRC", re.M),
                  "reference list": re.compile(r"^References\n\[1\]$", re.M),
                  "contents line": re.compile(r"[A-Za-z]\s*(?:\. ?){5,}\s*\S{0,8}$", re.M)}
        for path in sorted((ROOT / "Extracting" / "Processed").glob("*_chunks.jsonl")):
            if path.name.endswith("_tutorial_chunks.jsonl"):
                continue
            for row in map(json.loads, filter(str.strip, path.read_text(encoding="utf-8").splitlines())):
                if row.get("topic_id", "NONE") == "NONE" or row.get("needs_review"):
                    continue        # never retrieved (e.g. the copyright page, which names the book)
                problems += [f"{row['chunk_id']}: {what}" for what, rx in checks.items()
                             if rx.search(row["text"])]
        self.assertEqual(problems[:10], [], f"{len(problems)} problem(s); re-run: python Extracting/extract.py "
                                            f"&& python tag_topics.py")


if __name__ == "__main__":
    unittest.main()
