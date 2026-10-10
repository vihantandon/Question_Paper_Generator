"""
Tests for Extracting/extract.py (books -> section-aware chunks).

    python3 -m unittest tests.test_extract -v      (from the repo root)

No PDF and no PyMuPDF needed: a FakeDoc stands in for the opened book. It
only has what extract.py calls: get_toc(), len() and doc[page].get_text().
"""
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Extracting"))

try:
    import fitz  # noqa: F401
except ImportError:                 # PyMuPDF only opens real PDFs; these tests never do
    sys.modules["fitz"] = types.ModuleType("fitz")

import extract  # noqa: E402


class FakePage:
    def __init__(self, text):
        self.text = text

    def get_text(self, *args):
        return self.text


class FakeDoc:
    """pages: list of page texts; toc: [[level, title, 1-based page], ...]"""

    def __init__(self, pages, toc):
        self.pages = [FakePage(t) for t in pages]
        self.toc = toc

    def get_toc(self):
        return self.toc

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, i):
        return self.pages[i]


def page_text(n):
    """A page with enough words to be chunked, and a marker to find it again."""
    return f"PAGE-{n}-START " + f"words on page {n} " * 30 + f"PAGE-{n}-END"


def book(n_pages, toc):
    return FakeDoc([page_text(n) for n in range(1, n_pages + 1)], toc)


class TestTocSections(unittest.TestCase):
    def assert_pages_used_once(self, sections, first, last):
        """sections are 0-based (title, start, end): together they must cover
        first..last with no page missing and no page in two sections."""
        used = [p for _, start, end in sections for p in range(start, end + 1)]
        self.assertEqual(used, list(range(first, last + 1)))

    def test_entries_starting_on_the_same_page_make_one_section(self):
        # like the DS handbook: a heading and its sub-headings share a page
        doc = book(5, [[1, "1 Intro", 1], [2, "1.1 Scope", 1], [2, "1.2 Terms", 1],
                       [2, "1.3 History", 2], [1, "2 Arrays", 4]])
        sections = extract.get_toc_sections(doc)
        self.assertEqual(sections, [("1 Intro", 0, 0), ("1.3 History", 1, 2), ("2 Arrays", 3, 4)])
        self.assert_pages_used_once(sections, 0, 4)

    def test_section_stops_before_the_next_chapter_starts(self):
        # like CLRS: one entry per chapter. Chapter 2's first page used to be
        # stored at the end of chapter 1 as well.
        doc = book(6, [[1, "1 Role of Algorithms", 1], [1, "2 Getting Started", 3], [1, "3 Growth", 5]])
        sections = extract.get_toc_sections(doc)
        self.assertEqual(sections, [("1 Role of Algorithms", 0, 1), ("2 Getting Started", 2, 3),
                                    ("3 Growth", 4, 5)])

    def test_broken_bookmark_does_not_swallow_the_book(self):
        # DS: "53.3.3 Corner Block List" points at page 1 and became pages 1-1086
        doc = book(8, [[1, "7 Heaps", 3], [2, "7.1 Broken bookmark", 1], [1, "8 Hashing", 6]])
        sections = extract.get_toc_sections(doc)
        self.assertEqual(sections, [("7.1 Broken bookmark", 0, 1), ("7 Heaps", 2, 4), ("8 Hashing", 5, 7)])
        self.assert_pages_used_once(sections, 0, 7)

    def test_bookmarks_outside_the_book_are_ignored(self):
        doc = book(4, [[1, "No target", -1], [1, "Zero", 0], [1, "1 Real", 2], [1, "Past the end", 99]])
        self.assertEqual(extract.get_toc_sections(doc), [("1 Real", 1, 3)])

    def test_no_usable_bookmark_falls_back_to_headings(self):
        self.assertFalse(extract.get_toc_sections(book(3, [[1, "Nowhere", 0]])))
        self.assertIsNone(extract.get_toc_sections(book(3, [])))


class TestNoPageIsChunkedTwice(unittest.TestCase):
    def chunks(self, doc):
        with mock.patch.object(extract.fitz, "open", lambda path: doc, create=True):
            return extract.process_pdf("Books/FAKE.pdf", subject="FAKE")

    def test_each_page_appears_in_exactly_one_chunk(self):
        doc = book(6, [[1, "1 Intro", 1], [2, "1.1 Scope", 1], [2, "1.2 Terms", 1], [2, "1.3 History", 2],
                       [1, "2 Arrays", 4], [2, "2.1 Basics", 4], [2, "2.2 Search", 5]])
        chunks = self.chunks(doc)
        everything = " ".join(c.text for c in chunks)
        for n in range(1, 7):
            self.assertEqual(everything.count(f"PAGE-{n}-START"), 1, f"page {n}")
        texts = [c.text for c in chunks]
        self.assertEqual(len(texts), len(set(texts)), "two chunks have the same text")

    def test_page_range_of_a_chunk_is_its_own_section(self):
        doc = book(4, [[1, "1 Intro", 1], [1, "2 Arrays", 3]])
        self.assertEqual({(c.section_title, c.page_start, c.page_end) for c in self.chunks(doc)},
                         {("1 Intro", 1, 2), ("2 Arrays", 3, 4)})


if __name__ == "__main__":
    unittest.main()
