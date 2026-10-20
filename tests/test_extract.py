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
from tests.fake_pdf import FakePage  # noqa: E402  (plain text + the "dict" view text_cleanup reads)


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


# ---------------------------------------------------------------- chunk size
# The embedding model reads only EMBED_MAX_TOKENS tokens of a chunk. These
# tests use extract.estimate_tokens (no model needed) as the measure.
def prose(n_lines, first=0):
    """n printed lines of a book page, with no blank line between them
    (the way PyMuPDF returns a page of CLRS or the DS handbook)."""
    return "\n".join(f"line {i} of the page has about a dozen plain words in it"
                     for i in range(first, first + n_lines))


def lines_of(chunks):
    return [line for chunk in chunks for line in chunk.split("\n")]


def size(chunk):
    """A chunk's size the way the chunker adds it up: line by line."""
    return sum(extract.estimate_tokens(line) for line in chunk.split("\n"))


class TestChunksFitTheEmbeddingModel(unittest.TestCase):
    def test_books_and_tutorials_follow_the_same_size_rule(self):
        # the rule lives in token_budget.py; the tutorial extractor uses it too
        import token_budget
        self.assertIs(extract.split_into_chunks, token_budget.split_into_chunks)
        self.assertIs(extract.estimate_tokens, token_budget.estimate_tokens)
        self.assertEqual(extract.MAX_CHUNK_TOKENS, 254)

    def assert_all_fit(self, chunks):
        for c in chunks:
            self.assertLessEqual(extract.estimate_tokens(c), extract.MAX_CHUNK_TOKENS, c[:60])

    def test_a_whole_page_is_split_into_chunks_that_fit(self):
        page = prose(60)                     # ~700 words: one "paragraph" for the old code
        chunks = extract.split_into_chunks(page)
        self.assertGreater(len(chunks), 1)
        self.assert_all_fit(chunks)

    def test_no_line_is_lost_repeated_or_reordered(self):
        page = prose(60)
        self.assertEqual(lines_of(extract.split_into_chunks(page)), page.split("\n"))

    def test_blank_lines_and_page_breaks_change_nothing(self):
        # the old code cut at blank lines, which mostly meant "at every page"
        one_block = prose(40)
        with_breaks = prose(15) + "\n\n" + prose(15, first=15) + "\n \n\n" + prose(10, first=30)
        self.assertEqual(extract.split_into_chunks(with_breaks), extract.split_into_chunks(one_block))

    def test_a_line_longer_than_a_chunk_is_cut_between_words(self):
        words = [f"word{i}" for i in range(900)]       # e.g. a listing with no line breaks
        chunks = extract.split_into_chunks(" ".join(words))
        self.assertGreater(len(chunks), 1)
        self.assert_all_fit(chunks)
        self.assertEqual(" ".join(chunks).split(), words)

    def test_code_is_measured_by_its_symbols_not_its_words(self):
        loop = "for(i=0;i<n;i++)"                      # 1 word, 14 tokens for the real model
        self.assertEqual(len(loop.split()), 1)
        self.assertGreaterEqual(extract.estimate_tokens(loop), 14)
        code = "\n".join(f"for(i{k}=0;i{k}<n;i{k}++){{a[i{k}]=b[i{k}]+c[i{k}];}}" for k in range(40))
        chunks = extract.split_into_chunks(code)
        self.assertGreater(len(chunks), 1)             # 40 "words" would have been one chunk
        self.assert_all_fit(chunks)

    def test_a_short_section_is_one_chunk(self):
        self.assertEqual(extract.split_into_chunks(prose(3)), [prose(3)])
        self.assertEqual(extract.split_into_chunks(" \n\n "), [])


class TestNoTinyChunks(unittest.TestCase):
    TITLE_PAGE = "Chapter 16\nInheritance\n419\nC++\n"

    def test_chapter_title_page_is_not_a_chunk_of_its_own(self):
        # SDF-2: this 5-word page was a chunk, and the top search hit for "Inheritance"
        doc = FakeDoc([self.TITLE_PAGE, prose(60), prose(60, first=60)], [[1, "Ch16 Inheritance", 1]])
        with mock.patch.object(extract.fitz, "open", lambda path: doc, create=True):
            chunks = extract.process_pdf("Books/FAKE.pdf", subject="FAKE")
        self.assertTrue(chunks[0].text.startswith("Chapter 16\nInheritance\n419\nC++\nline 0 of the page"))
        for c in chunks:
            self.assertGreaterEqual(size(c.text), extract.MIN_CHUNK_TOKENS, c.text[:60])

    def test_short_last_chunk_is_evened_out_with_the_one_before(self):
        # find a length where packing line by line leaves a single line over
        for n in range(20, 60):
            page = prose(n)
            chunks = extract.split_into_chunks(page, max_tokens=10 ** 9)   # sanity: fits in one
            self.assertEqual(chunks, [page])
            chunks = extract.split_into_chunks(page)
            if len(chunks) < 2:
                continue
            self.assertEqual(lines_of(chunks), page.split("\n"), n)
            for c in chunks:
                self.assertLessEqual(size(c), extract.MAX_CHUNK_TOKENS, n)
                self.assertGreaterEqual(size(c), extract.MIN_CHUNK_TOKENS, n)

    def test_evening_out_never_makes_a_chunk_too_long(self):
        # 10 + 240 fill a chunk; the 30 left over cannot be shared without breaking the limit
        a, b, c = ("w " * 8).strip(), ("w " * 200).strip(), ("w " * 25).strip()
        self.assertEqual([extract.estimate_tokens(x) for x in (a, b, c)], [10, 240, 30])
        self.assertEqual(extract.split_into_chunks("\n".join([a, b, c])), [a + "\n" + b, c])


if __name__ == "__main__":
    unittest.main()
