"""
tag_topics.py options: --report-only, --push, --drop-suspect.

    python3 -m unittest tests.test_tag_topics -v      (from the repo root)

No Docker: FakeStore is an in-memory Chroma collection. The chunk files, the
topic map and the subject YAML are small made-up files in a temporary folder,
so the real ones are never read or written.
"""
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tag_topics  # noqa: E402

SUBJECT_YAML = """
subject: X
semester: 2
topics:
  - id: X_01
    name: Topic one
"""
TOPIC_MAP = r"""
min_margin: 2
books:
  BOOK:
    yaml_subject: X
    chapter_regex: '^(\d+)\s'
    chapters:
      1: [X_01]
"""


class FakeStore:
    """Just enough of a Chroma collection. rows: {id: metadata}"""

    def __init__(self, ids=()):
        self.rows = {i: {"subject": i.split("_")[0], "tag_method": "untagged"} for i in ids}

    def get(self, ids=None, where=None, include=None, limit=None, offset=0):
        found = [i for i in (ids if ids is not None else self.rows) if i in self.rows]
        found = found[offset:offset + limit] if limit else found[offset:]
        return {"ids": found, "metadatas": [dict(self.rows[i]) for i in found]}

    def update(self, ids, metadatas):
        for i, meta in zip(ids, metadatas):
            self.rows[i] = dict(meta)

    def delete(self, ids):
        if not ids:
            raise ValueError("Expected IDs to be a non-empty list, got 0 IDs in delete.")
        for i in ids:
            self.rows.pop(i, None)


def chunk(chunk_id, page_end=10, title="1 Intro"):
    return {"chunk_id": chunk_id, "subject": chunk_id.split("_")[0], "source_file": "Books/BOOK.pdf",
            "section_title": title, "page_start": 1, "page_end": page_end, "text": "some text"}


GOOD = [chunk("BOOK_000_00"), chunk("BOOK_000_01")]
SUSPECT = chunk("BOOK_009_00", page_end=900)             # one "section" covering 900 pages


class TagTopicsCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name)
        self.processed = folder / "Processed"
        self.processed.mkdir()
        (folder / "subjects").mkdir()
        (folder / "subjects" / "x.yaml").write_text(SUBJECT_YAML, encoding="utf-8")
        (folder / "topic_map.yaml").write_text(TOPIC_MAP, encoding="utf-8")
        self.book = self.processed / "BOOK_chunks.jsonl"
        self.write(self.book, GOOD + [SUSPECT])
        # a book with no entry in the topic map, and a tutorial file: both left alone
        self.write(self.processed / "OTHER_chunks.jsonl", [chunk("OTHER_000_00")])
        self.write(self.processed / "X_tutorial_chunks.jsonl", [{"chunk_id": "X_tut_chunk_001"}])
        for name, value in (("PROCESSED", self.processed), ("MAP_FILE", folder / "topic_map.yaml"),
                            ("SUBJECTS_DIR", folder / "subjects")):
            patcher = mock.patch.object(tag_topics, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.store = FakeStore(["BOOK_000_00", "BOOK_000_01", "BOOK_009_00", "OTHER_000_00"])
        self.chroma_opened = 0

    @staticmethod
    def write(path, chunks):
        path.write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")

    def ids_in_file(self):
        return [json.loads(line)["chunk_id"] for line in self.book.read_text(encoding="utf-8").splitlines()]

    def run_script(self, *options):
        """Run tag_topics.main() with these options; returns what it printed."""
        def http_client(host, port):
            self.chroma_opened += 1
            return types.SimpleNamespace(get_collection=lambda name: self.store)

        printed = io.StringIO()
        with mock.patch.dict(sys.modules, {"chromadb": types.SimpleNamespace(HttpClient=http_client)}), \
                mock.patch.object(sys, "argv", ["tag_topics.py", *options]), \
                contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
            tag_topics.main()
        return printed.getvalue()


class TestReportOnly(TagTopicsCase):
    def test_report_only_changes_no_file(self):
        before = self.book.read_bytes()
        printed = self.run_script("--report-only")
        self.assertIn("TAGGING SUMMARY", printed)
        self.assertEqual(self.book.read_bytes(), before)
        self.assertEqual(self.chroma_opened, 0)

    def test_report_only_with_push_is_refused_before_anything_is_touched(self):
        # it used to update Chroma, and with --drop-suspect it deleted vectors
        before_file, before_rows = self.book.read_bytes(), dict(self.store.rows)
        for options in (("--report-only", "--push"), ("--report-only", "--push", "--drop-suspect")):
            with self.assertRaises(SystemExit) as stop:
                self.run_script(*options)
            self.assertEqual(stop.exception.code, 2)
        self.assertEqual(self.book.read_bytes(), before_file)
        self.assertEqual(self.store.rows, before_rows)
        self.assertEqual(self.chroma_opened, 0)

    def test_report_only_with_drop_suspect_is_a_preview(self):
        before = self.book.read_bytes()
        printed = self.run_script("--report-only", "--drop-suspect")
        self.assertIn("would remove 1 suspect chunks", printed)
        self.assertEqual(self.book.read_bytes(), before)


class TestDropSuspect(TagTopicsCase):
    def assert_chroma_matches_the_files(self):
        self.assertEqual(sorted(self.store.rows), ["BOOK_000_00", "BOOK_000_01", "OTHER_000_00"])
        self.assertEqual(self.store.rows["BOOK_000_00"]["topic_id"], "X_01")        # tags were pushed
        self.assertEqual(self.store.rows["OTHER_000_00"]["tag_method"], "untagged")  # not this script's book

    def test_drop_now_and_push_later_still_removes_the_vector(self):
        # the first run takes the chunk out of the file, so the second run can
        # no longer name it; its vector used to stay in Chroma for good
        printed = self.run_script("--drop-suspect")
        self.assertEqual(self.ids_in_file(), ["BOOK_000_00", "BOOK_000_01"])
        self.assertIn("BOOK_009_00", self.store.rows)
        self.assertIn("still have vectors in Chroma", printed)

        printed = self.run_script("--push")
        self.assertIn("deleted 1 vector(s) whose chunk is no longer in the chunk files", printed)
        self.assert_chroma_matches_the_files()

    def test_drop_and_push_in_one_run(self):
        self.run_script("--drop-suspect", "--push")
        self.assertEqual(self.ids_in_file(), ["BOOK_000_00", "BOOK_000_01"])
        self.assert_chroma_matches_the_files()

    def test_push_does_not_clear_a_collection_from_an_older_extraction(self):
        # none of the file's chunks are in Chroma: it has to be re-embedded, not emptied
        self.store = FakeStore(["BOOK_050_00", "BOOK_051_00", "OTHER_000_00"])
        printed = self.run_script("--push")
        self.assertIn("not found in the collection", printed)
        self.assertEqual(sorted(self.store.rows), ["BOOK_050_00", "BOOK_051_00", "OTHER_000_00"])


class TestBookWithNoChunks(TagTopicsCase):
    def test_book_whose_chunks_are_all_dropped(self):
        # the summary line divided by the number of chunks, and it ran before the files were written
        self.write(self.book, [SUSPECT])
        printed = self.run_script("--drop-suspect")
        self.assertIn("BOOK       0 chunks", printed)
        self.assertEqual(self.ids_in_file(), [])

    def test_empty_chunk_file(self):
        self.write(self.book, [])
        self.assertIn("BOOK       0 chunks", self.run_script())



class TestAdvancedChapters(unittest.TestCase):
    """topic_map.yaml advanced_chapters: easy / medium questions are not
    grounded in these chapters (DS handbook ch 5, 6, 8 for Heaps)."""

    def tag(self, chapter_titles, bcfg_extra):
        bcfg = {"yaml_subject": "X", "chapter_regex": r"^(\d+)\s", "chapters": {5: ["X_01"], 3: ["X_01"]}}
        bcfg.update(bcfg_extra)
        chunks = [{"chunk_id": f"c{i}", "section_title": t, "text": "", "page_start": 1, "page_end": 2}
                  for i, t in enumerate(chapter_titles)]
        topics = {"X_01": {"subject": "X", "semester": 2}}
        return tag_topics.tag_book(chunks, "BOOK", bcfg, topics, {"X": 2}, {}, 2)

    def test_chunks_of_an_advanced_chapter_are_flagged(self):
        got = self.tag(["Preface", "3 Trees", "5 Leftist Trees"], {"advanced_chapters": [5]})
        self.assertEqual([c["advanced"] for c in got], [False, False, True])
        self.assertEqual([c["topic_id"] for c in got], ["NONE", "X_01", "X_01"])   # topic is unchanged

    def test_without_the_setting_nothing_is_advanced(self):
        self.assertEqual([c["advanced"] for c in self.tag(["5 Leftist Trees"], {})], [False])

    def test_the_flag_is_pushed_to_chroma_with_the_other_tags(self):
        self.assertIn("advanced", tag_topics.TAG_FIELDS)

    def test_the_ds_handbooks_research_chapters_are_advanced(self):
        import yaml
        cfg = yaml.safe_load((ROOT / "Extracting" / "topic_map.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cfg["books"]["DS"]["advanced_chapters"], [5, 6, 8])

if __name__ == "__main__":
    unittest.main()
