"""
Re-running an embed script must leave Chroma matching the chunk files: a
vector whose chunk (or whose topic_id on that chunk) is gone from the files
has to be deleted, or retrieval keeps serving it.

    python3 -m unittest tests.test_embed_sync -v      (from the repo root)

No Docker and no embedding model: FakeStore is an in-memory Chroma
collection, FakeModel returns made-up vectors, and the chunk files are
written to a temporary folder.
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
sys.path.insert(0, str(ROOT / "Embedding"))

import check_chroma  # noqa: E402
import embed_tutorial_chunks  # noqa: E402
import stale_vectors  # noqa: E402


class FakeStore:
    """Just enough of a Chroma collection: upsert / get / delete / count.
    rows: {id: metadata}. Like the real one, delete() refuses an empty list."""

    def __init__(self, rows=None):
        self.rows = dict(rows or {})
        self.deleted_batches = []

    def upsert(self, ids, embeddings, documents, metadatas):
        self.rows.update(zip(ids, metadatas))

    def get(self, where=None, include=None, limit=None, offset=0):
        ids = [i for i, m in self.rows.items()
               if not where or all(m.get(k) == v for k, v in where.items())]
        ids = ids[offset:offset + limit] if limit else ids[offset:]
        out = {"ids": ids}
        if include and "metadatas" in include:
            out["metadatas"] = [self.rows[i] for i in ids]
        return out

    def delete(self, ids):
        if not ids:
            raise ValueError("Expected IDs to be a non-empty list, got 0 IDs in delete.")
        self.deleted_batches.append(list(ids))
        for i in ids:
            del self.rows[i]

    def count(self):
        return len(self.rows)


class FakeClient:
    def __init__(self, **collections):
        self.collections = collections

    def get_or_create_collection(self, name):
        return self.collections.setdefault(name, FakeStore())

    def get_collection(self, name):
        return self.collections[name]

    def list_collections(self):
        return list(self.collections)


class FakeVectors(list):
    def tolist(self):
        return list(self)


class FakeModel:
    def __init__(self, name):
        pass

    def encode(self, texts, show_progress_bar=False):
        return FakeVectors([[float(len(t)), 0.0] for t in texts])


def fake_modules(client):
    """Stand-ins for the three packages the embed scripts import."""
    return {
        "chromadb": types.SimpleNamespace(HttpClient=lambda host, port: client),
        "sentence_transformers": types.SimpleNamespace(SentenceTransformer=FakeModel),
        "tqdm": types.SimpleNamespace(tqdm=lambda items, **kwargs: items),
    }


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        fn(*args, **kwargs)
    return out.getvalue()


# ---------------------------------------------------------------- the helper
class TestRemoveStaleVectors(unittest.TestCase):
    def test_only_ids_missing_from_the_files_are_deleted(self):
        store = FakeStore({"DS_000_00": {}, "DS_000_01": {}, "DS_009_00": {}})
        gone = stale_vectors.remove_stale_vectors(store, ["DS_000_00", "DS_000_01", "DS_001_00"])
        self.assertEqual(gone, ["DS_009_00"])
        self.assertEqual(sorted(store.rows), ["DS_000_00", "DS_000_01"])

    def test_where_limits_the_cleanup_to_one_subject(self):
        store = FakeStore({"APS_old": {"subject": "APS"}, "APS_new": {"subject": "APS"},
                           "DS_old": {"subject": "DS"}})
        gone = stale_vectors.remove_stale_vectors(store, ["APS_new"], where={"subject": "APS"})
        self.assertEqual(gone, ["APS_old"])
        self.assertIn("DS_old", store.rows)            # another subject's rows are never touched

    def test_nothing_stale_means_no_delete_call(self):
        store = FakeStore({"a": {}, "b": {}})
        self.assertEqual(stale_vectors.remove_stale_vectors(store, ["a", "b"]), [])
        self.assertEqual(store.deleted_batches, [])    # real Chroma rejects delete(ids=[])

    def test_no_chunks_loaded_never_empties_the_collection(self):
        store = FakeStore({"a": {}, "b": {}})
        self.assertEqual(stale_vectors.remove_stale_vectors(store, []), [])
        self.assertEqual(store.count(), 2)

    def test_big_collections_are_read_and_deleted_in_pages(self):
        store = FakeStore({f"id{i:02d}": {} for i in range(11)})
        with mock.patch.object(stale_vectors, "PAGE", 4):
            gone = stale_vectors.remove_stale_vectors(store, ["id00"])
        self.assertEqual(len(gone), 10)
        self.assertEqual([len(b) for b in store.deleted_batches], [4, 4, 2])
        self.assertEqual(list(store.rows), ["id00"])


# ---------------------------------------------------------------- books
def book_chunk(chunk_id, subject="DS", text="some book text"):
    return {"chunk_id": chunk_id, "subject": subject, "source_file": f"Books/{subject}.pdf",
            "section_title": "1 Intro", "page_start": 1, "page_end": 2, "text": text,
            "topic_id": "DS_01", "needs_review": False}


def run_embed_chunks(folder, store):
    """Run Embedding/embed_chunks.py's main() against `folder` and `store`."""
    client = FakeClient(book_content=store)
    with mock.patch.dict(sys.modules, fake_modules(client)):
        sys.modules.pop("embed_chunks", None)
        import embed_chunks                     # binds the fake chromadb / model / tqdm
        with mock.patch.object(embed_chunks, "PROCESSED_DIR", Path(folder)):
            return quiet(embed_chunks.main)


class TestEmbedBookChunks(unittest.TestCase):
    def test_re_extraction_removes_the_old_ids(self):
        # the book was re-chunked: DS_009_00 and DS_010_00 no longer exist in the file
        store = FakeStore({"DS_000_00": {"subject": "DS", "topic_id": "DS_01"},
                           "DS_009_00": {"subject": "DS", "topic_id": "DS_04"},
                           "DS_010_00": {"subject": "DS", "topic_id": "DS_04"}})
        with tempfile.TemporaryDirectory() as d:
            write_jsonl(Path(d) / "DS_chunks.jsonl", [book_chunk("DS_000_00"), book_chunk("DS_000_01")])
            write_jsonl(Path(d) / "DS_tutorial_chunks.jsonl", [{"chunk_id": "DS_tut_chunk_001"}])
            printed = run_embed_chunks(d, store)
        self.assertEqual(sorted(store.rows), ["DS_000_00", "DS_000_01"])
        self.assertIn("Removed 2 stale vector(s)", printed)

    def test_nothing_is_deleted_when_chroma_already_matches(self):
        store = FakeStore({"DS_000_00": {"subject": "DS"}})
        with tempfile.TemporaryDirectory() as d:
            write_jsonl(Path(d) / "DS_chunks.jsonl", [book_chunk("DS_000_00")])
            printed = run_embed_chunks(d, store)
        self.assertEqual(store.deleted_batches, [])
        self.assertNotIn("stale", printed)

    def test_no_chunk_files_leaves_chroma_alone(self):
        store = FakeStore({"DS_000_00": {"subject": "DS"}})
        with tempfile.TemporaryDirectory() as d:
            run_embed_chunks(d, store)
        self.assertEqual(list(store.rows), ["DS_000_00"])


# ---------------------------------------------------------------- tutorials
def tutorial_chunk(chunk_id, topic_ids):
    return {"chunk_id": chunk_id, "description": f"Q1. question text of {chunk_id}",
            "topics": topic_ids, "topic_ids": topic_ids, "cos": ["CO1"], "co_details": []}


def run_embed_tutorials(folder, store, *argv):
    client = FakeClient(tut_content=store)
    with mock.patch.dict(sys.modules, fake_modules(client)), \
            mock.patch.object(embed_tutorial_chunks, "PROCESSED_DIR", Path(folder)), \
            mock.patch.object(sys, "argv", ["embed_tutorial_chunks.py", *argv]):
        return quiet(embed_tutorial_chunks.main)


class TestEmbedTutorialChunks(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        folder = Path(self.dir.name)
        # chunk 007 used to be tagged DS_02 + DS_01 and is now DS_02 only;
        # chunk 009 no longer exists
        write_jsonl(folder / "DS_tutorial_chunks.jsonl", [tutorial_chunk("DS_tut_chunk_007", ["DS_02"])])
        write_jsonl(folder / "APS_tutorial_chunks.jsonl", [tutorial_chunk("APS_tut_chunk_001", ["ALGO_06"])])
        self.store = FakeStore({
            "DS_tut_chunk_007__DS_02": {"subject": "DS", "topic_id": "DS_02"},
            "DS_tut_chunk_007__DS_01": {"subject": "DS", "topic_id": "DS_01"},
            "DS_tut_chunk_009__DS_03": {"subject": "DS", "topic_id": "DS_03"},
            "APS_tut_chunk_004__ALGO_07": {"subject": "APS", "topic_id": "ALGO_07"},
        })

    def test_re_tagging_removes_the_row_of_the_dropped_topic(self):
        printed = run_embed_tutorials(self.dir.name, self.store)
        self.assertEqual(sorted(self.store.rows),
                         ["APS_tut_chunk_001__ALGO_06", "DS_tut_chunk_007__DS_02"])
        self.assertIn("Removed 3 stale vector(s)", printed)

    def test_subject_option_cleans_only_that_subject(self):
        run_embed_tutorials(self.dir.name, self.store, "--subject", "APS")
        self.assertEqual(sorted(self.store.rows),
                         ["APS_tut_chunk_001__ALGO_06",          # embedded now
                          "DS_tut_chunk_007__DS_01",             # DS rows: not this run's business
                          "DS_tut_chunk_007__DS_02",
                          "DS_tut_chunk_009__DS_03"])

    def test_dry_run_changes_nothing(self):
        before = dict(self.store.rows)
        run_embed_tutorials(self.dir.name, self.store, "--dry-run")
        self.assertEqual(self.store.rows, before)


# ---------------------------------------------------------------- check_chroma.py
class TestCheckChromaReportsStaleVectors(unittest.TestCase):
    def report(self, name, script, **kwargs):
        client = FakeClient(**{name: FakeStore({"keep": {"subject": "DS", "topic_id": "DS_01"},
                                                "leftover": {"subject": "DS", "topic_id": "DS_04"}})})
        return quiet(check_chroma.check_collection, client, name, {"keep"}, script, **kwargs)

    def test_leftovers_are_a_status_with_the_command_that_fixes_them(self):
        printed = self.report("book_content", "Embedding/embed_chunks.py")
        self.assertIn("STATUS            : STALE - 1 vector(s)", printed)
        self.assertIn("re-run: python Embedding/embed_chunks.py", printed)
        self.assertNotIn("OK - Chroma matches", printed)

    def test_pyq_bank_needs_reset(self):
        printed = self.report("pyq_bank", "Embedding/embed_pyqs.py",
                              stale_fix="Embedding/embed_pyqs.py --reset")
        self.assertIn("re-run: python Embedding/embed_pyqs.py --reset", printed)


if __name__ == "__main__":
    unittest.main()
