"""
The small fixes from section C of BUG_REPORT.md, one test class per item.

    python3 -m unittest tests.test_housekeeping -v      (from the repo root)

No Docker, no embedding model: Neo4j and Chroma are stand-in objects, and
docker-compose.yml is only read as a YAML file.
"""
import contextlib
import io
import json
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Extracting"))
sys.path.insert(0, str(ROOT / "Extracting" / "pyqs_processing"))
sys.path.insert(0, str(ROOT / "prerequisite_graph" / "loader"))

try:
    import fitz  # noqa: F401
except ImportError:                 # PyMuPDF only opens real PDFs; these tests never do
    sys.modules["fitz"] = types.ModuleType("fitz")

import build_prereq_graph as loader  # noqa: E402
import extract  # noqa: E402
import split_pyq_questions as splitter  # noqa: E402
from pipeline import difficulty_blend, resources  # noqa: E402


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


def own_python_files():
    for path in sorted(ROOT.rglob("*.py")):
        parts = path.relative_to(ROOT).parts
        if not any(p in ("tests", "venv", "__pycache__") or p.startswith(".") for p in parts):
            yield path


# ---------------------------------------------------------------- script names
class TestOnlyTestsAreNamedLikeTests(unittest.TestCase):
    def test_no_script_outside_tests_is_called_test_something(self):
        # Extracting/pyqs_processing/test_pyq_embed.py was a script that loads the model
        # and queries Chroma; a test runner pointed at the repo would have imported it
        named_like_tests = [str(p.relative_to(ROOT)) for p in own_python_files()
                            if p.name.startswith("test_") or p.name.endswith("_test.py")]
        self.assertEqual(named_like_tests, [])


# ---------------------------------------------------------------- leftover code
class TestBlendModuleHasNoLeftovers(unittest.TestCase):
    def test_old_copy_of_the_node_and_its_password_are_gone(self):
        self.assertFalse(hasattr(difficulty_blend, "graph_lookup_node"))
        source = Path(difficulty_blend.__file__).read_text(encoding="utf-8")
        self.assertNotIn("yourpassword", source)


# ---------------------------------------------------------------- pipeline start-up
class FakeDriver:
    def __init__(self):
        self.closed = False

    def verify_connectivity(self):
        pass

    def close(self):
        self.closed = True


class FakeChromaClient:
    def heartbeat(self):
        return 1

    def list_collections(self):
        return []


def refuse(*args, **kwargs):
    raise ValueError("Could not connect to a Chroma server. Are you sure it is running?")


class TestConnect(unittest.TestCase):
    def connect(self, http_client, **kwargs):
        driver = FakeDriver()
        chroma = types.SimpleNamespace(HttpClient=http_client)
        with mock.patch("neo4j.GraphDatabase.driver", return_value=driver), \
                mock.patch.dict(sys.modules, {"chromadb": chroma}):
            try:
                return driver, quiet(resources.connect, **kwargs)
            except BaseException as error:             # SystemExit is not an Exception
                return driver, error

    def test_chroma_down_gives_a_clear_message_and_closes_neo4j(self):
        driver, error = self.connect(refuse, embed_fn=lambda text: [0.0])
        self.assertIsInstance(error, SystemExit)
        self.assertIn("Cannot reach Chroma", str(error))
        self.assertIn("docker compose up -d", str(error))
        self.assertTrue(driver.closed)                 # used to stay open

    def test_neo4j_is_closed_when_a_later_step_fails_too(self):
        def no_model():
            raise RuntimeError("model file missing")
        with mock.patch.object(resources, "make_embed_fn", no_model):
            driver, error = self.connect(lambda host, port: FakeChromaClient())
        self.assertIsInstance(error, RuntimeError)
        self.assertTrue(driver.closed)

    def test_everything_up_returns_open_resources(self):
        driver, res = self.connect(lambda host, port: FakeChromaClient(), embed_fn=lambda text: [0.0])
        self.assertIs(res.driver, driver)
        self.assertFalse(driver.closed)


# ---------------------------------------------------------------- docker-compose.yml
class TestComposeFile(unittest.TestCase):
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]

    def test_every_image_names_a_version(self):
        # chromadb/chroma:latest could change under the data in ./chroma_data
        for name, service in self.services.items():
            if "image" in service:
                image, _, tag = service["image"].rpartition(":")
                self.assertTrue(image and tag and tag != "latest", f"{name}: {service['image']}")

    def test_ports_are_open_to_this_machine_only(self):
        # no database here has a real password, so none may listen on the network
        for name, service in self.services.items():
            for port in service.get("ports", []):
                self.assertTrue(str(port).startswith("127.0.0.1:"), f"{name}: {port}")


# ---------------------------------------------------------------- graph loader
def needs(topic, requires):
    return {"topic": topic, "requires": requires, "strength": "hard"}


class FakeTx:
    def __init__(self, fail_at=None):
        self.queries, self.fail_at = [], fail_at

    def run(self, query, **params):
        if len(self.queries) == self.fail_at:
            raise RuntimeError("Neo4j went away")
        self.queries.append(" ".join(query.split()))
        return types.SimpleNamespace(single=lambda: {"deleted": 7})


class FakeSession:
    """execute_write is one transaction: its queries only count as saved
    (committed) if the function finishes without an error."""

    def __init__(self, fail_at=None):
        self.committed, self.fail_at = [], fail_at

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_write(self, fn, *args):
        tx = FakeTx(self.fail_at)
        result = fn(tx, *args)
        self.committed.append(tx.queries)
        return result

    def run(self, query, **params):
        return []


TOPICS = [{"id": t, "name": t, "subject": "X", "semester": 1} for t in ("A", "B")]
COS = [{"co_id": "X:CO1", "short_id": "CO1", "subject": "X", "description": "one"}]
LINKS = [{"topic_id": "A", "co_id": "X:CO1"}, {"topic_id": "B", "co_id": "X:CO1"}]


class TestGraphLoader(unittest.TestCase):
    def test_prerequisite_loops_are_found(self):
        self.assertEqual(loader.find_prerequisite_loops([needs("A", "B"), needs("B", "C")]), [])
        self.assertEqual(loader.find_prerequisite_loops([needs("A", "A")]), [["A", "A"]])
        self.assertEqual(loader.find_prerequisite_loops(
            [needs("A", "B"), needs("B", "C"), needs("C", "A"), needs("D", "A")]), [["A", "B", "C", "A"]])

    def test_a_loop_stops_the_load_before_neo4j_is_touched(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit):
            loader.validate(TOPICS, [needs("A", "B"), needs("B", "A")], COS, LINKS)
        self.assertIn("go in a circle: A -> B -> A", errors.getvalue())

    def test_the_real_subject_files_are_valid(self):
        data = quiet(loader.load_subject_files, str(ROOT / "prerequisite_graph" / "subjects"))
        quiet(loader.validate, *data)                     # would raise SystemExit
        self.assertEqual(loader.find_prerequisite_loops(data[1]), [])

    def build(self, session, reset):
        driver = types.SimpleNamespace(session=lambda: session, close=lambda: None)
        with mock.patch.object(loader.GraphDatabase, "driver", return_value=driver), \
                mock.patch.object(loader, "load_subject_files",
                                  lambda folder: (TOPICS, [needs("B", "A")], COS, LINKS)):
            quiet(loader.build_graph, "subjects", reset=reset)

    def test_reset_and_reload_are_one_transaction(self):
        session = FakeSession()
        self.build(session, reset=True)
        self.assertEqual(len(session.committed), 1)       # was 1 + one per node and edge
        queries = session.committed[0]
        self.assertIn("DETACH DELETE", queries[0])
        self.assertEqual(len(queries), 1 + len(TOPICS) + len(COS) + 1 + len(LINKS))

    def test_a_load_that_fails_half_way_saves_nothing(self):
        session = FakeSession(fail_at=3)                  # after the delete and two topics
        with self.assertRaises(RuntimeError):
            self.build(session, reset=True)
        self.assertEqual(session.committed, [])           # the delete is undone with the rest


# ---------------------------------------------------------------- stored paths
class OnePageBook:
    def get_toc(self):
        return [[1, "1 Intro", 1]]

    def __len__(self):
        return 1

    def __getitem__(self, page):
        from tests.fake_pdf import FakePage       # plain text + the "dict" view text_cleanup reads
        return FakePage("a line of the book\n" * 40)


PAPER = ("Jaypee Institute of Information Technology, Noida\nTest-1 Examination, ODD 2024\n"
         "Course Title: Data Structures\nCourse Code: 15B11CI311\nMaximum Marks: 20\n\n"
         "Q1. Explain how a stack is built on an array, and show push and pop on it. [CO2, 5 Marks]\n")


class TestStoredPaths(unittest.TestCase):
    def test_book_path_is_stored_relative_to_the_repo(self):
        self.assertEqual(extract.display_path(ROOT / "Books" / "DS.pdf"), "Books/DS.pdf")
        self.assertEqual(extract.display_path("/somewhere/else/Notes.pdf"), "Notes.pdf")
        with mock.patch.object(extract.fitz, "open", lambda path: OnePageBook(), create=True):
            chunks = extract.process_pdf(str(ROOT / "Books" / "FAKE.pdf"), subject="FAKE")
        self.assertEqual({c.source_file for c in chunks}, {"Books/FAKE.pdf"})

    def test_committed_chunk_files_hold_no_local_folder_path(self):
        # a re-extraction had written /home/<user>/... into every chunk
        for path in sorted((ROOT / "Extracting" / "Processed").glob("*_chunks.jsonl")):
            if path.name.endswith("_tutorial_chunks.jsonl"):
                continue
            stored = set(re.findall(r'"source_file": ("(?:[^"\\]|\\.)*")', path.read_text(encoding="utf-8")))
            for value in map(json.loads, stored):
                self.assertFalse(Path(value).is_absolute() or "\\" in value or ":" in value,
                                 f"{path.name}: {value}")

    def test_committed_pyq_data_spells_paths_one_way(self):
        # the first ten papers were extracted on Windows: "PYQs\\T1 Dsa.pdf"
        processed = ROOT / "Extracting" / "Processed"
        paths = set()
        for path in sorted((processed / "pyq_structured").glob("*_pyq.json")):
            paths |= {r["metadata"]["source"] for r in json.loads(path.read_text(encoding="utf-8"))}
        for path in sorted((processed / "pyq_pages").glob("*_pages.jsonl")):
            paths |= {json.loads(line)["source_file"]
                      for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        self.assertTrue(paths)
        for value in sorted(paths):
            self.assertRegex(value, r"^PYQs/[^\\/]+\.pdf$")

    def test_splitter_stores_a_windows_path_with_forward_slashes(self):
        yamls = splitter.load_subject_yamls(sorted(str(p) for p in splitter.SUBJECTS_DIR.glob("*.y*ml")))
        with tempfile.TemporaryDirectory() as d:
            pages = Path(d) / "T1 Dsa_pages.jsonl"
            pages.write_text(json.dumps({"source_file": "PYQs\\T1 Dsa.pdf", "page_num": 1,
                                         "text": PAPER}) + "\n", encoding="utf-8")
            records = quiet(splitter.process_pages_file, pages, yamls)
        self.assertGreaterEqual(len(records), 2)          # header + the question
        self.assertEqual({r["metadata"]["source"] for r in records}, {"PYQs/T1 Dsa.pdf"})


# ---------------------------------------------------------------- embed_chunks.py
class TestEmbedChunksSkipsTensorflow(unittest.TestCase):
    def test_workaround_is_set_before_the_model_library_is_imported(self):
        source = (ROOT / "Embedding" / "embed_chunks.py").read_text(encoding="utf-8")
        self.assertIn('os.environ["USE_TF"] = "0"', source)
        self.assertLess(source.index('os.environ["USE_TF"] = "0"'),
                        source.index("from sentence_transformers import"))


if __name__ == "__main__":
    unittest.main()
