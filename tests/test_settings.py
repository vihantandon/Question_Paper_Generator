"""
Where the scripts find Chroma and Neo4j (settings.py): the terminal, else the
repo's .env file, else the defaults.

    python3 -m unittest tests.test_settings -v      (from the repo root)

No Docker. Your real .env is never read or changed here: the tests that need
a .env copy settings.py into a temporary folder, put a made-up .env next to
it, and import it there in a separate Python process.
"""
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import settings  # noqa: E402


# ---------------------------------------------------------------- the .env reader
class TestDotenvFile(unittest.TestCase):
    def test_reads_file_but_terminal_wins(self):
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / ".env"
            env.write_text('# comment\nQPG_TEST_A=from_file\nexport QPG_TEST_B="quoted"\n'
                           'QPG_TEST_C=from_file\nbad line\n', encoding="utf-8")
            os.environ["QPG_TEST_C"] = "from_terminal"
            try:
                settings.load_dotenv_file(env)
                self.assertEqual(os.environ["QPG_TEST_A"], "from_file")
                self.assertEqual(os.environ["QPG_TEST_B"], "quoted")
                self.assertEqual(os.environ["QPG_TEST_C"], "from_terminal")
            finally:
                for k in ("QPG_TEST_A", "QPG_TEST_B", "QPG_TEST_C"):
                    os.environ.pop(k, None)

    def test_a_missing_file_is_not_an_error(self):
        settings.load_dotenv_file(ROOT / "tests" / "there-is-no-such-file.env")

    def test_comment_after_a_value_is_not_part_of_the_value(self):
        # "8000 # the port" used to be kept whole, and int() then failed on it
        for line, want in (("8000 # the port", "8000"),
                           ("localhost\t# tab before the comment", "localhost"),
                           ("abc   ", "abc"),
                           ("# only a comment", ""),
                           ("", "")):
            self.assertEqual(settings.parse_dotenv_value(line), want, line)

    def test_hash_inside_a_value_is_kept(self):
        for line, want in (("pass#word", "pass#word"),              # no space before #
                           ('"pass # word"', "pass # word"),        # quoted
                           ("'pass # word' # note", "pass # word"),
                           ('"quoted"', "quoted"),
                           ('"', "")):
            self.assertEqual(settings.parse_dotenv_value(line), want, line)


# ---------------------------------------------------------------- settings.py itself
def run_python(code, folder, **variables):
    """Run `code` in a new Python process inside `folder`, with no CHROMA_* /
    NEO4J_* / LLM_* / OPENROUTER_* variable set except the ones given.
    Returns what it printed."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CHROMA_", "NEO4J_", "LLM_", "OPENROUTER_"))}
    env.update(variables, PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, "-c", code], cwd=folder, env=env,
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise AssertionError(done.stderr)
    return done.stdout.split()


SHOW = ("import settings as s; "
        "print(s.CHROMA_HOST, s.CHROMA_PORT, type(s.CHROMA_PORT).__name__, s.NEO4J_URI, s.NEO4J_PASSWORD)")
DOTENV = ('CHROMA_HOST=chroma.example   # where Chroma runs\n'
          'CHROMA_PORT=9123 # not the default\n'
          'NEO4J_URI=bolt://graph.example:7687\n'
          'NEO4J_PASSWORD="s3cret # not a comment"\n')


class TestSettingsModule(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.folder = Path(self.dir.name)
        shutil.copy(ROOT / "settings.py", self.folder)

    def test_values_come_from_the_env_file(self):
        (self.folder / ".env").write_text(DOTENV, encoding="utf-8")
        self.assertEqual(run_python(SHOW, self.folder),
                         ["chroma.example", "9123", "int", "bolt://graph.example:7687",
                          "s3cret", "#", "not", "a", "comment"])

    def test_the_terminal_wins_over_the_env_file(self):
        (self.folder / ".env").write_text(DOTENV, encoding="utf-8")
        got = run_python(SHOW, self.folder, CHROMA_PORT="7000", CHROMA_HOST="other.example")
        self.assertEqual(got[:2], ["other.example", "7000"])

    def test_defaults_without_an_env_file(self):
        self.assertEqual(run_python(SHOW, self.folder),
                         ["localhost", "8000", "int", "bolt://localhost:7687", "yourpassword"])

    def test_an_empty_value_means_the_default(self):
        (self.folder / ".env").write_text("CHROMA_PORT=\nCHROMA_HOST=   # fill me in\n", encoding="utf-8")
        self.assertEqual(run_python(SHOW, self.folder)[:2], ["localhost", "8000"])


SHOW_LLM = "import settings as s; print(s.LLM_BASE_URL, s.LLM_MODEL, s.LLM_API_KEY or '-', s.LLM_TIMEOUT)"
GROQ = "https://api.groq.com/openai/v1"


class TestLanguageModelSettings(unittest.TestCase):
    """Which service, model and key draft the questions (pipeline/llm.py)."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.folder = Path(self.dir.name)
        shutil.copy(ROOT / "settings.py", self.folder)

    def show(self, dotenv=None, **variables):
        if dotenv is not None:
            (self.folder / ".env").write_text(dotenv, encoding="utf-8")
        return run_python(SHOW_LLM, self.folder, **variables)

    def test_defaults(self):
        self.assertEqual(self.show(), ["https://openrouter.ai/api/v1", "qwen/qwen3.8-27b:free", "-", "180.0"])

    def test_the_openrouter_key_of_the_pyq_extractor_is_enough(self):
        self.assertEqual(self.show("OPENROUTER_API_KEY=sk-or-123\nLLM_API_KEY=\n")[2], "sk-or-123")

    def test_the_openrouter_key_is_not_sent_to_another_service(self):
        self.assertEqual(self.show(f"LLM_BASE_URL={GROQ}\nOPENROUTER_API_KEY=sk-or-123\n")[:3],
                         [GROQ, "qwen/qwen3.8-27b:free", "-"])

    def test_llm_settings_win(self):
        got = self.show(f"LLM_BASE_URL={GROQ}\nLLM_MODEL=qwen/qwen3.8-27b\nLLM_API_KEY=gsk-456\n"
                        "OPENROUTER_API_KEY=sk-or-123\nLLM_TIMEOUT=60\n")
        self.assertEqual(got, [GROQ, "qwen/qwen3.8-27b", "gsk-456", "60.0"])
        self.assertEqual(self.show(LLM_MODEL="from/terminal")[1], "from/terminal")


class TestImportOrder(unittest.TestCase):
    def test_pipeline_uses_the_env_file_whichever_module_is_imported_first(self):
        # pipeline/graph.py imports retrieval BEFORE pipeline.resources. retrieval used
        # to read CHROMA_* at that moment, when nothing had loaded .env yet.
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            for name in ("settings.py", "retrieval.py"):
                shutil.copy(ROOT / name, folder)
            shutil.copytree(ROOT / "pipeline", folder / "pipeline",
                            ignore=shutil.ignore_patterns("__pycache__"))
            (folder / ".env").write_text(DOTENV, encoding="utf-8")
            got = run_python("import pipeline.graph, retrieval, pipeline.resources as r; "
                             "print(retrieval.CHROMA_HOST, retrieval.CHROMA_PORT, "
                             "r.CHROMA_HOST, r.CHROMA_PORT, r.NEO4J_URI)", folder)
        self.assertEqual(got, ["chroma.example", "9123", "chroma.example", "9123",
                               "bolt://graph.example:7687"])


# ---------------------------------------------------------------- every script
def python_files():
    skip = {"tests", "venv", "__pycache__"}
    for path in sorted(ROOT.rglob("*.py")):
        parts = path.relative_to(ROOT).parts
        if not any(p in skip or p.startswith(".") for p in parts) and path.name != "settings.py":
            yield path


def is_literal(node):
    return isinstance(node, ast.Constant)


class TestNoScriptHasItsOwnChromaAddress(unittest.TestCase):
    """Seven scripts had "localhost" and 8000 written into them, so changing
    CHROMA_HOST / CHROMA_PORT in .env did nothing for them."""

    def test_the_address_is_never_written_into_a_script(self):
        found = []
        for path in python_files():
            where = str(path.relative_to(ROOT))
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=where)):
                if isinstance(node, ast.Assign) and is_literal(node.value):
                    names = {t.id for t in node.targets if isinstance(t, ast.Name)}
                    if names & {"CHROMA_HOST", "CHROMA_PORT"}:
                        found.append(f"{where}:{node.lineno} sets its own {sorted(names)}")
                if not isinstance(node, ast.Call):
                    continue
                called = getattr(node.func, "attr", getattr(node.func, "id", ""))
                literals = [is_literal(a) for a in node.args] + [is_literal(k.value) for k in node.keywords]
                if called == "HttpClient" and any(literals):
                    found.append(f"{where}:{node.lineno} HttpClient(...) with a fixed host or port")
                if (called == "add_argument" and node.args and is_literal(node.args[0])
                        and node.args[0].value in ("--host", "--port")
                        and any(k.arg == "default" and is_literal(k.value) for k in node.keywords)):
                    found.append(f"{where}:{node.lineno} {node.args[0].value} has a fixed default")
        self.assertEqual(found, [])

    def test_scripts_take_the_address_from_settings(self):
        import check_chroma
        import retrieval
        from pipeline import resources
        for module in (check_chroma, retrieval, resources):
            self.assertEqual((module.CHROMA_HOST, module.CHROMA_PORT),
                             (settings.CHROMA_HOST, settings.CHROMA_PORT), module.__name__)


if __name__ == "__main__":
    unittest.main()
