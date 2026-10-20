"""
settings.py -- where the databases and the language model are. ONE place,
read by every script.

    from settings import CHROMA_HOST, CHROMA_PORT
    from settings import LLM_BASE_URL, LLM_MODEL, LLM_API_KEY

Each value is taken from the first of these that has it:
    1. a variable set in the terminal      export CHROMA_PORT=9000
    2. the repo's .env file                CHROMA_PORT=9000
    3. the default below                   (what docker-compose.yml starts)

The .env file is read here, when this module is first imported, so it does
not matter which module a script imports first. (retrieval.py used to read
CHROMA_HOST / CHROMA_PORT itself, before pipeline/resources.py had loaded
.env, and the other scripts had "localhost" and 8000 written into them -- so
a Chroma address set in .env was ignored everywhere.)
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_dotenv_value(raw):
    """The VALUE of a KEY=VALUE line, read the way Docker Compose reads it:

        abc # note        -> abc       " #" starts a comment
        "a # b"  # note   -> a # b     inside quotes, # belongs to the value
        abc#def           -> abc#def   no space before #: not a comment
    """
    raw = raw.strip()
    if raw[:1] in ('"', "'"):
        end = raw.find(raw[0], 1)
        if end != -1:
            return raw[1:end]
    return re.split(r"(?:^|\s+)#", raw, maxsplit=1)[0].strip().strip('"').strip("'")


def load_dotenv_file(path=ROOT / ".env"):
    """Read KEY=VALUE lines from the repo's .env into os.environ.

    Docker Compose reads .env by itself, but Python does not -- without
    this, the scripts would use the defaults below instead of the values
    in .env. A variable already set in the terminal (export ...) wins over
    the file.
    """
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        if key:
            os.environ.setdefault(key, parse_dotenv_value(value))


def _setting(name, default):
    """An empty value (a bare KEY= line) counts as not set, as it does for
    docker compose's ${KEY:-default}."""
    return os.environ.get(name) or default


load_dotenv_file()      # must run before the settings below are read

CHROMA_HOST = _setting("CHROMA_HOST", "localhost")
CHROMA_PORT = int(_setting("CHROMA_PORT", "8000"))     # docker-compose exposes 8000
NEO4J_URI = _setting("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = _setting("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = _setting("NEO4J_PASSWORD", "yourpassword")

# --- the language model that drafts questions (pipeline/llm.py) -------------
# Any service that speaks the OpenAI chat API works: OpenRouter, Groq, a local
# Ollama ... Only these three values change. Model ids on free tiers come and
# go (Groq shut down qwen/qwen3-32b in July 2026), so the model is a setting,
# not something written into the code.
LLM_BASE_URL = _setting("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = _setting("LLM_MODEL", "qwen/qwen3.8-27b:free")
# With OpenRouter, the OPENROUTER_API_KEY the PYQ extractor already uses is
# enough; LLM_API_KEY is only needed for another service (or another key).
LLM_API_KEY = _setting("LLM_API_KEY", "") or (
    _setting("OPENROUTER_API_KEY", "") if "openrouter.ai" in LLM_BASE_URL else "")
LLM_TIMEOUT = float(_setting("LLM_TIMEOUT", "180"))     # seconds; free models can be slow
# The model that checks drafts (pipeline/nodes/quality_check.py). Empty = the
# drafting model. A different model is better at finding faults in a draft;
# it uses the same service and key.
LLM_CHECK_MODEL = _setting("LLM_CHECK_MODEL", "")
# How strictly a question must stick to the retrieved sources
# (pipeline/nodes/quality_check.py):
#   balanced -- the question's facts must be correct (from a source, or standard
#               for the course); the answer outline is only checked for mistakes
#   strict   -- every fact in the question AND its answer quoted from a source
GROUNDING = _setting("GROUNDING", "balanced").strip().lower()
