"""
Everything the nodes need from the outside world, opened ONCE.

The old integration.py opened a Neo4j connection the moment it was
imported. That makes the code hard to test (importing it needs a live
database) and hard to reuse. Here the connections are created in one place
and handed to the nodes -- "dependency injection". Tests hand in fakes
instead (see tests/test_phase1.py).
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# the repo's .env is read by settings.py, once, for every script
from settings import CHROMA_HOST, CHROMA_PORT, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

ROOT = Path(__file__).resolve().parent.parent
EMBED_MODEL = "all-MiniLM-L6-v2"      # must match the model used to embed the collections


@dataclass
class Resources:
    driver: Any                        # neo4j.Driver
    grounding_collections: list        # [book_content, tut_content]
    pyq_collection: Optional[Any]      # pyq_bank, or None if not embedded
    embed_fn: Callable[[str], list]    # text -> vector
    llm: Optional[Any] = None          # pipeline.llm.LLMClient, or None: stop after retrieve
    checker: Optional[Any] = None      # model for quality_check; None = the same as llm
    bank_collection: Optional[Any] = None   # generated_questions: accepted questions (pipeline/dedup.py)

    @property
    def tutorial_collection(self):
        """tut_content, when it is one of the grounding collections."""
        return next((c for c in self.grounding_collections if getattr(c, "name", "") == "tut_content"), None)

    def close(self):
        if self.driver is not None:
            self.driver.close()


def make_embed_fn(model_name=EMBED_MODEL):
    """Load the embedding model once. Results are cached, because the same
    topic names get embedded again and again while a paper is built."""
    os.environ.setdefault("USE_TF", "0")          # same TensorFlow workaround as the embed scripts
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    cache = {}

    def embed(text):
        if text not in cache:
            cache[text] = model.encode(text).tolist()
        return cache[text]
    return embed


def connect(embed_fn=None, llm=None, checker=None) -> Resources:
    """Open Neo4j + Chroma and load the embedding model. `llm` (from
    pipeline.llm.make_llm) is passed through as it is: make it BEFORE
    calling this, so a missing API key stops the run before the slow
    embedding model is loaded."""
    import chromadb
    from neo4j import GraphDatabase
    from pipeline.dedup import get_bank
    from retrieval import get_grounding_collections, get_pyq_collection

    from neo4j.exceptions import AuthError, ServiceUnavailable

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()   # fail now with a clear error, not mid-pipeline
    except AuthError:
        driver.close()
        raise SystemExit(
            f"Neo4j rejected user '{NEO4J_USER}' with the password from "
            f"{'.env' if (ROOT / '.env').exists() else 'the default (no .env found)'}.\n"
            f"Check NEO4J_PASSWORD in .env. Test it with:\n"
            f"  docker exec neo4j_server cypher-shell -u {NEO4J_USER} -p '<password>' 'RETURN 1'")
    except ServiceUnavailable:
        driver.close()
        raise SystemExit(f"Cannot reach Neo4j at {NEO4J_URI}. Is it running?  ->  docker compose up -d")
    # From here on the Neo4j driver is open. If anything below fails, nothing
    # is returned that could close it later, so it is closed here.
    try:
        try:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            client.heartbeat()
        except Exception as e:
            raise SystemExit(f"Cannot reach Chroma at {CHROMA_HOST}:{CHROMA_PORT} ({e}).\n"
                             f"Is it running?  ->  docker compose up -d") from None
        return Resources(
            driver=driver,
            grounding_collections=get_grounding_collections(client),
            pyq_collection=get_pyq_collection(client),
            bank_collection=get_bank(client),
            embed_fn=embed_fn or make_embed_fn(),
            llm=llm,
            checker=checker,
        )
    except BaseException:
        driver.close()
        raise