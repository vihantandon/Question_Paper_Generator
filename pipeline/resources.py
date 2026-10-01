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

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv_file(path=ROOT / ".env"):
    """Read KEY=VALUE lines from the repo's .env into os.environ.

    Docker Compose reads .env by itself, but Python does not -- without
    this, the pipeline would use the default password below instead of
    the one in .env. A variable already set in the terminal (export ...)
    wins over the file.
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
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


load_dotenv_file()      # must run before the settings below are read

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "yourpassword")
EMBED_MODEL = "all-MiniLM-L6-v2"      # must match the model used to embed the collections


@dataclass
class Resources:
    driver: Any                        # neo4j.Driver
    grounding_collections: list        # [book_content, tut_content]
    pyq_collection: Optional[Any]      # pyq_bank, or None if not embedded
    embed_fn: Callable[[str], list]    # text -> vector

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


def connect(embed_fn=None) -> Resources:
    """Open Neo4j + Chroma and load the embedding model."""
    import chromadb
    from neo4j import GraphDatabase
    from retrieval import (CHROMA_HOST, CHROMA_PORT,
                           get_grounding_collections, get_pyq_collection)

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
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return Resources(
        driver=driver,
        grounding_collections=get_grounding_collections(client),
        pyq_collection=get_pyq_collection(client),
        embed_fn=embed_fn or make_embed_fn(),
    )