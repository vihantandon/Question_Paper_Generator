import os
import sys
from pathlib import Path
os.environ["USE_TF"] = "0"
import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))   # repo root: settings.py
from settings import CHROMA_HOST, CHROMA_PORT

model = SentenceTransformer("all-MiniLM-L6-v2")
coll = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT).get_collection("pyq_bank")

res = coll.query(
    query_embeddings=[model.encode("minimum spanning tree of a weighted graph").tolist()],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)
for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(m["subject"], m["year"], m["exam"], f"Q{m['question_number']}", round(dist, 3))
    print("  ", d[:100].replace("\n", " "))