import os
os.environ["USE_TF"] = "0"
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
coll = chromadb.HttpClient(host="localhost", port=8000).get_collection("pyq_bank")

res = coll.query(
    query_embeddings=[model.encode("minimum spanning tree of a weighted graph").tolist()],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)
for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(m["subject"], m["year"], m["exam"], f"Q{m['question_number']}", round(dist, 3))
    print("  ", d[:100].replace("\n", " "))