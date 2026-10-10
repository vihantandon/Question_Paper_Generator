import chromadb

from settings import CHROMA_HOST, CHROMA_PORT

client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = []
collection.append(client.get_collection("book_content"))
collection.append(client.get_collection("tut_content"))

for subject in ["APS", "DS", "SDF-1", "SDF-2"]:
    for i in collection:
        result = i.get(where={"subject": subject}, limit=1000000)
        print(subject, "→", len(result["ids"]), "vectors")