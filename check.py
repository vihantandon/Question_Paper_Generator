import chromadb

client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_collection("book_content")

for subject in ["APS", "DS", "SDF-1", "SDF-2"]:
    result = collection.get(where={"subject": subject}, limit=1000000)
    print(subject, "→", len(result["ids"]), "vectors")