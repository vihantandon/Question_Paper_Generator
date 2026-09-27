import chromadb

client = chromadb.HttpClient(host="localhost", port=8000)
collection = []
collection.append(client.get_collection("book_content"))
collection.append(client.get_collection("tut_content"))

for subject in ["APS", "DS", "SDF-1", "SDF-2"]:
    for i in collection:
        result = i.get(where={"subject": subject}, limit=1000000)
        print(subject, "→", len(result["ids"]), "vectors")