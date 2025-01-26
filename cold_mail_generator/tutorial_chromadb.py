import chromadb

client = chromadb.Client()
collection = client.create_collection(name="my_collection")
collection.add(
    documents=[
        "This document is about New York",
        "This document is about Delhi"
    ],
    ids = ['id1', 'id2']
)

all_docs = collection.get()
print(f"all_docs => {all_docs}")