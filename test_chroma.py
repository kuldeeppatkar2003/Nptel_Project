# import chromadb

# client = chromadb.Client(
#     settings=chromadb.Settings(
#         persist_directory="./chroma_db"
#     )
# )

# collection = client.get_collection(name="nptel_ml")

# print("Vector count:", collection.count())

# results = collection.query(
#     query_texts=["What is machine learning?"],
#     n_results=2,
#     include=["documents"]
# )

# print("\nRetrieved documents:")
# for doc in results["documents"][0]:
#     print("-", doc[:100])
import chromadb

client = chromadb.Client(
    settings=chromadb.Settings(
        persist_directory="./chroma_db"
    )
)

print("Existing collections:", [c.name for c in client.list_collections()])

collection = client.get_collection(name="nptel_ml")

print("Vector count:", collection.count())
