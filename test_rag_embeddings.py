import json
from sentence_transformers import SentenceTransformer

print("Loading data...")

with open("historical_cases.json", "r", encoding="utf-8") as file:
    historical_cases = json.load(file)

with open("knowledge_base.json", "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)

print("Historical cases:", len(historical_cases))
print("Knowledge base articles:", len(knowledge_base))


print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# ============================================================
# 1. HISTORICAL CASE EMBEDDINGS
# ============================================================

print("\n1. Creating historical case embeddings...")

historical_documents = [
    case["problem_text"]
    for case in historical_cases
]

print("Number of historical documents:", len(historical_documents))

for i, document in enumerate(historical_documents):
    print(f"  Case {i + 1} text length:", len(document))


historical_embeddings = model.encode(
    historical_documents,
    convert_to_tensor=True
)

print("Historical embeddings created successfully!")


# ============================================================
# 2. KNOWLEDGE BASE EMBEDDINGS
# ============================================================

print("\n2. Creating knowledge base embeddings...")

kb_documents = []

for article in knowledge_base:

    document = (
        article["title"]
        + ". "
        + article["problem"]
        + ". "
        + " ".join(article["symptoms"])
    )

    kb_documents.append(document)


print("Number of KB documents:", len(kb_documents))

for i, document in enumerate(kb_documents):
    print(f"  KB {i + 1} text length:", len(document))


kb_embeddings = model.encode(
    kb_documents,
    convert_to_tensor=True
)

print("Knowledge base embeddings created successfully!")


# ============================================================
# 3. CUSTOMER QUERY EMBEDDING
# ============================================================

print("\n3. Creating customer query embedding...")

query = (
    "My calls keep dropping and my mobile internet "
    "is very slow."
)

print("Query:", query)


query_embedding = model.encode(
    query,
    convert_to_tensor=True
)

print("Customer query embedding created successfully!")


# ============================================================
# COMPLETE
# ============================================================

print("\n==========================================")
print("ALL EMBEDDING TESTS PASSED!")
print("==========================================")