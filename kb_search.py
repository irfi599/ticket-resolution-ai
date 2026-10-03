import json
from sentence_transformers import SentenceTransformer, util


# ==========================================
# 1. LOAD KNOWLEDGE BASE
# ==========================================

with open("knowledge_base.json", "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)

print("Knowledge base articles loaded:", len(knowledge_base))


# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# ==========================================
# 3. PREPARE KNOWLEDGE BASE DOCUMENTS
# ==========================================

documents = []

for article in knowledge_base:

    document = (
        article["title"]
        + ". "
        + article["problem"]
        + ". "
        + " ".join(article["symptoms"])
    )

    documents.append(document)


# ==========================================
# 4. NEW CUSTOMER COMPLAINT
# ==========================================

query = "My mobile internet is very slow and my calls keep dropping."


# ==========================================
# 5. CREATE EMBEDDINGS
# ==========================================

document_embeddings = model.encode(
    documents,
    convert_to_tensor=True
)

query_embedding = model.encode(
    query,
    convert_to_tensor=True
)


# ==========================================
# 6. CALCULATE SIMILARITY
# ==========================================

similarity_scores = util.cos_sim(
    query_embedding,
    document_embeddings
)[0]


# ==========================================
# 7. COMBINE SCORES WITH ARTICLES
# ==========================================

results = []

for i, score in enumerate(similarity_scores):

    results.append(
        (
            float(score),
            knowledge_base[i]
        )
    )


# ==========================================
# 8. SORT BY SIMILARITY
# ==========================================

results.sort(
    reverse=True,
    key=lambda x: x[0]
)


# ==========================================
# 9. RETRIEVE TOP 2 ARTICLES
# ==========================================

top_k = 2

top_results = results[:top_k]


# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("       KNOWLEDGE BASE SEARCH")
print("==========================================")

print("\nCustomer Complaint:")
print(query)

print(f"\nTop {top_k} Relevant Knowledge Base Articles:")


for rank, (score, article) in enumerate(
    top_results,
    start=1
):

    print("\n------------------------------------------")

    print(f"Rank: {rank}")

    print(f"Similarity Score: {score:.4f}")

    print(f"Article ID: {article['article_id']}")

    print(f"Title: {article['title']}")

    print("\nProblem:")
    print(article["problem"])

    print("\nRecommended Steps:")

    for step in article["steps"]:
        print(f" - {step}")

    print("\nEscalation Condition:")
    print(article["escalation_condition"])


print("\n==========================================")
print("Knowledge base search completed!")
print("==========================================")