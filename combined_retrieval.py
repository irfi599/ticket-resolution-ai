import json
from sentence_transformers import SentenceTransformer, util


# ==========================================
# 1. LOAD HISTORICAL CASES
# ==========================================

with open("historical_cases.json", "r", encoding="utf-8") as file:
    historical_cases = json.load(file)


# ==========================================
# 2. LOAD KNOWLEDGE BASE
# ==========================================

with open("knowledge_base.json", "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)


print("Historical cases loaded:", len(historical_cases))
print("Knowledge base articles loaded:", len(knowledge_base))


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# ==========================================
# 4. NEW CUSTOMER COMPLAINT
# ==========================================

query = "My calls keep dropping and my mobile internet is very slow."


# ==========================================
# 5. PREPARE HISTORICAL CASE DOCUMENTS
# ==========================================

historical_documents = [
    case["customer_text"]
    for case in historical_cases
]


# ==========================================
# 6. PREPARE KNOWLEDGE BASE DOCUMENTS
# ==========================================

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


# ==========================================
# 7. CREATE EMBEDDINGS
# ==========================================

print("\nCreating embeddings...")

historical_embeddings = model.encode(
    historical_documents,
    convert_to_tensor=True
)

kb_embeddings = model.encode(
    kb_documents,
    convert_to_tensor=True
)

query_embedding = model.encode(
    query,
    convert_to_tensor=True
)


# ==========================================
# 8. SEARCH HISTORICAL CASES
# ==========================================

historical_scores = util.cos_sim(
    query_embedding,
    historical_embeddings
)[0]


historical_results = []

for i, score in enumerate(historical_scores):

    historical_results.append(
        (
            float(score),
            historical_cases[i]
        )
    )


historical_results.sort(
    reverse=True,
    key=lambda x: x[0]
)


# ==========================================
# 9. SEARCH KNOWLEDGE BASE
# ==========================================

kb_scores = util.cos_sim(
    query_embedding,
    kb_embeddings
)[0]


kb_results = []

for i, score in enumerate(kb_scores):

    kb_results.append(
        (
            float(score),
            knowledge_base[i]
        )
    )


kb_results.sort(
    reverse=True,
    key=lambda x: x[0]
)


# ==========================================
# 10. SELECT TOP RESULTS
# ==========================================

top_historical = historical_results[:2]

top_kb = kb_results[:2]


# ==========================================
# 11. DISPLAY RETRIEVED EVIDENCE
# ==========================================

print("\n==========================================")
print("       COMBINED RETRIEVAL SYSTEM")
print("==========================================")

print("\nCustomer Complaint:")
print(query)


# ==========================================
# HISTORICAL CASE RESULTS
# ==========================================

print("\n\n========== HISTORICAL CASES ==========")


for rank, (score, case) in enumerate(
    top_historical,
    start=1
):

    print("\n------------------------------------------")

    print(f"Rank: {rank}")

    print(f"Similarity: {score:.4f}")

    print(f"Case ID: {case['case_id']}")

    print(f"Issue: {case['issue_type']}")

    print(f"Device: {case['device']}")

    print(f"Sentiment: {case['sentiment']}")

    print("\nResolution Steps:")

    for step in case["troubleshooting_steps"]:
        print(f" - {step}")

    print(f"\nOutcome: {case['outcome']}")

    print(
        f"\nSource: Historical Case {case['case_id']}"
    )


# ==========================================
# KNOWLEDGE BASE RESULTS
# ==========================================

print("\n\n========== KNOWLEDGE BASE ==========")


for rank, (score, article) in enumerate(
    top_kb,
    start=1
):

    print("\n------------------------------------------")

    print(f"Rank: {rank}")

    print(f"Similarity: {score:.4f}")

    print(f"Article ID: {article['article_id']}")

    print(f"Title: {article['title']}")

    print("\nRecommended Steps:")

    for step in article["steps"]:
        print(f" - {step}")

    print(
        f"\nEscalation: {article['escalation_condition']}"
    )

    print(
        f"\nSource: {article['article_id']}"
    )


# ==========================================
# END
# ==========================================

print("\n==========================================")
print("Combined retrieval completed!")
print("==========================================")