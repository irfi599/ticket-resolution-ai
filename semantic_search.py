import json
from sentence_transformers import SentenceTransformer, util


# ==========================================
# LOAD HISTORICAL CASES
# ==========================================

with open(
    "historical_cases.json",
    "r",
    encoding="utf-8"
) as file:

    cases = json.load(file)


print(
    "Historical cases loaded:",
    len(cases)
)


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ==========================================
# CUSTOMER COMPLAINT
# ==========================================

query = (
    "My calls keep dropping and "
    "my mobile internet is very slow."
)


# ==========================================
# CREATE DOCUMENTS FOR SEARCH
# ==========================================

documents = [
    case["problem_text"]
    for case in cases
]


# ==========================================
# CREATE EMBEDDINGS
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
# CALCULATE SIMILARITY
# ==========================================

similarity_scores = util.cos_sim(
    query_embedding,
    document_embeddings
)[0]


# ==========================================
# RANK RESULTS
# ==========================================

results = []

for i, score in enumerate(similarity_scores):

    results.append(
        (
            float(score),
            cases[i]
        )
    )


results.sort(
    reverse=True,
    key=lambda x: x[0]
)


# ==========================================
# SELECT TOP RESULTS
# ==========================================

top_k = 2

top_results = results[:top_k]


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("       INTELLIGENT SUPPORT SEARCH")
print("==========================================")

print("\nNew Customer Complaint:")
print(query)

print(
    f"\nTop {top_k} Relevant Historical Cases:"
)


for rank, (score, case) in enumerate(
    top_results,
    start=1
):

    print(
        "\n------------------------------------------"
    )

    print(
        f"Rank: {rank}"
    )

    print(
        f"Similarity Score: {score:.4f}"
    )

    print(
        f"Case ID: {case['case_id']}"
    )

    print(
        "\nCustomer Problem:"
    )

    print(
        case["problem_text"]
    )

    print(
        "\nIssue Type:"
    )

    print(
        case["issue_type"]
    )

    print(
        "\nDevice:"
    )

    print(
        case["device"]
    )

    print(
        "\nCustomer Sentiment:"
    )

    print(
        case["sentiment"]
    )

    print(
        "\nTroubleshooting / Resolution Steps:"
    )

    if case["troubleshooting_steps"]:

        for step in case[
            "troubleshooting_steps"
        ]:

            print(
                f" - {step}"
            )

    else:

        print(
            " - No troubleshooting steps identified"
        )

    print(
        "\nOutcome:"
    )

    print(
        case["outcome"]
    )

    print(
        "\nSource Conversation:"
    )

    print(
        case["source_conversation"]
    )


print("\n==========================================")

print(
    "Retrieval completed successfully!"
)

print("==========================================")