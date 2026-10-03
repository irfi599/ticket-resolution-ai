import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 2


# ============================================================
# LOAD DATA
# ============================================================

print("Loading evaluation data...")

with open("historical_cases.json", "r", encoding="utf-8") as file:
    historical_cases = json.load(file)

with open("knowledge_base.json", "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)

print("Historical cases:", len(historical_cases))
print("Knowledge base articles:", len(knowledge_base))


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# ============================================================
# PREPARE DOCUMENTS
# ============================================================

historical_documents = [
    case["problem_text"]
    for case in historical_cases
]


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


print("\nCreating document embeddings...")

historical_embeddings = model.encode(
    historical_documents
)

kb_embeddings = model.encode(
    kb_documents
)

print("Document embeddings created!")


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):

    numerator = np.dot(
        vector_a,
        vector_b
    )

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


# ============================================================
# EVALUATION TEST CASES
# ============================================================

test_cases = [

    {
        "id": "TEST001",
        "query": "My calls keep dropping and my mobile internet is very slow.",
        "expected_kb": ["KB001", "KB002"],
        "expected_steps": [
            "Restart the phone",
            "Check for software updates",
            "Switch network mode",
            "Reset network settings"
        ]
    },

    {
        "id": "TEST002",
        "query": "My phone has very poor signal and the reception keeps dropping.",
        "expected_kb": ["KB003"],
        "expected_steps": [
            "Restart the phone",
            "Check the current network mode",
            "Move to an area with better signal coverage",
            "Reset network settings"
        ]
    },

    {
        "id": "TEST003",
        "query": "My mobile data is extremely slow and apps are having trouble connecting.",
        "expected_kb": ["KB002"],
        "expected_steps": [
            "Restart the phone",
            "Check for software updates",
            "Switch network mode",
            "Reset network settings"
        ]
    },

    {
        "id": "TEST004",
        "query": "My calls disconnect unexpectedly and keep dropping.",
        "expected_kb": ["KB001"],
        "expected_steps": [
            "Restart the phone",
            "Check for software updates",
            "Switch network mode",
            "Reset network settings"
        ]
    },

    {
        "id": "TEST005",
        "query": "I have repeated network problems even after trying the normal troubleshooting steps.",
        "expected_kb": ["KB004"],
        "expected_steps": [
            "Confirm that standard troubleshooting has been attempted",
            "Record the customer's symptoms",
            "Escalate the issue to network support"
        ]
    }
]


# ============================================================
# EVALUATION METRICS
# ============================================================

retrieval_scores = []

source_correct = 0

step_coverage_scores = []

latencies = []


print("\n==========================================")
print("       SYSTEM EVALUATION")
print("==========================================")

print(
    "\nNumber of test cases:",
    len(test_cases)
)


# ============================================================
# RUN EVALUATION
# ============================================================

for test in test_cases:

    print("\n------------------------------------------")

    print(
        "Test:",
        test["id"]
    )

    print(
        "Query:",
        test["query"]
    )


    # --------------------------------------------------------
    # START TIMER
    # --------------------------------------------------------

    start_time = time.perf_counter()


    # --------------------------------------------------------
    # CREATE QUERY EMBEDDING
    # --------------------------------------------------------

    query_embedding = model.encode(
        test["query"]
    )


    # --------------------------------------------------------
    # SEARCH KNOWLEDGE BASE
    # --------------------------------------------------------

    results = []

    for i, embedding in enumerate(
        kb_embeddings
    ):

        score = cosine_similarity(
            query_embedding,
            embedding
        )

        results.append(
            (
                float(score),
                knowledge_base[i]
            )
        )


    results.sort(
        reverse=True,
        key=lambda x: x[0]
    )


    top_results = results[:TOP_K]


    # --------------------------------------------------------
    # DISPLAY RETRIEVED ARTICLES
    # --------------------------------------------------------

    retrieved_ids = [
        article["article_id"]
        for score, article in top_results
    ]


    print(
        "Retrieved KB articles:",
        retrieved_ids
    )


    # --------------------------------------------------------
    # RETRIEVAL RELEVANCE
    # --------------------------------------------------------

    relevant_found = 0

    for expected_id in test["expected_kb"]:

        if expected_id in retrieved_ids:
            relevant_found += 1


    retrieval_score = (
        relevant_found
        / len(test["expected_kb"])
    )


    retrieval_scores.append(
        retrieval_score
    )


    print(
        f"Retrieval relevance: "
        f"{retrieval_score * 100:.1f}%"
    )


    # --------------------------------------------------------
    # SOURCE CORRECTNESS
    # --------------------------------------------------------

    if relevant_found == len(
        test["expected_kb"]
    ):

        source_correct += 1

        print(
            "Source correctness: PASS"
        )

    else:

        print(
            "Source correctness: CHECK"
        )


    # --------------------------------------------------------
    # STEP COVERAGE
    # --------------------------------------------------------

    retrieved_steps = set()

    for score, article in top_results:

        for step in article["steps"]:

            retrieved_steps.add(
                step
            )


    expected_steps = set(
        test["expected_steps"]
    )


    if expected_steps:

        covered_steps = (
            expected_steps
            .intersection(retrieved_steps)
        )

        step_coverage = (
            len(covered_steps)
            / len(expected_steps)
        )

    else:

        step_coverage = 1.0


    step_coverage_scores.append(
        step_coverage
    )


    print(
        f"Step coverage: "
        f"{step_coverage * 100:.1f}%"
    )


    # --------------------------------------------------------
    # LATENCY
    # --------------------------------------------------------

    end_time = time.perf_counter()

    latency = end_time - start_time

    latencies.append(
        latency
    )


    print(
        f"Retrieval latency: "
        f"{latency:.4f} seconds"
    )


# ============================================================
# CALCULATE FINAL METRICS
# ============================================================

average_retrieval = (
    sum(retrieval_scores)
    / len(retrieval_scores)
)


source_accuracy = (
    source_correct
    / len(test_cases)
)


average_step_coverage = (
    sum(step_coverage_scores)
    / len(step_coverage_scores)
)


average_latency = (
    sum(latencies)
    / len(latencies)
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n\n==========================================")
print("       FINAL EVALUATION REPORT")
print("==========================================")

print(
    f"\nTest cases evaluated: "
    f"{len(test_cases)}"
)

print(
    f"\nAverage retrieval relevance: "
    f"{average_retrieval * 100:.2f}%"
)

print(
    f"Source correctness: "
    f"{source_accuracy * 100:.2f}%"
)

print(
    f"Average troubleshooting "
    f"step coverage: "
    f"{average_step_coverage * 100:.2f}%"
)

print(
    f"Average retrieval latency: "
    f"{average_latency:.4f} seconds"
)


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

evaluation_results = {

    "test_cases": len(test_cases),

    "average_retrieval_relevance":
        round(
            average_retrieval,
            4
        ),

    "source_correctness":
        round(
            source_accuracy,
            4
        ),

    "average_step_coverage":
        round(
            average_step_coverage,
            4
        ),

    "average_retrieval_latency_seconds":
        round(
            average_latency,
            4
        )
}


with open(
    "evaluation_results.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        evaluation_results,
        file,
        indent=4
    )


print("\nEvaluation results saved to:")
print("evaluation_results.json")


print("\n==========================================")
print("       EVALUATION COMPLETED")
print("==========================================")