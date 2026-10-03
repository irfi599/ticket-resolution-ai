import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

TOP_HISTORICAL = 2
TOP_KB = 2


# ============================================================
# LOAD DATA
# ============================================================

print("Loading project data...")

with open("historical_cases.json", "r", encoding="utf-8") as file:
    historical_cases = json.load(file)

with open("knowledge_base.json", "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)

print("Historical cases loaded:", len(historical_cases))
print("Knowledge base articles loaded:", len(knowledge_base))


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# ============================================================
# CUSTOMER COMPLAINT
# ============================================================

query = (
    "My calls keep dropping and my mobile internet "
    "is very slow."
)

print("\nCustomer complaint:")
print(query)


# ============================================================
# PREPARE HISTORICAL CASE DOCUMENTS
# ============================================================

historical_documents = [
    case["problem_text"]
    for case in historical_cases
]


# ============================================================
# PREPARE KNOWLEDGE BASE DOCUMENTS
# ============================================================

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


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("\nCreating embeddings...")

historical_embeddings = model.encode(
    historical_documents
)

print("Historical embeddings created.")

kb_embeddings = model.encode(
    kb_documents
)

print("Knowledge base embeddings created.")

query_embedding = model.encode(
    query
)

print("Customer query embedding created.")


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
# SEARCH HISTORICAL CASES
# ============================================================

print("\nSearching historical cases...")

historical_results = []

for i, embedding in enumerate(
    historical_embeddings
):

    score = cosine_similarity(
        query_embedding,
        embedding
    )

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

top_historical = historical_results[
    :TOP_HISTORICAL
]


# ============================================================
# SEARCH KNOWLEDGE BASE
# ============================================================

print("Searching knowledge base...")

kb_results = []

for i, embedding in enumerate(
    kb_embeddings
):

    score = cosine_similarity(
        query_embedding,
        embedding
    )

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

top_kb = kb_results[
    :TOP_KB
]


# ============================================================
# DISPLAY RETRIEVAL RESULTS
# ============================================================

print("\n==========================================")
print("       RETRIEVAL RESULTS")
print("==========================================")


# ------------------------------------------------------------
# HISTORICAL CASES
# ------------------------------------------------------------

print("\n========== HISTORICAL CASES ==========")

for rank, (score, case) in enumerate(
    top_historical,
    start=1
):

    print("\n------------------------------------------")

    print("Rank:", rank)

    print(
        f"Similarity Score: {score:.4f}"
    )

    print(
        "Case ID:",
        case["case_id"]
    )

    print(
        "Issue:",
        case["issue_type"]
    )

    print(
        "Device:",
        case["device"]
    )

    print(
        "Sentiment:",
        case["sentiment"]
    )

    print("\nResolution Steps:")

    if case["troubleshooting_steps"]:

        for step in case["troubleshooting_steps"]:
            print(" -", step)

    else:

        print(" - No troubleshooting steps identified")

    print(
        "\nOutcome:",
        case["outcome"]
    )


# ------------------------------------------------------------
# KNOWLEDGE BASE
# ------------------------------------------------------------

print("\n========== KNOWLEDGE BASE ==========")

for rank, (score, article) in enumerate(
    top_kb,
    start=1
):

    print("\n------------------------------------------")

    print("Rank:", rank)

    print(
        f"Similarity Score: {score:.4f}"
    )

    print(
        "Article ID:",
        article["article_id"]
    )

    print(
        "Title:",
        article["title"]
    )

    print("\nRecommended Steps:")

    for step in article["steps"]:

        print(" -", step)

    print(
        "\nEscalation:",
        article["escalation_condition"]
    )


# ============================================================
# BUILD HISTORICAL CONTEXT
# ============================================================

historical_context = ""

for score, case in top_historical:

    steps = "\n".join(
        "- " + step
        for step in case["troubleshooting_steps"]
    )

    historical_context += f"""
Historical Case ID: {case["case_id"]}

Similarity Score:
{score:.4f}

Issue Type:
{case["issue_type"]}

Device:
{case["device"]}

Customer Sentiment:
{case["sentiment"]}

Troubleshooting Steps:
{steps}

Outcome:
{case["outcome"]}

Source:
Historical Case {case["case_id"]}

"""


# ============================================================
# BUILD KNOWLEDGE BASE CONTEXT
# ============================================================

kb_context = ""

for score, article in top_kb:

    steps = "\n".join(
        "- " + step
        for step in article["steps"]
    )

    kb_context += f"""
Knowledge Base Article:
{article["article_id"]}

Similarity Score:
{score:.4f}

Title:
{article["title"]}

Problem:
{article["problem"]}

Symptoms:
{", ".join(article["symptoms"])}

Recommended Steps:
{steps}

Escalation Condition:
{article["escalation_condition"]}

Source:
{article["article_id"]}

"""


# ============================================================
# RAG PROMPT
# ============================================================

prompt = f"""
You are an intelligent telecom customer support
resolution assistant.

RAG means Retrieval-Augmented Generation.

Your task is to provide a grounded resolution for
the customer's complaint using ONLY the retrieved
evidence below.


STRICT GROUNDING RULES
==============================

1. Do NOT invent troubleshooting steps.

2. Every recommended troubleshooting step must
   come directly from the retrieved knowledge base.

3. You MUST include ALL DISTINCT troubleshooting
   steps from the relevant knowledge-base articles.

4. Do NOT skip a supported step just to make the
   response shorter.

5. If the same step appears in multiple articles,
   mention it only once.

6. Use the knowledge base as the PRIMARY source
   for troubleshooting recommendations.

7. Use historical cases only as SUPPORTING evidence.

8. Do NOT introduce technical information that is
   not present in the retrieved evidence.

9. If the recommended troubleshooting does not
   resolve the issue, use the escalation condition
   provided by the knowledge base.

10. Sources must identify the actual knowledge-base
    articles used.


CUSTOMER COMPLAINT
==============================

{query}


RETRIEVED HISTORICAL CASES
==============================

{historical_context}


RETRIEVED KNOWLEDGE BASE
==============================

{kb_context}


RESPONSE FORMAT
==============================

Issue Summary:
<short summary of the customer's problem>

Recommended Steps:
1. <all supported step 1>
2. <all supported step 2>
3. <all supported step 3>
4. <all supported step 4>

Escalation Guidance:
<explain when escalation is required>

Sources:
- <knowledge-base article ID and title>
- <knowledge-base article ID and title>


FINAL CHECK BEFORE ANSWERING:

Make sure that every distinct troubleshooting step
from the retrieved knowledge base is included in
Recommended Steps.

Do not omit a supported step.
"""


# ============================================================
# GENERATE RESPONSE USING OLLAMA
# ============================================================

print("\n==========================================")
print("       RAG GENERATION")
print("==========================================")

print("\nSending retrieved context to Ollama...")


try:

    response = requests.post(

        OLLAMA_URL,

        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,

            "options": {
                "temperature": 0.2,
                "num_predict": 400
            }
        },

        timeout=300
    )


    # --------------------------------------------------------
    # CHECK RESPONSE
    # --------------------------------------------------------

    print(
        "\nOllama status code:",
        response.status_code
    )


    if response.status_code != 200:

        print(
            "\nOllama returned an error:"
        )

        print(response.text)

        raise SystemExit


    # --------------------------------------------------------
    # PARSE JSON
    # --------------------------------------------------------

    result = response.json()


    generated_answer = result.get(
        "response",
        ""
    ).strip()


    # --------------------------------------------------------
    # CHECK FOR EMPTY RESPONSE
    # --------------------------------------------------------

    if not generated_answer:

        print(
            "\nOllama returned an empty response."
        )

        print(
            "\nRaw response:"
        )

        print(response.text)

        raise SystemExit


    # ========================================================
    # DISPLAY FINAL ANSWER
    # ========================================================

    print("\n")
    print("==========================================")
    print("       INTELLIGENT SUPPORT ASSISTANT")
    print("==========================================")

    print("\nCustomer Complaint:")
    print(query)

    print("\nGenerated Resolution:")
    print("------------------------------------------")

    print(generated_answer)

    print("\n==========================================")
    print("       RAG PIPELINE COMPLETED")
    print("==========================================")


# ============================================================
# ERROR HANDLING
# ============================================================

except requests.exceptions.Timeout:

    print(
        "\nERROR: Ollama took too long to respond."
    )


except requests.exceptions.ConnectionError:

    print(
        "\nERROR: Could not connect to Ollama."
    )

    print(
        "\nMake sure Ollama is running."
    )

    print(
        "Test with:"
    )

    print(
        "ollama run llama3.2:3b"
    )


except Exception as e:

    print(
        "\nUnexpected error:"
    )

    print(
        type(e).__name__,
        ":",
        e
    )