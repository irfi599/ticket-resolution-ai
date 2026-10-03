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


# ============================================================
# COMPLAINT ANALYSIS
# ============================================================

def analyze_complaint(text):

    text_lower = text.lower()

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    categories = []

    if "drop" in text_lower and "call" in text_lower:
        categories.append("Dropped calls")

    if (
        "slow" in text_lower
        and (
            "data" in text_lower
            or "internet" in text_lower
        )
    ):
        categories.append("Slow mobile data")

    if (
        "poor reception" in text_lower
        or "weak signal" in text_lower
    ):
        categories.append("Poor network reception")

    # --------------------------------------------------------
    # INTENT
    # --------------------------------------------------------

    if categories:
        intent = "Network connectivity issue"
    else:
        intent = "General support issue"

    # --------------------------------------------------------
    # PRODUCT
    # --------------------------------------------------------

    has_voice = (
        "call" in text_lower
        or "calls" in text_lower
    )

    has_data = (
        "data" in text_lower
        or "internet" in text_lower
    )

    if has_voice and has_data:
        product = "Mobile voice and data"
    elif has_voice:
        product = "Mobile voice service"
    elif has_data:
        product = "Mobile data service"
    else:
        product = "Telecom service"

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    negative_words = [
        "frustrated",
        "frustrating",
        "angry",
        "annoying",
        "terrible",
        "disappointed",
        "unhappy",
        "problem",
        "issue",
        "not working",
        "can't",
        "cannot"
    ]

    negative_count = 0

    for word in negative_words:

        if word in text_lower:
            negative_count += 1

    if negative_count >= 2:
        sentiment = "Negative"
    elif negative_count == 1:
        sentiment = "Slightly negative"
    else:
        sentiment = "Neutral"

    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    high_severity_words = [
        "emergency",
        "completely down",
        "no service",
        "cannot make calls",
        "unable to call",
        "all day"
    ]

    medium_severity_words = [
        "keep dropping",
        "repeatedly",
        "very slow",
        "frequent",
        "constant",
        "keeps"
    ]

    severity = "Low"

    for word in high_severity_words:

        if word in text_lower:
            severity = "High"
            break

    if severity == "Low":

        for word in medium_severity_words:

            if word in text_lower:
                severity = "Medium"
                break

    return {
        "intent": intent,
        "category": categories if categories else ["Other"],
        "product": product,
        "severity": severity,
        "sentiment": sentiment
    }


# ============================================================
# RUN COMPLAINT ANALYSIS
# ============================================================

analysis = analyze_complaint(query)


# ============================================================
# DISPLAY COMPLAINT ANALYSIS
# ============================================================

print("\n==========================================")
print("       CUSTOMER COMPLAINT ANALYSIS")
print("==========================================")

print("\nCustomer Complaint:")
print(query)

print("\nIntent:")
print(analysis["intent"])

print("\nCategory:")

for category in analysis["category"]:
    print(" -", category)

print("\nProduct:")
print(analysis["product"])

print("\nSeverity:")
print(analysis["severity"])

print("\nSentiment:")
print(analysis["sentiment"])


# ============================================================
# PREPARE HISTORICAL CASES
# ============================================================

historical_documents = [
    case["problem_text"]
    for case in historical_cases
]


# ============================================================
# PREPARE KNOWLEDGE BASE
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

print("\n==========================================")
print("       SEMANTIC RETRIEVAL")
print("==========================================")

print("\nCreating embeddings...")

historical_embeddings = model.encode(
    historical_documents
)

kb_embeddings = model.encode(
    kb_documents
)

query_embedding = model.encode(
    query
)

print("Embeddings created!")


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
Similarity Score: {score:.4f}
Issue Type: {case["issue_type"]}
Device: {case["device"]}
Customer Sentiment: {case["sentiment"]}

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
Knowledge Base Article: {article["article_id"]}
Similarity Score: {score:.4f}
Title: {article["title"]}

Problem:
{article["problem"]}

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

Your job is to produce a grounded resolution using
ONLY the retrieved evidence below.

STRICT RULES:

1. Do NOT invent troubleshooting steps.

2. Every recommended troubleshooting step must
come directly from the retrieved knowledge base.

3. You MUST include ALL DISTINCT troubleshooting
steps from the relevant knowledge-base articles.

4. Do NOT remove or skip a supported troubleshooting
step just to make the answer shorter.

5. If the same step appears in multiple articles,
mention it only once.

6. Use the knowledge base as the PRIMARY source
for troubleshooting recommendations.

7. Use historical cases only as SUPPORTING evidence.

8. Do NOT introduce technical information that is
not present in the retrieved evidence.

9. If troubleshooting does not resolve the issue,
use the escalation condition provided by the
knowledge base.

10. Sources must identify the actual knowledge-base
articles used.

Customer Complaint:
{query}

Complaint Analysis:

Intent:
{analysis["intent"]}

Category:
{", ".join(analysis["category"])}

Product:
{analysis["product"]}

Severity:
{analysis["severity"]}

Sentiment:
{analysis["sentiment"]}


RETRIEVED HISTORICAL CASES:
{historical_context}


RETRIEVED KNOWLEDGE BASE:
{kb_context}


Return the answer in exactly this format:

Issue Summary:
<short summary>

Recommended Steps:
1. <step>
2. <step>
3. <step>
4. <step>

Escalation Guidance:
<explain when escalation is required>

Sources:
- <knowledge-base article ID and title>
- <knowledge-base article ID and title>


FINAL CHECK:

Make sure every distinct troubleshooting step
from the retrieved knowledge base is included.

Do not invent unsupported information.
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


    print(
        "\nOllama status code:",
        response.status_code
    )


    if response.status_code != 200:

        print("\nOllama returned an error:")
        print(response.text)

        raise SystemExit


    result = response.json()

    generated_answer = result.get(
        "response",
        ""
    ).strip()


    if not generated_answer:

        print("\nOllama returned an empty response.")
        print("\nRaw response:")
        print(response.text)

        raise SystemExit


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("==========================================")
    print("       INTELLIGENT SUPPORT ASSISTANT")
    print("==========================================")

    print("\nCustomer Complaint:")
    print(query)

    print("\nComplaint Analysis:")

    print(
        "Intent:",
        analysis["intent"]
    )

    print(
        "Category:",
        ", ".join(analysis["category"])
    )

    print(
        "Product:",
        analysis["product"]
    )

    print(
        "Severity:",
        analysis["severity"]
    )

    print(
        "Sentiment:",
        analysis["sentiment"]
    )

    print("\nGenerated Resolution:")
    print("------------------------------------------")

    print(generated_answer)

    print("\n==========================================")
    print("       RAG PIPELINE COMPLETED")
    print("==========================================")


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