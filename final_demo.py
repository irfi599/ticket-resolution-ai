import os
import json
import requests
import numpy as np

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

HISTORICAL_FILE = "historical_cases_10000.json"
KB_FILE = "knowledge_base.json"
EMBEDDING_FILE = "historical_embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"

TOP_HISTORICAL_CASES = 5
TOP_KB_ARTICLES = 2


# ============================================================
# LOAD PROJECT DATA
# ============================================================

print()
print("Loading project data...")

with open(HISTORICAL_FILE, "r", encoding="utf-8") as file:
    historical_cases = json.load(file)

with open(KB_FILE, "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)

print(f"Historical cases loaded: {len(historical_cases)}")
print(f"Knowledge base articles loaded: {len(knowledge_base)}")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print()
print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

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

def analyze_complaint(query):

    text = query.lower()

    categories = []

    if "drop" in text and "call" in text:
        categories.append("Dropped calls")

    if "slow" in text and (
        "data" in text or
        "internet" in text
    ):
        categories.append("Slow mobile data")

    if (
        "poor reception" in text or
        "weak signal" in text
    ):
        categories.append("Poor network reception")

    if "network" in text or "signal" in text:
        if "Network issue" not in categories:
            categories.append("Network issue")

    if not categories:
        categories.append("General support issue")

    if (
        "call" in text or
        "internet" in text or
        "data" in text
    ):
        product = "Mobile voice and data"
    else:
        product = "Telecom service"

    negative_words = [
        "frustrating",
        "frustrated",
        "angry",
        "disappointed",
        "annoying",
        "terrible",
        "unhappy",
        "hate",
        "worst"
    ]

    negative_count = sum(
        1
        for word in negative_words
        if word in text
    )

    if negative_count >= 2:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    severity_words = [
        "emergency",
        "completely",
        "cannot",
        "unable",
        "critical"
    ]

    if any(
        word in text
        for word in severity_words
    ):
        severity = "High"

    elif any(
        word in text
        for word in [
            "slow",
            "dropping",
            "disconnect",
            "problem"
        ]
    ):
        severity = "Medium"

    else:
        severity = "Low"

    if (
        "call" in text or
        "data" in text or
        "internet" in text
    ):
        intent = "Network connectivity issue"
    else:
        intent = "General support issue"

    return {
        "intent": intent,
        "categories": categories,
        "product": product,
        "severity": severity,
        "sentiment": sentiment
    }


# ============================================================
# DISPLAY COMPLAINT ANALYSIS
# ============================================================

analysis = analyze_complaint(query)

print()
print("==========================================")
print("CUSTOMER COMPLAINT ANALYSIS")
print("==========================================")

print()
print("Customer Complaint:")
print(query)

print()
print("Intent:")
print(analysis["intent"])

print()
print("Category:")

for category in analysis["categories"]:
    print(f" - {category}")

print()
print("Product:")
print(analysis["product"])

print()
print("Severity:")
print(analysis["severity"])

print()
print("Sentiment:")
print(analysis["sentiment"])


# ============================================================
# PREPARE HISTORICAL CASE INDEX
# ============================================================

print()
print("==========================================")
print("PREPARING HISTORICAL CASE INDEX")
print("==========================================")

case_texts = []

for case in historical_cases:

    text = (
        str(case.get("problem_text", "")) + " "
        + str(case.get("issue_type", "")) + " "
        + str(case.get("device", "")) + " "
        + str(case.get("sentiment", "")) + " "
        + " ".join(
            case.get("troubleshooting_steps", [])
        )
    )

    case_texts.append(text)


# ============================================================
# LOAD OR CREATE HISTORICAL EMBEDDINGS
# ============================================================

if (
    os.path.exists(EMBEDDING_FILE)
    and len(np.load(EMBEDDING_FILE, mmap_mode="r"))
    == len(historical_cases)
):

    print()
    print("Saved historical embeddings found.")
    print("Loading historical case embeddings...")

    historical_embeddings = np.load(
        EMBEDDING_FILE
    )

    print(
        f"Historical case embeddings loaded: "
        f"{len(historical_embeddings)}"
    )

else:

    print()
    print(
        f"Creating embeddings for "
        f"{len(historical_cases)} historical cases..."
    )

    historical_embeddings = model.encode(
        case_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    np.save(
        EMBEDDING_FILE,
        historical_embeddings
    )

    print()
    print(
        "Historical case embeddings "
        "created and saved!"
    )


# ============================================================
# KNOWLEDGE BASE EMBEDDINGS
# ============================================================

print()
print("Creating knowledge-base embeddings...")

kb_texts = []

for article in knowledge_base:

    text = (
        article["title"]
        + " "
        + article["problem"]
        + " "
        + " ".join(article["symptoms"])
    )

    kb_texts.append(text)


kb_embeddings = model.encode(
    kb_texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Knowledge-base embeddings created!")


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

print()
print("==========================================")
print("SEMANTIC RETRIEVAL")
print("==========================================")

print()
print(
    f"Searching {len(historical_cases)} "
    "historical cases..."
)


query_embedding = model.encode(
    query,
    normalize_embeddings=True
)


# ============================================================
# HISTORICAL CASE RETRIEVAL
# ============================================================

historical_similarities = np.dot(
    historical_embeddings,
    query_embedding
)

top_historical_indices = np.argsort(
    historical_similarities
)[::-1][
    :TOP_HISTORICAL_CASES
]

historical_results = []

for index in top_historical_indices:

    historical_results.append(
        {
            "case": historical_cases[index],
            "score": float(
                historical_similarities[index]
            )
        }
    )


print()
print("Top Historical Cases:")


for rank, result in enumerate(
    historical_results,
    start=1
):

    case = result["case"]

    print()
    print(f"Rank {rank}")

    print(
        f"Similarity: "
        f"{result['score']:.4f}"
    )

    # Case ID
    case_id = case.get(
        "case_id",
        case.get(
            "conversation_id",
            case.get("id", "N/A")
        )
    )

    print(
        f"Case ID: "
        f"{case_id}"
    )

    print(
        f"Issue: "
        f"{case.get('issue_type', 'N/A')}"
    )

    print(
        f"Device: "
        f"{case.get('device', 'N/A')}"
    )

    print(
        f"Sentiment: "
        f"{case.get('sentiment', 'N/A')}"
    )

    print(
        f"Outcome: "
        f"{case.get('outcome', 'N/A')}"
    )

    print("Troubleshooting:")

    steps = case.get(
        "troubleshooting_steps",
        []
    )

    if steps:

        for step in steps:
            print(f" - {step}")

    else:

        print(" - None recorded")


# ============================================================
# KNOWLEDGE BASE RETRIEVAL
# ============================================================

kb_similarities = np.dot(
    kb_embeddings,
    query_embedding
)

top_kb_indices = np.argsort(
    kb_similarities
)[::-1][
    :TOP_KB_ARTICLES
]

kb_results = []

for index in top_kb_indices:

    kb_results.append(
        {
            "article": knowledge_base[index],
            "score": float(
                kb_similarities[index]
            )
        }
    )


print()
print("Top Knowledge Base Articles:")


for rank, result in enumerate(
    kb_results,
    start=1
):

    article = result["article"]

    print()
    print(f"Rank {rank}")

    print(
        f"Similarity: "
        f"{result['score']:.4f}"
    )

    print(
        f"Article: "
        f"{article['article_id']}"
    )

    print(
        f"Title: "
        f"{article['title']}"
    )


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

print()
print("==========================================")
print("BUILDING RAG CONTEXT")
print("==========================================")

context = ""

context += "\nKNOWLEDGE BASE:\n"

for result in kb_results:

    article = result["article"]

    context += f"""
Article ID: {article["article_id"]}
Title: {article["title"]}
Problem: {article["problem"]}
Symptoms: {", ".join(article["symptoms"])}
Steps: {", ".join(article["steps"])}
Escalation: {article["escalation_condition"]}
"""


context += "\nHISTORICAL CASES:\n"

for result in historical_results:

    case = result["case"]

    context += f"""
Issue: {case.get("issue_type", "N/A")}
Device: {case.get("device", "N/A")}
Sentiment: {case.get("sentiment", "N/A")}
Troubleshooting: {", ".join(case.get("troubleshooting_steps", []))}
Outcome: {case.get("outcome", "N/A")}
"""


# ============================================================
# RAG PROMPT
# ============================================================

prompt = f"""
You are an intelligent telecom support assistant.

Customer complaint:
{query}

Complaint analysis:

Intent:
{analysis["intent"]}

Category:
{", ".join(analysis["categories"])}

Product:
{analysis["product"]}

Severity:
{analysis["severity"]}

Sentiment:
{analysis["sentiment"]}


Retrieved information:

{context}


Instructions:

1. Provide a short issue summary.

2. Give troubleshooting steps supported by
   the knowledge base.

3. The knowledge base is the primary source
   for troubleshooting instructions.

4. Historical cases are supporting evidence
   and should not override the knowledge base.

5. Do not invent troubleshooting steps.

6. Include all distinct relevant troubleshooting
   steps supported by the retrieved knowledge base.

7. If the issue continues, provide the supported
   escalation guidance.

8. Do not claim that the issue will definitely
   be resolved after escalation.

9. Do not invent customer information.

10. Mention the actual knowledge-base article IDs
    used in the response.

11. Keep the response clear, concise and practical.

Return the answer using exactly this structure:

Issue Summary:

Recommended Steps:

Escalation Guidance:

Sources:
"""


# ============================================================
# GENERATE RAG RESPONSE USING OLLAMA
# ============================================================

print()
print("==========================================")
print("GENERATING RAG RESPONSE")
print("==========================================")

print()
print("Connecting to Ollama...")


try:

    response = requests.post(
        "http://localhost:11434/api/generate",

        json={
            "model": OLLAMA_MODEL,

            "prompt": prompt,

            "stream": False,

            "options": {
                "temperature": 0.2,
                "num_predict": 400
            }
        },

        timeout=120
    )

    print(
        f"Ollama status: "
        f"{response.status_code}"
    )

    if response.status_code == 200:

        generated_response = (
            response.json()["response"]
        )

    else:

        generated_response = (
            "Unable to generate a response "
            "from Ollama."
        )


except Exception as error:

    generated_response = (
        f"Ollama connection error: {error}"
    )


# ============================================================
# DISPLAY GENERATED RESPONSE
# ============================================================

print()
print("==========================================")
print("GENERATED RESOLUTION")
print("==========================================")

print()
print(generated_response)


# ============================================================
# FINAL PIPELINE SUMMARY
# ============================================================

print()
print("==========================================")
print("RAG PIPELINE COMPLETED")
print("==========================================")

print(
    f"Historical cases searched: "
    f"{len(historical_cases)}"
)

print(
    f"Knowledge-base articles searched: "
    f"{len(knowledge_base)}"
)

print(
    f"Historical cases retrieved: "
    f"{len(historical_results)}"
)

print(
    f"Knowledge-base articles retrieved: "
    f"{len(kb_results)}"
)

print(
    f"Historical embeddings file: "
    f"{EMBEDDING_FILE}"
)

print("==========================================")