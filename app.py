import json
import os

import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent Support Ticket Resolution Assistant",
    page_icon="🎫",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

HISTORICAL_FILE = "historical_cases_10000.json"
KB_FILE = "knowledge_base.json"
EMBEDDING_FILE = "historical_embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

TOP_HISTORICAL_CASES = 5
TOP_KB_ARTICLES = 2


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    .title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD PROJECT DATA
# ============================================================

@st.cache_data
def load_project_data():

    if not os.path.exists(HISTORICAL_FILE):

        st.error(
            f"Missing file: {HISTORICAL_FILE}"
        )

        st.stop()

    if not os.path.exists(KB_FILE):

        st.error(
            f"Missing file: {KB_FILE}"
        )

        st.stop()

    with open(
        HISTORICAL_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        historical_cases = json.load(file)

    with open(
        KB_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        knowledge_base = json.load(file)

    return historical_cases, knowledge_base


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        MODEL_NAME
    )


# ============================================================
# CREATE TEXT FOR HISTORICAL CASE
# ============================================================

def get_case_text(case):

    parts = []

    issue = case.get(
        "issue",
        ""
    )

    if issue:
        parts.append(
            str(issue)
        )

    device = case.get(
        "device",
        ""
    )

    if device:
        parts.append(
            str(device)
        )

    sentiment = case.get(
        "sentiment",
        ""
    )

    if sentiment:
        parts.append(
            str(sentiment)
        )

    outcome = case.get(
        "outcome",
        ""
    )

    if outcome:
        parts.append(
            str(outcome)
        )

    troubleshooting = case.get(
        "troubleshooting",
        []
    )

    if isinstance(
        troubleshooting,
        list
    ):

        parts.extend(
            str(step)
            for step in troubleshooting
        )

    elif troubleshooting:

        parts.append(
            str(troubleshooting)
        )

    # Some versions of the dataset may contain
    # additional conversation/text fields.

    for field in [
        "conversation",
        "conversation_text",
        "text",
        "customer_message",
        "description",
        "problem"
    ]:

        value = case.get(
            field
        )

        if value:

            if isinstance(
                value,
                list
            ):

                parts.extend(
                    str(item)
                    for item in value
                )

            else:

                parts.append(
                    str(value)
                )

    if not parts:

        return (
            "Telecom customer support case"
        )

    return " ".join(parts)


# ============================================================
# LOAD OR CREATE HISTORICAL EMBEDDINGS
# ============================================================

@st.cache_resource
def load_historical_embeddings():

    # --------------------------------------------------------
    # First try the locally saved embeddings.
    # --------------------------------------------------------

    if os.path.exists(
        EMBEDDING_FILE
    ):

        try:

            embeddings = np.load(
                EMBEDDING_FILE
            )

            if len(embeddings) == len(
                historical_cases
            ):

                return embeddings

        except Exception:

            pass

    # --------------------------------------------------------
    # Create embeddings if the file does not exist.
    # --------------------------------------------------------

    texts = [
        get_case_text(case)
        for case in historical_cases
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    np.save(
        EMBEDDING_FILE,
        embeddings
    )

    return embeddings


# ============================================================
# CREATE KNOWLEDGE BASE EMBEDDINGS
# ============================================================

@st.cache_resource
def load_kb_embeddings():

    kb_texts = []

    for article in knowledge_base:

        title = article.get(
            "title",
            ""
        )

        problem = article.get(
            "problem",
            ""
        )

        symptoms = article.get(
            "symptoms",
            []
        )

        steps = article.get(
            "steps",
            []
        )

        escalation = article.get(
            "escalation_condition",
            ""
        )

        text = " ".join(
            [
                str(title),
                str(problem),
                " ".join(
                    str(x)
                    for x in symptoms
                ),
                " ".join(
                    str(x)
                    for x in steps
                ),
                str(escalation)
            ]
        )

        kb_texts.append(
            text
        )

    return model.encode(
        kb_texts,
        normalize_embeddings=True
    )


# ============================================================
# ANALYZE CUSTOMER COMPLAINT
# ============================================================

def analyze_complaint(query):

    text = query.lower()

    categories = []

    # --------------------------------------------------------
    # Dropped calls
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "drop",
            "dropped",
            "disconnect",
            "disconnecting",
            "call drops",
            "calls drop"
        ]
    ):

        categories.append(
            "Dropped calls"
        )

    # --------------------------------------------------------
    # Slow mobile data
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "slow data",
            "slow internet",
            "slow mobile internet",
            "mobile data",
            "data is slow",
            "internet is slow"
        ]
    ):

        categories.append(
            "Slow mobile data"
        )

    # --------------------------------------------------------
    # Poor reception
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "poor signal",
            "weak signal",
            "poor reception",
            "weak reception",
            "network reception"
        ]
    ):

        categories.append(
            "Poor network reception"
        )

    if not categories:

        categories.append(
            "General telecom support issue"
        )

    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    if any(
        category in categories
        for category in [
            "Dropped calls",
            "Slow mobile data",
            "Poor network reception"
        ]
    ):

        intent = (
            "Network connectivity issue"
        )

    else:

        intent = (
            "General support issue"
        )

    # --------------------------------------------------------
    # Product
    # --------------------------------------------------------

    has_voice = (
        "Dropped calls"
        in categories
    )

    has_data = (
        "Slow mobile data"
        in categories
    )

    if has_voice and has_data:

        product = (
            "Mobile voice and data"
        )

    elif has_voice:

        product = (
            "Mobile voice"
        )

    elif has_data:

        product = (
            "Mobile data"
        )

    else:

        product = (
            "Telecom service"
        )

    # --------------------------------------------------------
    # Sentiment
    # --------------------------------------------------------

    negative_words = [
        "bad",
        "terrible",
        "frustrated",
        "angry",
        "annoyed",
        "worst",
        "problem",
        "issue",
        "not working",
        "costing",
        "unacceptable"
    ]

    negative_count = sum(
        1
        for word in negative_words
        if word in text
    )

    if negative_count >= 2:

        sentiment = "Negative"

    elif negative_count == 1:

        sentiment = "Slightly negative"

    else:

        sentiment = "Neutral"

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    high_severity_words = [
        "emergency",
        "completely down",
        "cannot work",
        "critical",
        "no service",
        "multiple days"
    ]

    medium_severity_words = [
        "frequent",
        "repeated",
        "every day",
        "every evening",
        "keeps dropping",
        "constantly"
    ]

    if any(
        word in text
        for word in high_severity_words
    ):

        severity = "High"

    elif any(
        word in text
        for word in medium_severity_words
    ):

        severity = "Medium"

    else:

        severity = "Low"

    return {
        "intent": intent,
        "categories": categories,
        "product": product,
        "severity": severity,
        "sentiment": sentiment
    }


# ============================================================
# RETRIEVE HISTORICAL CASES
# ============================================================

def retrieve_historical_cases(
    query,
    historical_embeddings
):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    similarities = np.dot(
        historical_embeddings,
        query_embedding
    )

    top_indices = np.argsort(
        similarities
    )[::-1][
        :TOP_HISTORICAL_CASES
    ]

    results = []

    for index in top_indices:

        case = dict(
            historical_cases[
                int(index)
            ]
        )

        case["_similarity"] = float(
            similarities[index]
        )

        results.append(
            case
        )

    return results


# ============================================================
# RETRIEVE KNOWLEDGE BASE ARTICLES
# ============================================================

def retrieve_kb_articles(
    query,
    kb_embeddings
):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    similarities = np.dot(
        kb_embeddings,
        query_embedding
    )

    top_indices = np.argsort(
        similarities
    )[::-1][
        :TOP_KB_ARTICLES
    ]

    results = []

    for index in top_indices:

        article = dict(
            knowledge_base[
                int(index)
            ]
        )

        article["_similarity"] = float(
            similarities[index]
        )

        results.append(
            article
        )

    return results


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_rag_context(
    complaint,
    analysis,
    historical_results,
    kb_results
):

    context = []

    # --------------------------------------------------------
    # Knowledge base
    # --------------------------------------------------------

    context.append(
        "AUTHORITATIVE KNOWLEDGE BASE:"
    )

    for article in kb_results:

        steps = article.get(
            "steps",
            []
        )

        symptoms = article.get(
            "symptoms",
            []
        )

        context.append(
            f"""
Article ID:
{article.get("article_id", "N/A")}

Title:
{article.get("title", "N/A")}

Problem:
{article.get("problem", "N/A")}

Symptoms:
{", ".join(symptoms)}

Troubleshooting Steps:
- {"\n- ".join(steps)}

Escalation Guidance:
{article.get("escalation_condition", "N/A")}
"""
        )

    # --------------------------------------------------------
    # Historical cases
    # --------------------------------------------------------

    context.append(
        "\nHISTORICAL CASES — SUPPORTING EVIDENCE ONLY:"
    )

    for case in historical_results:

        case_id = case.get(
            "case_id",
            case.get(
                "conversation_id",
                case.get(
                    "id",
                    "N/A"
                )
            )
        )

        troubleshooting = case.get(
            "troubleshooting",
            []
        )

        context.append(
            f"""
Case ID:
{case_id}

Issue:
{case.get("issue", "N/A")}

Device:
{case.get("device", "N/A")}

Sentiment:
{case.get("sentiment", "N/A")}

Outcome:
{case.get("outcome", "N/A")}

Troubleshooting:
- {"\n- ".join(troubleshooting)}
"""
        )

    # --------------------------------------------------------
    # Complaint analysis
    # --------------------------------------------------------

    context.append(
        f"""
CUSTOMER COMPLAINT:
{complaint}

INTENT:
{analysis["intent"]}

CATEGORIES:
{", ".join(analysis["categories"])}

PRODUCT:
{analysis["product"]}

SEVERITY:
{analysis["severity"]}

SENTIMENT:
{analysis["sentiment"]}
"""
    )

    return "\n".join(
        context
    )


# ============================================================
# GENERATE GROUNDED RAG RESPONSE
# ============================================================

def generate_resolution(
    complaint,
    analysis,
    historical_results,
    kb_results
):

    context = build_rag_context(
        complaint,
        analysis,
        historical_results,
        kb_results
    )

    prompt = f"""
You are an enterprise telecom support resolution assistant.

Generate a concise and practical resolution for the
customer complaint.

GROUNDING RULES:

1. The Knowledge Base is the authoritative source.
2. Use Knowledge Base articles for troubleshooting steps.
3. Historical cases are supporting evidence only.
4. Do not invent troubleshooting steps.
5. Do not invent customer information.
6. Do not invent websites, portals, phone numbers,
   teams, departments or contact methods.
7. Do not claim guaranteed resolution.
8. Use the escalation guidance from the Knowledge Base.
9. Include all distinct relevant troubleshooting steps
   from the retrieved Knowledge Base articles.
10. Cite the actual Knowledge Base article IDs.
11. Keep the answer concise.

Return exactly:

Issue Summary:
<short summary>

Recommended Steps:
1. <step>
2. <step>

Escalation Guidance:
<grounded escalation guidance>

Sources:
- <KB article ID>: <title>
- <KB article ID>: <title>

Do not add unsupported information.

CONTEXT:

{context}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 400
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:

            return (
                f"Ollama returned HTTP "
                f"{response.status_code}."
            )

        data = response.json()

        return data.get(
            "response",
            "No response generated."
        )

    except requests.exceptions.ConnectionError:

        return (
            "Unable to connect to Ollama. "
            "Please make sure Ollama is running."
        )

    except Exception as error:

        return (
            f"Error generating resolution: {error}"
        )


# ============================================================
# LOAD EVERYTHING
# ============================================================

historical_cases, knowledge_base = (
    load_project_data()
)

model = load_embedding_model()


# ------------------------------------------------------------
# Create/load historical embeddings.
# ------------------------------------------------------------

if os.path.exists(
    EMBEDDING_FILE
):

    historical_embeddings = (
        load_historical_embeddings()
    )

else:

    with st.spinner(
        "Creating embeddings for 9,926 historical cases. "
        "This happens only on the first run..."
    ):

        historical_embeddings = (
            load_historical_embeddings()
        )


# ------------------------------------------------------------
# Create/load KB embeddings.
# ------------------------------------------------------------

kb_embeddings = (
    load_kb_embeddings()
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">'
    '🎫 Intelligent Support Ticket Resolution Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Semantic retrieval + grounded RAG for telecom support ticket resolution'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# METRICS
# ============================================================

col1, col2, col3, col4 = (
    st.columns(4)
)

with col1:

    st.metric(
        "Historical Cases",
        f"{len(historical_cases):,}"
    )

with col2:

    st.metric(
        "KB Articles",
        len(knowledge_base)
    )

with col3:

    st.metric(
        "Historical Retrieved",
        TOP_HISTORICAL_CASES
    )

with col4:

    st.metric(
        "KB Retrieved",
        TOP_KB_ARTICLES
    )


st.divider()


# ============================================================
# CUSTOMER COMPLAINT INPUT
# ============================================================

st.subheader(
    "Customer Complaint"
)

default_query = (
    "My calls keep dropping and my mobile internet "
    "is very slow."
)

query = st.text_area(
    "Enter the customer complaint:",
    value=default_query,
    height=130
)

analyze_button = st.button(
    "🔍 Analyze & Resolve",
    type="primary",
    use_container_width=True
)


# ============================================================
# MAIN PIPELINE
# ============================================================

if analyze_button:

    if not query.strip():

        st.warning(
            "Please enter a customer complaint."
        )

    else:

        with st.spinner(
            "Analyzing complaint, retrieving relevant cases "
            "and generating resolution..."
        ):

            # ------------------------------------------------
            # 1. Complaint analysis
            # ------------------------------------------------

            analysis = analyze_complaint(
                query
            )

            # ------------------------------------------------
            # 2. Historical retrieval
            # ------------------------------------------------

            historical_results = (
                retrieve_historical_cases(
                    query,
                    historical_embeddings
                )
            )

            # ------------------------------------------------
            # 3. Knowledge base retrieval
            # ------------------------------------------------

            kb_results = (
                retrieve_kb_articles(
                    query,
                    kb_embeddings
                )
            )

            # ------------------------------------------------
            # 4. RAG generation
            # ------------------------------------------------

            resolution = (
                generate_resolution(
                    query,
                    analysis,
                    historical_results,
                    kb_results
                )
            )


        # ====================================================
        # CUSTOMER ANALYSIS
        # ====================================================

        st.divider()

        st.header(
            "🧠 Customer Complaint Analysis"
        )

        st.write(
            f"**Customer Complaint:** {query}"
        )

        col1, col2 = (
            st.columns(2)
        )

        with col1:

            st.write(
                f"**Intent:** "
                f"{analysis['intent']}"
            )

            st.write(
                f"**Product:** "
                f"{analysis['product']}"
            )

            st.write(
                f"**Severity:** "
                f"{analysis['severity']}"
            )

        with col2:

            st.write(
                f"**Sentiment:** "
                f"{analysis['sentiment']}"
            )

            st.write(
                "**Categories:**"
            )

            for category in analysis[
                "categories"
            ]:

                st.write(
                    f"- {category}"
                )


        # ====================================================
        # RETRIEVAL
        # ====================================================

        st.divider()

        st.header(
            "🔎 Semantic Retrieval"
        )

        historical_col, kb_col = (
            st.columns(2)
        )

        # ----------------------------------------------------
        # Historical cases
        # ----------------------------------------------------

        with historical_col:

            st.subheader(
                "Similar Historical Cases"
            )

            for rank, case in enumerate(
                historical_results,
                start=1
            ):

                case_id = case.get(
                    "case_id",
                    case.get(
                        "conversation_id",
                        case.get(
                            "id",
                            "N/A"
                        )
                    )
                )

                with st.expander(
                    f"Case {rank} — "
                    f"{case.get('issue', 'Support case')}"
                ):

                    st.write(
                        f"**Similarity:** "
                        f"{case.get('_similarity', 0):.4f}"
                    )

                    st.write(
                        f"**Case ID:** "
                        f"{case_id}"
                    )

                    st.write(
                        f"**Issue:** "
                        f"{case.get('issue', 'N/A')}"
                    )

                    st.write(
                        f"**Device:** "
                        f"{case.get('device', 'N/A')}"
                    )

                    st.write(
                        f"**Sentiment:** "
                        f"{case.get('sentiment', 'N/A')}"
                    )

                    st.write(
                        f"**Outcome:** "
                        f"{case.get('outcome', 'N/A')}"
                    )

                    troubleshooting = (
                        case.get(
                            "troubleshooting",
                            []
                        )
                    )

                    if troubleshooting:

                        st.write(
                            "**Troubleshooting:**"
                        )

                        for step in troubleshooting:

                            st.write(
                                f"- {step}"
                            )

                    else:

                        st.write(
                            "**Troubleshooting:** "
                            "None recorded"
                        )


        # ----------------------------------------------------
        # Knowledge base
        # ----------------------------------------------------

        with kb_col:

            st.subheader(
                "Relevant Knowledge Base"
            )

            for rank, article in enumerate(
                kb_results,
                start=1
            ):

                with st.expander(
                    f"{article.get('article_id', 'KB')} — "
                    f"{article.get('title', 'Article')}"
                ):

                    st.write(
                        f"**Similarity:** "
                        f"{article.get('_similarity', 0):.4f}"
                    )

                    st.write(
                        f"**Article ID:** "
                        f"{article.get('article_id', 'N/A')}"
                    )

                    st.write(
                        f"**Title:** "
                        f"{article.get('title', 'N/A')}"
                    )

                    st.write(
                        f"**Problem:** "
                        f"{article.get('problem', 'N/A')}"
                    )

                    st.write(
                        "**Troubleshooting Steps:**"
                    )

                    for step in article.get(
                        "steps",
                        []
                    ):

                        st.write(
                            f"- {step}"
                        )

                    st.write(
                        f"**Escalation:** "
                        f"{article.get('escalation_condition', 'N/A')}"
                    )


        # ====================================================
        # AI RESOLUTION
        # ====================================================

        st.divider()

        st.header(
            "🤖 AI-Generated Resolution"
        )

        st.markdown(
            resolution
        )


        # ====================================================
        # PIPELINE SUMMARY
        # ====================================================

        st.divider()

        st.header(
            "📊 Pipeline Summary"
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "Historical Cases Searched",
                f"{len(historical_cases):,}"
            )

        with col2:

            st.metric(
                "Historical Cases Retrieved",
                len(historical_results)
            )

        with col3:

            st.metric(
                "KB Articles Retrieved",
                len(kb_results)
            )

        st.success(
            "RAG pipeline completed successfully."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Prototype: semantic retrieval + grounded RAG. "
    "Historical cases are supporting evidence; "
    "the knowledge base is the authoritative source."
)