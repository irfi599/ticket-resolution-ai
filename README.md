# 🤖 Intelligent Support Ticket Resolution Assistant

An AI-powered telecom support assistant that uses **semantic search, historical support cases, a knowledge base, and Retrieval-Augmented Generation (RAG)** to help support agents resolve customer complaints.

---

## 📌 Problem Statement

Telecom support agents often search historical tickets and knowledge-base articles using keyword-based search.

This can fail when two customers describe the same issue using different words.

For example:

> "My calls keep dropping and my mobile internet is very slow."

A keyword-based system may fail to retrieve relevant historical cases if the wording is different.

The goal of this project is to build an intelligent support assistant that:

1. Understands the customer's complaint.
2. Identifies the intent, category, product, severity, and sentiment.
3. Retrieves semantically similar historical support cases.
4. Retrieves relevant knowledge-base articles.
5. Uses an LLM to generate a grounded resolution.
6. Provides citations to the knowledge-base sources used.
7. Supports a scalable architecture for evolving support data.

---

# 🎯 Solution Overview

The system follows a Retrieval-Augmented Generation architecture.

```text
Customer Complaint
        │
        ▼
Complaint Analysis
        │
        ├── Intent
        ├── Category
        ├── Product
        ├── Severity
        └── Sentiment
        │
        ▼
Query Embedding
        │
        ▼
Semantic Retrieval
        │
        ├───────────────┐
        ▼               ▼
Historical Cases    Knowledge Base
        │               │
        └───────┬───────┘
                ▼
        Retrieved Context
                │
                ▼
          Ollama LLM
                │
                ▼
      Grounded Resolution
                │
                ▼
       Sources / Citations
```

---

# 🧠 Key Features

### 1. Complaint Analysis

The system analyzes the customer complaint and extracts:

- Intent
- Category
- Product
- Severity
- Customer sentiment

Example:

```text
Complaint:
"My calls keep dropping and my mobile internet is very slow."

Intent:
Network connectivity issue

Categories:
- Dropped calls
- Slow mobile data

Product:
Mobile voice and data

Severity:
Medium

Sentiment:
Neutral
```

> Note: The current prototype uses lightweight rule-based heuristics for complaint analysis. A production implementation can replace these rules with a trained classifier or LLM-based structured extraction.

---

### 2. Semantic Search

Instead of relying only on keyword matching, the system converts the complaint into an embedding using:

```text
all-MiniLM-L6-v2
```

The embedding is compared with embeddings of historical support cases.

This allows semantically similar complaints to be retrieved even when their wording differs.

---

### 3. Historical Support Cases

The prototype contains:

```text
9,926 usable historical support cases
```

These cases were derived from the Telecom Conversation Corpus.

The original dataset contains millions of conversation messages across more than 228,000 conversations.

The project uses a prepared subset of 10,000 conversations, resulting in 9,926 usable historical cases after preprocessing.

Each historical case contains information such as:

- Issue type
- Device
- Sentiment
- Troubleshooting steps
- Outcome
- Conversation identifier

---

### 4. Knowledge Base Retrieval

The system maintains a structured telecom knowledge base containing troubleshooting articles.

Current knowledge-base articles include:

```text
KB001 — Troubleshooting Dropped Calls

KB002 — Troubleshooting Slow Mobile Data

KB003 — Troubleshooting Poor Network Reception

KB004 — Network Issue Escalation
```

Each article contains:

- Problem description
- Symptoms
- Troubleshooting steps
- Escalation condition

---

# 🔎 Retrieval Pipeline

For every incoming complaint:

### Step 1 — Encode the complaint

The complaint is converted into a vector representation using:

```text
all-MiniLM-L6-v2
```

### Step 2 — Search historical cases

The query embedding is compared with the pre-computed historical case embeddings.

The top 5 most similar cases are retrieved.

### Step 3 — Search the knowledge base

The same query is compared with the knowledge-base embeddings.

The top 2 relevant knowledge-base articles are retrieved.

### Step 4 — Build RAG context

The retrieved knowledge-base articles and historical cases are provided as context to the LLM.

### Step 5 — Generate resolution

The local LLM generates a concise resolution using the retrieved information.

The knowledge base is treated as the authoritative source for troubleshooting instructions.

---

# 🤖 Large Language Model

The project uses:

```text
Ollama
└── llama3.2:3b
```

The model runs locally rather than depending on an external cloud LLM API.

This makes the prototype suitable for experimentation with locally hosted inference.

---

# 📚 Grounded RAG

The system explicitly instructs the LLM to:

- Use the knowledge base as the primary source.
- Use historical cases as supporting evidence.
- Avoid inventing troubleshooting steps.
- Avoid inventing escalation methods.
- Cite the relevant knowledge-base article IDs.
- Avoid unsupported customer information.
- Avoid claiming guaranteed resolution.

Example output:

```text
Issue Summary:

The customer is experiencing dropped calls and slow mobile data.

Recommended Steps:

1. Restart the phone (KB001, Step 1)
2. Check for software updates (KB001, Step 2)
3. Switch network mode (KB001, Step 3)
4. Reset network settings (KB001, Step 4)

Escalation Guidance:

If the issue persists after trying the recommended steps,
escalate to network support.

Sources:

KB001: Troubleshooting Dropped Calls
KB002: Troubleshooting Slow Mobile Data
```

---

# 🖥️ Streamlit Interface

The project includes an interactive Streamlit UI.

The interface displays:

### Complaint Analysis

- Intent
- Product
- Severity
- Sentiment
- Categories

### Similar Historical Cases

The top 5 semantically similar historical cases are displayed with:

- Similarity score
- Case ID
- Device
- Sentiment
- Troubleshooting steps
- Outcome

### Retrieved Knowledge Base

The top 2 relevant articles are displayed with:

- Article ID
- Problem
- Symptoms
- Supported steps
- Escalation guidance

### AI-Generated Resolution

The final grounded RAG response is displayed to the support agent.

### System Information

The interface also displays:

```text
Historical cases searched: 9,926
Historical cases retrieved: 5
KB articles retrieved: 2
```

---

# 📊 Evaluation

A prototype evaluation was created using 5 manually designed test cases covering:

1. Dropped calls + slow mobile internet
2. Poor network reception
3. Slow mobile data
4. Calls disconnecting
5. Persistent network problems after troubleshooting

### Prototype Results

| Metric | Result |
|---|---:|
| Test cases | 5 |
| Retrieval relevance | 100% |
| Source correctness | 100% |
| Troubleshooting step coverage | 100% |
| Average retrieval latency | 0.0154 seconds |

These results represent a **small prototype evaluation**, not production-level accuracy.

The evaluation is stored in:

```text
evaluation_results.json
```

---

# 📈 System Health and Monitoring

The project evaluates several aspects of system behavior:

### Retrieval Relevance

Checks whether the expected knowledge-base articles are retrieved.

### Source Correctness

Checks whether the generated response uses the expected sources.

### Troubleshooting Step Coverage

Checks whether the generated response contains the required troubleshooting steps from the knowledge base.

### Retrieval Latency

Measures the time required for semantic retrieval.

---

# 🏗️ Production Scale Considerations

The current implementation is a working prototype. A production deployment would require several improvements.

## 1. Vector Database

Instead of storing embeddings in a NumPy file, a production system could use a vector database such as:

- FAISS
- Qdrant
- Milvus
- pgvector

This would support efficient large-scale retrieval.

---

## 2. Incremental Data Ingestion

New support tickets should not require rebuilding the entire index.

A production pipeline could:

```text
New Ticket
    ↓
PII Redaction
    ↓
Preprocessing
    ↓
Embedding
    ↓
Vector Database
```

This allows the retrieval system to continuously incorporate new support cases.

---

## 3. Hybrid Retrieval

Pure semantic search can occasionally retrieve noisy results.

A production system could combine:

```text
Semantic similarity
        +
Keyword/BM25 search
        +
Metadata filtering
        +
Reranking
```

For example, metadata such as:

- Product
- Device
- Issue category
- Region
- Ticket status

could be used to improve retrieval precision.

---

## 4. Reranking

The initial vector search could retrieve the top 20–50 candidates.

A reranker could then select the most relevant 5–10 cases before sending them to the LLM.

```text
Query
  ↓
Vector Retrieval
  ↓
Top 50 candidates
  ↓
Reranker
  ↓
Top 5 relevant cases
  ↓
LLM
```

---

## 5. PII Protection

Customer support conversations can contain sensitive information.

A production system should apply PII detection and redaction before indexing.

Examples include:

- Phone numbers
- Account numbers
- PINs
- Names
- Addresses
- Email addresses

The prototype already performs basic redaction during historical-case preparation.

---

## 6. Access Control

A production system should implement:

- Authentication
- Role-based access control
- Secure API endpoints
- Audit logging
- Data access policies

---

## 7. LLM Guardrails

The generation layer should validate that:

- Recommended steps exist in the knowledge base.
- Sources actually exist.
- Unsupported claims are removed.
- Escalation guidance is grounded.
- Sensitive customer information is not exposed.

---

## 8. Monitoring

Production monitoring should track:

```text
Retrieval latency
LLM latency
Retrieval relevance
Source correctness
Grounding failures
Hallucination rate
User feedback
Escalation rate
Resolution rate
```

---

## 9. Data Drift

Telecom issues and ticket categories can evolve over time.

The system should monitor:

- New issue categories
- Changes in vocabulary
- New devices
- New products
- Changes in resolution patterns

This allows the retrieval and classification pipeline to evolve with the support data.

---

# 🔄 Future Architecture

A production version could use:

```text
                    ┌──────────────────────┐
                    │   Support Agent UI   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Complaint Processing │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
        Intent/Category     Severity        Sentiment
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                       Query Embedding
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Hybrid Retrieval     │
                    │ Vector + Keyword     │
                    └──────────┬───────────┘
                               │
                               ▼
                         Reranking
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
             Historical Tickets       Knowledge Base
                    │                      │
                    └──────────┬───────────┘
                               ▼
                        Context Builder
                               │
                               ▼
                         Local / Cloud LLM
                               │
                               ▼
                     Grounding Validation
                               │
                               ▼
                     Resolution + Sources
                               │
                               ▼
                         Support Agent
```

---

# 📁 Project Structure

```text
ticket-resolution-ai/
│
├── app.py
│
├── final_demo.py
│
├── query_analysis.py
│
├── semantic_search.py
│
├── kb_search.py
│
├── combined_retrieval.py
│
├── rag_generator.py
│
├── evaluate_system.py
│
├── test_embedding.py
│
├── test_rag_embeddings.py
│
├── prepare_cases.py
│
├── prepare_10000_cases.py
│
├── check_data.py
│
├── historical_cases.json
│
├── historical_cases_10000.json
│
├── knowledge_base.json
│
├── evaluation_results.json
│
├── architecture.md
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core development |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Semantic embedding model |
| NumPy | Vector similarity |
| Hugging Face Datasets | Dataset loading |
| Ollama | Local LLM inference |
| Llama 3.2 3B | Response generation |
| Streamlit | Interactive UI |
| JSON | Knowledge base and processed data |
| Git | Version control |
| GitHub | Source-code repository |

---

# 📦 Dataset

The project uses the:

**Telecom Conversation Corpus**

Source:

```text
PRAPAREE/telecom-conversation-corpus
```

The dataset contains synthetic telecom customer-service conversations covering issues such as:

- Dropped calls
- Poor network reception
- Slow mobile data
- Network problems
- Troubleshooting
- Support escalation

The project prepares a subset of 10,000 conversations and produces:

```text
9,926 usable historical cases
```

after preprocessing.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/irfi599/ticket-resolution-ai.git
```

Move into the project:

```bash
cd ticket-resolution-ai
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

# 🤖 Install and Run Ollama

Install Ollama and make sure the required model is available:

```bash
ollama pull llama3.2:3b
```

Test the model:

```bash
ollama run llama3.2:3b
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 🧪 Run the Evaluation

Run:

```bash
python evaluate_system.py
```

The evaluation results are saved to:

```text
evaluation_results.json
```

---

# 🔬 Run the Command-Line Demo

Run:

```bash
python final_demo.py
```

This demonstrates the complete pipeline:

```text
Complaint
   ↓
Complaint Analysis
   ↓
Historical Semantic Search
   ↓
Knowledge Base Search
   ↓
RAG Context Construction
   ↓
Ollama
   ↓
Grounded Resolution
```

---

# ⚠️ Prototype Limitations

This implementation is a hackathon prototype rather than a production deployment.

Current limitations include:

- Rule-based complaint classification
- Rule-based sentiment detection
- Rule-based severity estimation
- A limited knowledge base
- 9,926 indexed historical cases rather than the complete dataset
- NumPy-based embedding storage instead of a production vector database
- Local LLM inference
- Prototype-scale evaluation

These components can be replaced or extended for production deployment.

---

# 🚀 Future Improvements

Potential extensions include:

- Hybrid semantic + keyword retrieval
- Vector database integration
- Cross-encoder reranking
- LLM-based structured complaint classification
- Automated PII detection
- Continuous ticket ingestion
- Retrieval feedback loops
- Human-in-the-loop corrections
- Automated hallucination detection
- Production monitoring dashboards
- API-based microservice deployment
- Docker containerization
- Authentication and role-based access control

---

# 🏁 Conclusion

The Intelligent Support Ticket Resolution Assistant demonstrates how semantic retrieval and Retrieval-Augmented Generation can improve telecom support workflows.

Instead of relying only on keyword matching, the system:

```text
Understands the complaint
        ↓
Finds semantically similar cases
        ↓
Retrieves relevant knowledge
        ↓
Generates a grounded resolution
        ↓
Provides source citations
```

The prototype combines historical support data, a structured knowledge base, semantic search, and a locally hosted LLM into an interactive support-assistance workflow.

---

## 👩‍💻 Project

**Intelligent Support Ticket Resolution Assistant**

Built as a hackathon project using Python, semantic search, RAG, Ollama, and Streamlit.