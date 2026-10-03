# Intelligent Support Ticket Resolution Assistant

## Overview

This project implements an AI-powered support ticket resolution assistant for a telecom support desk.

The system takes a customer complaint, identifies important characteristics of the issue, retrieves semantically similar historical support cases and knowledge-base articles, and uses Retrieval-Augmented Generation (RAG) to produce a grounded resolution with source citations.

## Problem

Traditional keyword-based ticket search can miss complaints that describe the same problem using different words.

For example:

> "My calls keep dropping and my mobile internet is very slow."

A semantic retrieval system can identify related historical cases and knowledge-base articles even when the wording is different.

## Solution

The system follows this pipeline:

```text
Customer Complaint
        ↓
Preprocessing & PII Redaction
        ↓
Complaint Analysis
        ↓
Embedding Generation
        ↓
Semantic Retrieval
        ↓
Historical Cases + Knowledge Base
        ↓
Context Assembly
        ↓
LLM-based RAG Generation
        ↓
Grounded Resolution + Citations
        ↓
Evaluation & Monitoring
```
## Key Features

- Customer complaint analysis
- Intent and category identification
- Product identification
- Severity estimation
- Customer sentiment estimation
- Semantic search using sentence embeddings
- Historical support case retrieval
- Knowledge-base retrieval
- Retrieval-Augmented Generation (RAG)
- Grounded troubleshooting recommendations
- Knowledge-base source citations
- Retrieval and response evaluation
- PII redaction in historical data
## Technologies Used

- Python
- Hugging Face Datasets
- Sentence Transformers
- `all-MiniLM-L6-v2`
- Ollama
- Llama 3.2 3B
- JSON
- Pandas
## Dataset

The project uses the `PRAPAREE/telecom-conversation-corpus` dataset from Hugging Face.

The dataset contains synthetic telecom customer-service conversations covering issues such as:

- Dropped calls
- Poor network reception
- Slow mobile data
- Network connectivity problems
- Troubleshooting interactions
- Support escalations

The original dataset contains approximately 3.7 million conversation records and more than 228,000 unique conversations.

For this prototype, selected historical conversations are transformed into structured support cases.
## Knowledge Base

The prototype contains four knowledge-base articles:

- KB001 — Troubleshooting Dropped Calls
- KB002 — Troubleshooting Slow Mobile Data
- KB003 — Troubleshooting Poor Network Reception
- KB004 — Network Issue Escalation

The knowledge base provides the primary source for troubleshooting instructions.
## RAG Pipeline

The system uses the following RAG process:

1. Convert the customer complaint into an embedding.
2. Convert historical cases and knowledge-base articles into embeddings.
3. Calculate cosine similarity.
4. Retrieve the most relevant historical cases.
5. Retrieve the most relevant knowledge-base articles.
6. Combine the retrieved information into context.
7. Send the context and complaint to the LLM.
8. Generate a grounded support resolution.
9. Include citations to the knowledge-base sources.

The LLM is instructed not to invent troubleshooting steps that are unsupported by the retrieved context.
## Example

### Customer Complaint

> My calls keep dropping and my mobile internet is very slow.

### Complaint Analysis

- Intent: Network connectivity issue
- Category: Dropped calls, Slow mobile data
- Product: Mobile voice and data
- Severity: Medium
- Sentiment: Neutral

### Retrieved Knowledge Base

- KB001 — Troubleshooting Dropped Calls
- KB002 — Troubleshooting Slow Mobile Data

### Generated Resolution

Recommended troubleshooting steps include:

1. Restart the phone
2. Check for software updates
3. Switch network mode
4. Reset network settings

If the issue continues after troubleshooting, the issue can be escalated to network support.
## Evaluation

A prototype evaluation was performed using five manually designed test cases covering different telecom support scenarios.

The evaluation measured:

- Retrieval relevance
- Source correctness
- Troubleshooting-step coverage
- Retrieval latency

### Prototype Results

| Metric | Result |
|---|---:|
| Test cases | 5 |
| Average retrieval relevance | 100% |
| Source correctness | 100% |
| Troubleshooting-step coverage | 100% |
| Average retrieval latency | 0.0154 seconds |

These results represent the current prototype evaluation and should not be interpreted as production-level accuracy.
## Production Scale Considerations

The current implementation is a prototype. A production deployment can scale the architecture using:

### 1. Vector Database

Store embeddings in a vector database such as FAISS, Qdrant, pgvector, or another production vector store.

### 2. Hybrid Retrieval

Combine semantic retrieval with keyword and metadata-based retrieval.

### 3. Reranking

Use a reranking model to improve the ordering of retrieved historical tickets and knowledge-base articles.

### 4. Incremental Indexing

New tickets and knowledge-base articles can be continuously processed and added to the retrieval index without rebuilding the entire dataset.

### 5. Evolving Ticket Classes

Monitor emerging issue patterns and update categories, metadata, and knowledge-base content as new ticket types appear.

### 6. PII Protection

Sensitive customer information should be detected and redacted before data is stored or indexed.

### 7. Monitoring

Production monitoring should track:

- Retrieval relevance
- Response quality
- Citation correctness
- Latency
- Escalation rate
- User feedback
- Failure cases
- Data and category drift

### 8. Scalability

Model serving, embedding generation, retrieval, and indexing can be independently scaled based on workload.
## Project Structure

ticket_resolution_ai/
│
├── final_demo.py
├── rag_generator.py
├── query_analysis.py
├── semantic_search.py
├── kb_search.py
├── combined_retrieval.py
├── evaluate_system.py
├── prepare_cases.py
├── check_data.py
├── test_embedding.py
├── test_rag_embeddings.py
│
├── historical_cases.json
├── knowledge_base.json
├── evaluation_results.json
│
├── architecture.md
├── requirements.txt
└── README.md

## Running the Project
### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and run Ollama

Install Ollama and make sure the required model is available:

```bash
ollama pull llama3.2:3b
```

### 3. Run the final demonstration

```bash
python final_demo.py
```

The program performs:

```text
Complaint Analysis
       ↓
Semantic Retrieval
       ↓
RAG Generation
       ↓
Grounded Resolution
       ↓
Source Citations
```
## Future Improvements

The prototype can be further improved by:

- Using a production vector database
- Indexing the complete historical conversation corpus
- Adding a dedicated reranking model
- Improving intent and sentiment classification
- Adding automated evaluation datasets
- Adding real-time ticket ingestion
- Adding human feedback loops
- Adding authentication and access control
- Deploying the system as an API/service

## Architecture

See [`architecture.md`](architecture.md) for the system architecture diagram and production-scale design.

## Disclaimer

This project is a prototype implementation demonstrating semantic retrieval and RAG-based support ticket resolution. The current evaluation is limited to a small manually designed test set and should not be treated as production validation.