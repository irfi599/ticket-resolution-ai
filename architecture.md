# System Architecture

## Intelligent Support Ticket Resolution Assistant

The system uses a Retrieval-Augmented Generation (RAG) architecture to help telecom support agents resolve customer complaints using historical support cases and a structured knowledge base.

---

## 1. High-Level Architecture

```text
                    ┌─────────────────────────────┐
                    │      Streamlit UI            │
                    │   Support Agent Interface    │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │   Customer Complaint         │
                    │        Analysis              │
                    │                              │
                    │ • Intent                     │
                    │ • Category                   │
                    │ • Product                    │
                    │ • Severity                   │
                    │ • Sentiment                  │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ Query Embedding              │
                    │ all-MiniLM-L6-v2             │
                    └──────────────┬──────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     │                           │
                     ▼                           ▼
          ┌─────────────────────┐     ┌─────────────────────┐
          │ Historical Cases    │     │ Knowledge Base      │
          │                     │     │                     │
          │ 9,926 cases         │     │ KB001               │
          │ Pre-computed        │     │ KB002               │
          │ embeddings          │     │ KB003               │
          │                     │     │ KB004               │
          └──────────┬──────────┘     └──────────┬──────────┘
                     │                           │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │   Semantic Retrieval         │
                    │                              │
                    │ Top 5 historical cases       │
                    │ Top 2 KB articles            │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │      RAG Context Builder     │
                    │                              │
                    │ KB = Primary Source          │
                    │ Historical cases = Evidence  │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │       Ollama LLM             │
                    │        llama3.2:3b            │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ Grounded Resolution          │
                    │                              │
                    │ • Issue Summary              │
                    │ • Troubleshooting Steps      │
                    │ • Escalation Guidance        │
                    │ • KB Source Citations        │
                    └─────────────────────────────┘