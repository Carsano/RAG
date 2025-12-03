
# FULL RAG — FROM KNOWLEDGE TO WISDOM

## Overview

FULL RAG is a local, secure, and modular Retrieval-Augmented Generation system. It allows anyone to create a personal or organizational assistant capable of understanding and answering questions based on internal documentation — entirely offline.

The project aims to transform scattered knowledge into structured wisdom. It is currently under **active development**.

## Core Philosophy

> Knowledge is data. Wisdom is understanding.

FULL RAG bridges the gap by enabling users to convert raw documents into meaningful, queryable insights.

## Key Features

- 100% local processing — no cloud dependency
- Markdown-based knowledge source (all documents are converted to `.md`)
- FAISS vector indexing for efficient similarity search
- Embeddings generated through Mistral AI
- Modular design ready for Streamlit and Docker integration

## Quick Start

### Prerequisites

- Python 3.10+
- A valid Mistral API key

### Installation

```bash
# Clone the repository
git clone https://github.com/Carsano/RAG.git
cd RAG

# init project
uv sync
```

### Configuration

1. Create a `.env` file at the root:

   ```bash
   MISTRAL_API_KEY=your_mistral_key_here
   ```

2. Place your Markdown knowledge files in the `data/` folder.

### Usage

#### 1. Index your documents (CLI)

Run the indexer to embed documents and build the FAISS index:

```bash
uv run python -m src.rag.adapters.cli.indexation_documentation
```

#### 2. Start the assistant (Streamlit UI)

Launch the Streamlit web app:

```bash
uv run streamlit run src/rag/adapters/ui/app.py
```

## Project Structure

```md
.
├── data/                         # Source documents, DB dumps, indexes
│   ├── clean_md_database/        # Normalized Markdown knowledge base
│   ├── controlled_documentation/ # Original controlled docs
│   ├── db/                       # Analytics databases (DuckDB)
│   │   └── rag.duckdb
│   ├── eval/                     # Evaluation datasets and reports
│   └── indexes/                  # Vector store artifacts + metadata
│       ├── all_chunk_sources.json
│       ├── all_chunks.pkl
│       ├── chunker_version.txt
│       └── faiss_index.idx
├── logs/                        # Application, audit, and usage logs
│   ├── app/
│   ├── evaluations/
│   ├── interactions/
│   ├── rag_audit/
│   │   └── rag_audit.jsonl
│   ├── usages/
├── src/
│   └── rag/
│       ├── adapters              # CLI + Streamlit entrypoints
│       │   ├── cli/
│       │   │   ├── evaluation_rag_answers.py
│       │   │   └── indexation_documentation.py
│       │   └── ui/               # Streamlit UI modules
│       │       ├── app.py
│       │       ├── components/   # UI widgets (messages, feedback, sources)
│       │       ├── layout/       # Sidebar, theme, etc.
│       │       ├── services/     # Chat service wiring
│       │       └── ui_pages/     # Chat + settings pages
│       ├── application           # Core use cases + ports
│       │   ├── ports/
│       │   └── use_cases/
│       ├── infrastructure        # Concrete adapters (LLM, retriever…)
│       │   ├── chunking/
│       │   ├── config/
│       │   ├── embedders/
│       │   ├── evaluation/
│       │   ├── llm/
│       │   ├── logging/
│       │   ├── rerankers/
│       │   └── vectorstores/
│       └── utils/                # Shared helpers (logging builder, etc.)
├── pyproject.toml                # Project dependencies & build config
├── README.md
└── uv.lock                       # Locked dependency versions (uv)
```

## Roadmap

- [x] Streamlit web interface for local querying
- [x] Database saving
- [ ] Tests
- [ ] Docker container for deployment
- [ ] Analytics (Ragas, usage)
- [ ] Incremental indexing
- [ ] Plugin system for external models

## Security

All data is processed locally. No information is sent to external servers. API keys are stored only in your local environment.

## License

MIT License. See `LICENSE` file for details.

## Vision

FULL RAG is not just a retrieval system. It is an attempt to make human knowledge actionable and sovereign.

## Database structure

### A. Metadata

- `request_id`
- `timestamp`
- `user_id`
- `pipeline_version`
- `components`:
  - `llm_version`
  - `embedder_version`
  - `reranker_version`
  - `retriever_version`
  - `chunker_version`

### B. User Input

- `raw_question`
- `input_parameters`

### C. Effective Parameters

- `effective_parameters`

### D. Retrieval & Reranking

- `retrieval`:
  - `top_k_returned`
  - `chunks`:
    - `chunk_id`
    - `score_retriever`
    - `content_snapshot`
    - `source_path`
- `rerank`:
  - `chunks_after_rerank`
  - `score_reranker`

### E. LLM Generation

- `prompt_final`
- `llm_metadata`:
  - `input_tokens`
  - `output_tokens`
  - `total_tokens`
  - `latency_ms`
- `raw_answer`
- `clean_answer`

### F. RAG Metrics

- `faithfulness`
- `answer_relevancy`
- `context_precision`
- `context_recall`

### G. Integrity

- `integrity_hash` (SHA-256)

---

## 4. Relational Tables (Analytics)

### Table: request

- request_id (PK)
- timestamp
- user_id
- llm_version
- embedder_version
- reranker_version
- retriever_version
- chunker_version
- latency_retrieval_ms
- latency_rerank_ms
- latency_generation_ms
- total_latency_ms
- input_tokens
- output_tokens
- total_tokens

### Table: rag_metrics

- request_id (FK)
- faithfulness
- answer_relevancy
- context_precision
- context_recall

### Table: chunks_used

- id
- request_id (FK)
- chunk_snapshot_text
- chunk_source_path
- retriever_score
- reranker_score
- was_selected

### Table: users

- user_id (PK)
- creation_date

---
