
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
├── data/
│   ├── ccontrolled_documentation/
│   ├── clean_md_database/
│   └── indexes/
│       ├── all_chunk_sources.json
│       ├── all_chunks.json
│       ├── all_chunks.pkl
│       └── faiss_index.idx
├── src
│   └── rag
│       ├── adapters                        # Adapters layer (CLI, UI, etc.)
│       │   ├── cli                         # Command-line tools (indexing, evaluation)
│       │   │   ├── evaluation_rag_answers.py
│       │   │   └── indexation_documentation.py
│       │   └── ui                          # Streamlit UI
│       │       ├── app.py                  # Streamlit entry point
│       │       ├── ui_pages/               # Application pages
│       │       │   ├── chat.py             # Chat page
│       │       │   └── settings.py         # Configuration/settings page
│       │       ├── components/             # Reusable UI components
│       │       │   ├── message.py
│       │       │   ├── feedback.py
│       │       │   └── sources.py
│       │       ├── layout/                 # Layout utilities (sidebar, theme, etc.)
│       │       │   ├── sidebar.py
│       │       │   └── theme.py
│       │       └── services/               # Services/factories used by the UI
│       │           └── rag_chat.py
│       ├── application                     # RAG system business logic
│       │   ├── ports                       # Abstract interfaces between application and 
│       │   │   ├── chunkers.py
│       │   │   ├── converters.py
│       │   │   ├── embedders.py
│       │   │   ├── evaluation_storage.py
│       │   │   ├── evaluator.py
│       │   │   ├── indexer.py
│       │   │   ├── llm.py
│       │   │   ├── retriever.py
│       │   │   ├── reranker.py
│       │   │   └── vector_store_manager.py
│       │   └── use_cases                   # Concrete orchestrations: indexing, chat, evaluation...
│       │       ├── documents_indexer.py
│       │       ├── intent_classifier.py
│       │       ├── prompting.py
│       │       ├── rag_chat.py
│       │       └── rag_evaluation.py
│       ├── infrastructure                  # Technical implementations
│       │   ├── chunking                    # Document chunking
│       │   │   └── chunkers.py
│       │   ├── config                      # Configuration and environment management
│       │   │   ├── config.py
│       │   │   └── types.py
│       │   ├── converters                  # File conversion (PDF, Markdown, etc.)
│       │   │   ├── converters.py
│       │   │   └── default_exporter.py
│       │   ├── embedders                   # Embedding models
│       │   │   ├── fake_embedder.py
│       │   │   └── mistral_embedder.py
│       │   ├── evaluation                  # RAG evaluation tools
│       │   │   └── ragas_evaluator.py
│       │   ├── llm                         # LLM clients (Mistral, LangChain)
│       │   │   ├── langchain_mistral_client.py
│       │   │   └── mistral_client.py
│       │   ├── logging                     # Logging and instrumentation
│       │   │   ├── interaction_logger.py   # Q/A + contexts -> JSONL
│       │   │   └── logger.py               # App / usage / evaluation loggers
│       │   ├── rerankers                   # Document reranking strategies
│       │   │   ├── cross_encoder_reranker.py
│       │   │   ├── keywords_overlap_scorer.py
│       │   │   └── llm_reranker.py
│       │   ├── storage                     # Storage for evaluation runs
│       │   │   └── evaluation_run_store.py
│       │   └── vectorstores                # Vector stores (FAISS, etc.)
│       │       ├── faiss_store_manager.py
│       │       ├── faiss_store_retriever.py
│       │       └── faiss_store_writer.py
│       └── utils
│           └── utils.py
├── logs/
├── pyproject.toml
├── README.md
└── uv.lock
```

## Roadmap

- [x] Streamlit web interface for local querying
- [x] Database saving
- [ ] Docker container for deployment
- [ ] Tests
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
