

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
Run the indexer to embed and build the FAISS index:
```bash
uv run python -m utils.indexer
```


## Project Structure
```
.
├── data/
│   ├── ccontrolled_documentation/
│   ├── clean_md_database/
│   └── indexes/
│       ├── all_chunk_sources.json
│       ├── all_chunks.json
│       ├── all_chunks.pkl
│       └── faiss_index.idx
├── src/
│   └── rag/
│       ├── adapters/ # Interface between application and outside world
│       │   ├── cli/
│       │   └── ui/
│       ├── application/ # Business logic for the RAG application.
│       │   ├── ports/
│       │   └── use_cases/
│       └── infrastructure/ # Concrete implementations.
│           ├── chunking/
│           ├── config/
│           ├── converters/
│           ├── llm/
│           ├── logging/
│           └── vectorstores/
├── logs/
├── pyproject.toml
├── README.md
├── utils/
│   ├── chunkers.py
│   ├── converters.py
│   ├── embedders.py
│   ├── indexer.py
│   └── logger.py
└── uv.lock
```

## Roadmap
- [ ] Management conversion per file type
- [ ] Streamlit web interface for local querying
- [ ] Docker container for deployment
- [ ] Incremental indexing
- [ ] Plugin system for external models

## Security
All data is processed locally. No information is sent to external servers. API keys are stored only in your local environment.

## License
MIT License. See `LICENSE` file for details.

## Vision
FULL RAG is not just a retrieval system. It is an attempt to make human knowledge actionable and sovereign.