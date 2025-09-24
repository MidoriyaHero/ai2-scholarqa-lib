# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ai2 Scholar QA is a scientific query answering and literature review generation system built on RAG architecture with:
- Retrieval component using Semantic Scholar API (11M+ full text, 100M+ abstracts)
- Multi-step generation pipeline using Claude Sonnet 3.7 (default)
- Web UI (React/TypeScript), API (FastAPI), and Python package

## Development Commands

### Python API Development
```bash
# Install dependencies (Python 3.11.3)
cd api
pip install -r requirements.txt

# Install with sentence transformer models
pip install -e ".[all]"

# Run development server
./dev.sh  # Runs uvicorn on port 8000 with reload

# Format code
black .

# Type checking
mypy .
```

### UI Development  
```bash
cd ui
npm install

# Development server
npm start  # Webpack dev server on port 4000

# Build production
npm run build

# Lint and fix
npm run lint
npm run lint:fix
```

### Docker Development
```bash
# Run full stack (API, UI, proxy, sonar)
docker compose up --build

# Verbose build output
docker compose build --progress plain
```

## Architecture

### Core Components

1. **ScholarQA** (`api/scholarqa/scholar_qa.py`): Main pipeline orchestrator
   - Preprocesses queries with LLM to extract filters
   - Manages retrieval and multi-step generation

2. **PaperFinder** (`api/scholarqa/rag/retrieval.py`): Retrieval system
   - FullTextRetriever: Semantic Scholar API integration
   - Reranker: mixedbread-ai/mxbai-rerank-large-v1 or Modal deployment

3. **MultiStepQAPipeline** (`api/scholarqa/rag/multi_step_qa_pipeline.py`): Generation pipeline
   - Quote extraction from papers
   - Planning/clustering quotes into sections
   - Summary generation with citations

### API Structure
- `api/scholarqa/app.py`: FastAPI async endpoints
- `api/scholarqa/llms/`: LLM integration and prompts
- `api/scholarqa/rag/`: Retrieval and generation components
- `api/scholarqa/config/`: Configuration management
- `api/run_configs/default.json`: Runtime configuration

### Key API Endpoints
- `POST /query_corpusqa`: Submit query, returns task_id
- `GET /query_corpusqa/{task_id}`: Poll for results

## Environment Setup

Required environment variables in `.env`:
```bash
S2_API_KEY=           # Semantic Scholar API key
ANTHROPIC_API_KEY=    # Claude API key  
OPENAI_API_KEY=       # GPT-4o fallback & moderation
```

Optional for Modal deployment:
```bash
MODAL_TOKEN=
MODAL_TOKEN_SECRET=
```

## Extension Points

- Custom retrievers: Extend `AbstractRetriever`
- Custom rerankers: Extend `AbstractReranker`  
- Custom paper finders: Extend `AbsPaperFinder`
- Custom API endpoints: Add to `app.py` or create custom router
- Custom ScholarQA: Subclass or reimplement `lazy_load_scholarqa`

## Configuration

Runtime config (`api/run_configs/default.json`):
- `retrieval_service`: "public_api" for Semantic Scholar
- `reranker_service`: "modal" or "cross_encoder" 
- `n_retrieval`: Number of passages to retrieve (256)
- `n_rerank`: Top passages after reranking (50)
- `llm`: Model for generation (anthropic/claude-3-5-sonnet-20241022)