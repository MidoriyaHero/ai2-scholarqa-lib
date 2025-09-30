"""
ASTA ScholarQA API endpoints.

Provides REST API for academic paper search and question-answering
using Semantic Scholar data with reranking capabilities.
"""
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException
import logging

from api.scholarqa import ScholarQA
from api.scholarqa.rag.retrieval import PaperFinderWithReranker
from api.scholarqa.rag.retriever_base import FullTextRetriever
from api.scholarqa.rag.reranker.reranker_base import CrossEncoderScores
from api.scholarqa.llms.constants import GEMINI_25_FLASH

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/asta", tags=["ASTA QA"])

# Global reranker instance - loaded once at startup
_reranker_instance = None

def get_reranker():
    """Get or initialize the global reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        log.info("[ASTA][STARTUP] Loading reranker model: mixedbread-ai/mxbai-rerank-large-v1")
        _reranker_instance = CrossEncoderScores("mixedbread-ai/mxbai-rerank-large-v1")
        log.info("[ASTA][STARTUP] Reranker model loaded successfully")
    return _reranker_instance

def preload_reranker():
    """Preload the reranker model at application startup."""
    log.info("[ASTA][PRELOAD] Starting reranker model preload...")
    get_reranker()
    log.info("[ASTA][PRELOAD] Reranker model preload completed")

class QARequest(BaseModel):
    """Request schema for ScholarQA endpoint."""
    query: str = Field(..., min_length=1)

class QAResult(BaseModel):
    """Response schema wrapping ScholarQA result payload."""
    result: Dict[str, Any]

class SearchResult(BaseModel):
    """Response schema for search endpoint."""
    papers: List[Dict[str, Any]]
    total_count: int

@router.post("/qa", response_model=QAResult)
async def asta_qa(req: QARequest) -> QAResult:
    """
    Run the ScholarQA pipeline end-to-end for a user query.

    Flow:
    - Retrieve passages via Semantic Scholar full-text and keyword search
    - Rerank passages with a cross-encoder
    - Generate structured answer with citations via LLM

    Args:
        req: QARequest containing the query

    Returns:
        QAResult: Structured response with answer sections and citations
    """
    try:
        log.info(f"[ASTA][API] /asta/qa start | query='{req.query}'")

        # Build retrieval + reranking stack
        retriever = FullTextRetriever(n_retrieval=100, n_keyword_srch=20)

        # Use pre-loaded reranker instance
        reranker = get_reranker()

        # Wrap into PaperFinderWithReranker; n_rerank=-1 keeps all candidates post-rerank
        paper_finder = PaperFinderWithReranker(
            retriever=retriever,
            reranker=reranker,
            n_rerank=-1,
            context_threshold=0.0
        )

        # ScholarQA orchestrates the multi-step QA with the chosen LLM
        scholar_qa = ScholarQA(paper_finder=paper_finder, llm_model=GEMINI_25_FLASH)
        result = scholar_qa.answer_query(req.query)

        log.info("[ASTA][API] /asta/qa done | sections=%s", len(result.get("sections", [])) if isinstance(result, dict) else 'N/A')
        return QAResult(result=result)
    except Exception as e:
        log.exception("ASTA ScholarQA failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", response_model=SearchResult)
async def search(req: QARequest) -> SearchResult:
    """
    Search for papers using PaperFinder without full QA pipeline.

    Returns a list of relevant papers with metadata and text snippets.

    Args:
        req: QARequest containing the search query

    Returns:
        SearchResult: List of papers with metadata and total count
    """
    def log_callback(message: str, level: str = "info"):
        """Callback to forward logs to the backend logger"""
        if level == "info":
            log.info(f"[ASTA][RETRIEVAL] {message}")
        elif level == "warning":
            log.warning(f"[ASTA][RETRIEVAL] {message}")
        elif level == "error":
            log.error(f"[ASTA][RETRIEVAL] {message}")
        else:
            log.debug(f"[ASTA][RETRIEVAL] {message}")

    try:
        retriever = FullTextRetriever(n_retrieval=100, n_keyword_srch=20)

        # Use pre-loaded reranker instance
        reranker = get_reranker()

        paper_finder = PaperFinderWithReranker(
            retriever=retriever,
            reranker=reranker,
            n_rerank=-1,
            context_threshold=0.0
        )
        log.info(f"[ASTA][API] /asta/search start | query='{req.query}'")

        # Set callback on paper_finder if it supports it
        if hasattr(paper_finder, 'set_log_callback'):
            paper_finder.set_log_callback(log_callback)

        log_callback("Starting paper search with PaperFinder")

        # Get search results as DataFrame
        results_df = paper_finder.search(req.query, log_callback=log_callback)

        # Convert DataFrame to list of dictionaries
        if results_df.empty:
            papers_list = []
            log_callback("No papers found for query", "warning")
        else:
            papers_list = results_df.to_dict('records')
            log_callback(f"Successfully found {len(papers_list)} papers")

        log.info(f"[ASTA][API] /asta/search done | papers_found={len(papers_list)}")
        return SearchResult(papers=papers_list, total_count=len(papers_list))

    except Exception as e:
        log_callback(f"Search failed with error: {str(e)}", "error")
        log.exception("ASTA search failed")
        raise HTTPException(status_code=500, detail=str(e))