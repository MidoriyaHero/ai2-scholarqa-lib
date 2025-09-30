"""
FastAPI application for ASTA ScholarQA service.

This service provides academic paper search and question-answering capabilities
using Semantic Scholar data with reranking and LLM-based answer generation.
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from api.v1.asta import router as asta_router, preload_reranker

# Create FastAPI app
app = FastAPI(
    title="ASTA ScholarQA API",
    description="Academic paper search and question-answering service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(asta_router)

# Startup event to preload reranker model
@app.on_event("startup")
async def startup_event():
    """Preload heavy models at startup for better response times."""
    preload_reranker()

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint providing service information.

    Returns:
        dict: Service information including name, version, and available endpoints
    """
    return {
        "service": "ASTA ScholarQA API",
        "version": "1.0.0",
        "endpoints": {
            "qa": "/api/v1/asta/qa",
            "search": "/api/v1/asta/search",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.

    Returns:
        dict: Health status
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,  # ASTA service on port 8002, backend on 8001
        reload=True,
        log_level="info"
    )