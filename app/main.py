from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import logging
from app.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag: RAGPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    logger.info("Initializing Order Intelligence RAG pipeline...")
    rag = RAGPipeline()
    rag.ingest_documents()
    logger.info("RAG pipeline ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Order Intelligence RAG Assistant",
    description=(
        "An AI-powered assistant for e-commerce operations teams. "
        "Ask natural language questions about order management SOPs, "
        "carrier SLAs, return policies, and support runbooks."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the SLA for resolving a false delivery complaint?"
            }
        }


class QueryResponse(BaseModel):
    question: str
    answer: str
    source_chunks: list[str]
    model_used: str


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "running",
        "service": "Order Intelligence RAG Assistant",
        "version": "1.0.0",
        "description": "Natural language querying over order management SOPs, carrier SLAs, and support runbooks.",
        "endpoints": {
            "query": "POST /query",
            "health": "GET /health",
            "sample_questions": "GET /sample-questions",
            "docs": "GET /docs"
        }
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "vectorstore": "loaded", "documents": "3 sources indexed"}


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = rag.query(request.question)
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            source_chunks=result["source_chunks"],
            model_used=result["model_used"]
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")


@app.get("/sample-questions", tags=["RAG"])
def sample_questions():
    return {
        "categories": {
            "Order Lifecycle": [
                "What are the different states an order can be in?",
                "Can a customer cancel an order that is already packed?",
                "What happens after 3 failed delivery attempts?"
            ],
            "Returns & Refunds": [
                "What is the return window for electronics?",
                "How long does a UPI refund take?",
                "Can flash sale items be returned for a refund?"
            ],
            "Carrier & Delivery": [
                "What is the SLA for Tier-2 city deliveries?",
                "Which carrier is preferred for reverse logistics?",
                "What penalty applies for a false delivery by a carrier?"
            ],
            "Escalations & Support": [
                "What are the escalation triggers for an order issue?",
                "How should a false delivery complaint be handled?",
                "What compensation does a customer get for a delayed order?"
            ],
            "Fraud & Risk": [
                "What triggers a high-risk order flag?",
                "How long does a fraud review take?",
                "What is the chargeback rate threshold before a risk review?"
            ]
        }
    }
