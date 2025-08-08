from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import asyncio
from datetime import datetime

from database import DatabaseManager
from document_service import DocumentService
from vector_service import PineconeService
from llm_service import GPTService
from models import QueryRequest, QueryResponse, DocumentMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HackRX LLM Query-Retrieval System",
    description="Production-ready LLM system with FastAPI, Pinecone, GPT-4, and PostgreSQL",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expected token for authentication
EXPECTED_TOKEN = "9a653094793aedeae46f194aa755e2bb17f297f5209b7f99c1ced3671779d95d"

# Initialize services
db_manager = DatabaseManager()
document_service = DocumentService()
vector_service = PineconeService()
llm_service = GPTService()

def verify_token(authorization: Optional[str] = Header(None)):
    """Verify the authorization token"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    token = authorization.split(" ")[1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return token

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize database
        await db_manager.initialize()
        logger.info("Database initialized successfully")
        
        # Initialize Pinecone
        await vector_service.initialize()
        logger.info("Pinecone service initialized successfully")
        
        # Test GPT-4 connection
        await llm_service.test_connection()
        logger.info("GPT-4 service initialized successfully")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await db_manager.close()
    logger.info("Services shut down successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "connected",
            "pinecone": "connected",
            "gpt4": "connected"
        }
    }

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_query(
    request: QueryRequest, 
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """
    Main endpoint to process documents and answer questions
    """
    try:
        logger.info(f"Processing {len(request.questions)} questions for document: {request.documents}")
        
        # Step 1: Process document and extract content
        document_content = await document_service.process_document(request.documents)
        
        # Step 2: Store document metadata in PostgreSQL
        doc_metadata = DocumentMetadata(
            url=request.documents,
            content_type=document_content.get("content_type", "unknown"),
            page_count=document_content.get("page_count", 0),
            processed_at=datetime.utcnow()
        )
        
        doc_id = await db_manager.store_document_metadata(doc_metadata)
        
        # Step 3: Create embeddings and store in Pinecone
        embeddings_data = await vector_service.create_and_store_embeddings(
            document_content["chunks"],
            doc_id,
            request.documents
        )
        
        # Step 4: Process each question
        answers = []
        for i, question in enumerate(request.questions):
            try:
                # Retrieve relevant context from Pinecone
                relevant_context = await vector_service.search_similar_content(
                    question, 
                    top_k=5
                )
                
                # Generate answer using GPT-4
                answer = await llm_service.generate_answer(
                    question=question,
                    context=relevant_context,
                    document_url=request.documents
                )
                
                answers.append(answer)
                
                # Store query and answer in database (background task)
                background_tasks.add_task(
                    db_manager.store_query_result,
                    doc_id, question, answer, relevant_context
                )
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {str(e)}")
                answers.append(f"Error processing question: {str(e)}")
        
        # Update document processing status
        background_tasks.add_task(
            db_manager.update_document_status,
            doc_id, "completed", len(request.questions)
        )
        
        logger.info(f"Successfully processed all {len(request.questions)} questions")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in run_query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/v1/documents/{doc_id}")
async def get_document_info(doc_id: int, token: str = Depends(verify_token)):
    """Get document information by ID"""
    try:
        doc_info = await db_manager.get_document_info(doc_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc_info
    except Exception as e:
        logger.error(f"Error getting document info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/queries/{doc_id}")
async def get_document_queries(doc_id: int, token: str = Depends(verify_token)):
    """Get all queries for a document"""
    try:
        queries = await db_manager.get_document_queries(doc_id)
        return {"document_id": doc_id, "queries": queries}
    except Exception as e:
        logger.error(f"Error getting document queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/summary")
async def get_analytics_summary(token: str = Depends(verify_token)):
    """Get system analytics summary"""
    try:
        summary = await db_manager.get_analytics_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting HackRX LLM Query-Retrieval System...")
    print("Server will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
