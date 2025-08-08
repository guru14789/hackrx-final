from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime

class QueryRequest(BaseModel):
    """Request model for document queries"""
    documents: str = Field(..., description="URL of the document to process")
    questions: List[str] = Field(..., min_items=1, max_items=20, description="List of questions to answer")

class QueryResponse(BaseModel):
    """Response model for document queries"""
    answers: List[str] = Field(..., description="List of answers corresponding to the questions")

class DocumentMetadata(BaseModel):
    """Document metadata model"""
    url: str
    content_type: Optional[str] = None
    page_count: Optional[int] = 0
    file_size: Optional[int] = 0
    processed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryResult(BaseModel):
    """Query result model"""
    id: Optional[int] = None
    document_id: int
    question: str
    answer: str
    confidence_score: Optional[float] = 0.0
    processing_time_ms: Optional[int] = 0
    created_at: Optional[datetime] = None

class DocumentInfo(BaseModel):
    """Document information model"""
    id: int
    url: str
    content_type: Optional[str]
    page_count: Optional[int]
    file_size: Optional[int]
    processed_at: datetime
    status: str
    total_queries: int
    metadata: Optional[Dict[str, Any]]

class AnalyticsSummary(BaseModel):
    """Analytics summary model"""
    total_documents: int
    total_queries: int
    average_processing_time_ms: float
    documents_by_status: Dict[str, int]
    recent_activity: Dict[str, int]
