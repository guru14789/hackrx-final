import asyncpg
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from models import DocumentMetadata

logger = logging.getLogger(__name__)

class DatabaseManager:
    """PostgreSQL database manager for storing document metadata and query results"""
    
    def __init__(self):
        self.pool = None
        # Database connection parameters
        self.db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "hackrx_db",
            "user": "hackrx_user",
            "password": "hackrx_password"
        }
    
    async def initialize(self):
        """Initialize database connection pool and create tables"""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(**self.db_config, min_size=5, max_size=20)
            
            # Create tables
            await self.create_tables()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise
    
    async def create_tables(self):
        """Create necessary database tables"""
        async with self.pool.acquire() as conn:
            # Documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    url TEXT NOT NULL,
                    content_type VARCHAR(100),
                    page_count INTEGER,
                    file_size BIGINT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'processing',
                    total_queries INTEGER DEFAULT 0,
                    metadata JSONB
                )
            """)
            
            # Queries table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id),
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    context_used JSONB,
                    confidence_score FLOAT,
                    processing_time_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);
                CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
                CREATE INDEX IF NOT EXISTS idx_queries_document_id ON queries(document_id);
                CREATE INDEX IF NOT EXISTS idx_queries_created_at ON queries(created_at);
            """)
            
            logger.info("Database tables created successfully")
    
    async def store_document_metadata(self, doc_metadata: DocumentMetadata) -> int:
        """Store document metadata and return document ID"""
        async with self.pool.acquire() as conn:
            doc_id = await conn.fetchval("""
                INSERT INTO documents (url, content_type, page_count, file_size, metadata)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """, 
            doc_metadata.url,
            doc_metadata.content_type,
            doc_metadata.page_count,
            doc_metadata.file_size,
            json.dumps(doc_metadata.metadata) if doc_metadata.metadata else None
            )
            
            logger.info(f"Stored document metadata with ID: {doc_id}")
            return doc_id
    
    async def store_query_result(
        self, 
        document_id: int, 
        question: str, 
        answer: str, 
        context: List[Dict[str, Any]],
        confidence_score: float = 0.0,
        processing_time_ms: int = 0
    ):
        """Store query result in database"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO queries (document_id, question, answer, context_used, confidence_score, processing_time_ms)
                VALUES ($1, $2, $3, $4, $5, $6)
            """,
            document_id,
            question,
            answer,
            json.dumps(context),
            confidence_score,
            processing_time_ms
            )
            
            logger.info(f"Stored query result for document {document_id}")
    
    async def update_document_status(self, document_id: int, status: str, total_queries: int):
        """Update document processing status"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE documents 
                SET status = $1, total_queries = $2
                WHERE id = $3
            """, status, total_queries, document_id)
    
    async def get_document_info(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get document information by ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, url, content_type, page_count, file_size, 
                       processed_at, status, total_queries, metadata
                FROM documents 
                WHERE id = $1
            """, document_id)
            
            if row:
                return dict(row)
            return None
    
    async def get_document_queries(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all queries for a document"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, question, answer, confidence_score, 
                       processing_time_ms, created_at
                FROM queries 
                WHERE document_id = $1
                ORDER BY created_at DESC
            """, document_id)
            
            return [dict(row) for row in rows]
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get system analytics summary"""
        async with self.pool.acquire() as conn:
            # Total documents processed
            total_docs = await conn.fetchval("SELECT COUNT(*) FROM documents")
            
            # Total queries processed
            total_queries = await conn.fetchval("SELECT COUNT(*) FROM queries")
            
            # Average processing time
            avg_processing_time = await conn.fetchval("""
                SELECT AVG(processing_time_ms) FROM queries 
                WHERE processing_time_ms > 0
            """)
            
            # Documents by status
            status_counts = await conn.fetch("""
                SELECT status, COUNT(*) as count 
                FROM documents 
                GROUP BY status
            """)
            
            # Recent activity (last 24 hours)
            recent_docs = await conn.fetchval("""
                SELECT COUNT(*) FROM documents 
                WHERE processed_at > NOW() - INTERVAL '24 hours'
            """)
            
            recent_queries = await conn.fetchval("""
                SELECT COUNT(*) FROM queries 
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            
            return {
                "total_documents": total_docs,
                "total_queries": total_queries,
                "average_processing_time_ms": float(avg_processing_time) if avg_processing_time else 0,
                "documents_by_status": {row['status']: row['count'] for row in status_counts},
                "recent_activity": {
                    "documents_24h": recent_docs,
                    "queries_24h": recent_queries
                }
            }
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
