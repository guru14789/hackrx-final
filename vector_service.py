import pinecone
from pinecone import Pinecone, ServerlessSpec
import openai
from typing import List, Dict, Any, Tuple
import hashlib
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)

class PineconeService:
    """Pinecone vector database service for semantic search"""
    
    def __init__(self):
        self.pc = None
        self.index = None
        self.index_name = "hackrx-documents"
        self.dimension = 1536  # OpenAI ada-002 embedding dimension
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Configuration
        self.pinecone_api_key = "your-pinecone-api-key"  # Set your Pinecone API key
        self.openai_api_key = "your-openai-api-key"      # Set your OpenAI API key
    
    async def initialize(self):
        """Initialize Pinecone connection and index"""
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Create index if it doesn't exist
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            # Set OpenAI API key
            openai.api_key = self.openai_api_key
            
            logger.info("Pinecone service initialized successfully")
            
        except Exception as e:
            logger.error(f"Pinecone initialization error: {str(e)}")
            raise
    
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using OpenAI's text-embedding-ada-002"""
        try:
            # Run embedding creation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _create_embeddings():
                response = openai.embeddings.create(
                    model="text-embedding-ada-002",
                    input=texts
                )
                return [embedding.embedding for embedding in response.data]
            
            embeddings = await loop.run_in_executor(self.executor, _create_embeddings)
            logger.info(f"Created embeddings for {len(texts)} text chunks")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
    
    async def create_and_store_embeddings(
        self, 
        text_chunks: List[str], 
        document_id: int, 
        document_url: str
    ) -> Dict[str, Any]:
        """Create embeddings and store them in Pinecone"""
        try:
            # Create embeddings
            embeddings = await self.create_embeddings(text_chunks)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
                # Create unique ID for each chunk
                chunk_id = f"doc_{document_id}_chunk_{i}"
                
                # Create metadata
                metadata = {
                    "document_id": document_id,
                    "document_url": document_url,
                    "chunk_index": i,
                    "text": chunk[:1000],  # Limit text size in metadata
                    "chunk_hash": hashlib.md5(chunk.encode()).hexdigest()
                }
                
                vectors.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upsert vectors to Pinecone
            loop = asyncio.get_event_loop()
            
            def _upsert_vectors():
                return self.index.upsert(vectors=vectors)
            
            upsert_response = await loop.run_in_executor(self.executor, _upsert_vectors)
            
            logger.info(f"Stored {len(vectors)} vectors in Pinecone for document {document_id}")
            
            return {
                "vectors_stored": len(vectors),
                "upsert_response": upsert_response
            }
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            raise
    
    async def search_similar_content(
        self, 
        query: str, 
        top_k: int = 5,
        document_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar content using semantic similarity"""
        try:
            # Create query embedding
            query_embeddings = await self.create_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Prepare filter if document_id is specified
            filter_dict = {}
            if document_id:
                filter_dict["document_id"] = document_id
            
            # Search in Pinecone
            loop = asyncio.get_event_loop()
            
            def _search():
                return self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict if filter_dict else None
                )
            
            search_response = await loop.run_in_executor(self.executor, _search)
            
            # Format results
            results = []
            for match in search_response.matches:
                results.append({
                    "id": match.id,
                    "score": float(match.score),
                    "text": match.metadata.get("text", ""),
                    "document_id": match.metadata.get("document_id"),
                    "chunk_index": match.metadata.get("chunk_index"),
                    "document_url": match.metadata.get("document_url")
                })
            
            logger.info(f"Found {len(results)} similar content chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar content: {str(e)}")
            return []
    
    async def delete_document_vectors(self, document_id: int):
        """Delete all vectors for a specific document"""
        try:
            # Query to get all vector IDs for the document
            filter_dict = {"document_id": document_id}
            
            loop = asyncio.get_event_loop()
            
            def _query_and_delete():
                # Get all vector IDs for the document
                query_response = self.index.query(
                    vector=[0.0] * self.dimension,  # Dummy vector
                    top_k=10000,  # Large number to get all
                    include_metadata=True,
                    filter=filter_dict
                )
                
                # Extract IDs
                ids_to_delete = [match.id for match in query_response.matches]
                
                # Delete vectors
                if ids_to_delete:
                    self.index.delete(ids=ids_to_delete)
                
                return len(ids_to_delete)
            
            deleted_count = await loop.run_in_executor(self.executor, _query_and_delete)
            logger.info(f"Deleted {deleted_count} vectors for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error deleting document vectors: {str(e)}")
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        try:
            loop = asyncio.get_event_loop()
            
            def _get_stats():
                return self.index.describe_index_stats()
            
            stats = await loop.run_in_executor(self.executor, _get_stats)
            return dict(stats)
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}
