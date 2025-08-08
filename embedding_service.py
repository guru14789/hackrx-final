import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import pickle
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Handles document embeddings and vector search using FAISS"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.index = None
        self.documents = []
        self.embeddings = []
        
    def create_embeddings(self, content: Dict[str, Any]) -> np.ndarray:
        """Create embeddings for document content"""
        try:
            # Extract text chunks
            if 'text_content' in content:
                texts = [item['content'] for item in content['text_content']]
            else:
                texts = [content.get('full_text', '')]
            
            # Create embeddings
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            logger.info(f"Created embeddings for {len(texts)} text chunks")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise Exception(f"Failed to create embeddings: {str(e)}")
    
    def store_embeddings(self, embeddings: np.ndarray, content: Dict[str, Any]):
        """Store embeddings in FAISS index"""
        try:
            # Initialize FAISS index if not exists
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings)
            
            # Store document content
            if 'text_content' in content:
                self.documents.extend(content['text_content'])
            else:
                self.documents.append({'content': content.get('full_text', '')})
            
            self.embeddings.extend(embeddings.tolist())
            
            logger.info(f"Stored {len(embeddings)} embeddings in FAISS index")
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            raise Exception(f"Failed to store embeddings: {str(e)}")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using semantic similarity"""
        try:
            if self.index is None or len(self.documents) == 0:
                return []
            
            # Create query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append({
                        'content': self.documents[idx]['content'],
                        'similarity_score': float(score),
                        'document_index': int(idx)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            return []
    
    def get_relevant_context(self, query: str, max_context_length: int = 2000) -> str:
        """Get relevant context for a query"""
        similar_docs = self.search_similar(query, top_k=3)
        
        context_parts = []
        current_length = 0
        
        for doc in similar_docs:
            content = doc['content']
            if current_length + len(content) <= max_context_length:
                context_parts.append(content)
                current_length += len(content)
            else:
                # Add partial content if it fits
                remaining_space = max_context_length - current_length
                if remaining_space > 100:  # Only add if meaningful space remains
                    context_parts.append(content[:remaining_space] + "...")
                break
        
        return "\n\n".join(context_parts)
    
    def save_index(self, filepath: str):
        """Save FAISS index and documents to file"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, f"{filepath}.faiss")
                
                with open(f"{filepath}.pkl", 'wb') as f:
                    pickle.dump({
                        'documents': self.documents,
                        'embeddings': self.embeddings
                    }, f)
                
                logger.info(f"Saved index to {filepath}")
                
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def load_index(self, filepath: str):
        """Load FAISS index and documents from file"""
        try:
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.embeddings = data['embeddings']
            
            logger.info(f"Loaded index from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
