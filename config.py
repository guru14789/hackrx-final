import os
from typing import Dict, Any

class Config:
    """Configuration management"""
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://hackrx_user:hackrx_password@localhost:5432/hackrx_db")
    
    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-documents")
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # API Configuration
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
    API_TOKEN = os.getenv("API_TOKEN", "9a653094793aedeae46f194aa755e2bb17f297f5209b7f99c1ced3671779d95d")
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", 30))
    API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", 3))
    
    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", 2000))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Document Processing Configuration
    SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", ".pdf,.docx,.doc,.txt").split(",")
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024))  # 50MB
    MAX_PAGES = int(os.getenv("MAX_PAGES", 1000))
    
    # Query Configuration
    MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", 20))
    MIN_QUESTION_LENGTH = int(os.getenv("MIN_QUESTION_LENGTH", 5))
    MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", 500))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
    
    # Streamlit Configuration
    STREAMLIT_PAGE_TITLE = os.getenv("STREAMLIT_PAGE_TITLE", "LLM Query-Retrieval System")
    STREAMLIT_PAGE_ICON = os.getenv("STREAMLIT_PAGE_ICON", "🔍")
    STREAMLIT_LAYOUT = os.getenv("STREAMLIT_LAYOUT", "wide")
    STREAMLIT_INITIAL_SIDEBAR_STATE = os.getenv("STREAMLIT_INITIAL_SIDEBAR_STATE", "expanded")
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", 5432)),
            "database": os.getenv("DB_NAME", "hackrx_db"),
            "user": os.getenv("DB_USER", "hackrx_user"),
            "password": os.getenv("DB_PASSWORD", "hackrx_password")
        }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get complete configuration"""
        return {
            "database": cls.get_database_config(),
            "pinecone": {
                "api_key": cls.PINECONE_API_KEY,
                "environment": cls.PINECONE_ENVIRONMENT,
                "index_name": cls.PINECONE_INDEX_NAME
            },
            "openai": {
                "api_key": cls.OPENAI_API_KEY,
                "model": cls.OPENAI_MODEL
            },
            "api": {
                "base_url": cls.API_BASE_URL,
                "token": cls.API_TOKEN,
                "timeout": cls.API_TIMEOUT,
                "max_retries": cls.API_MAX_RETRIES
            },
            "model": {
                "embedding_model": cls.EMBEDDING_MODEL,
                "embedding_dimension": cls.EMBEDDING_DIMENSION,
                "max_context_length": cls.MAX_CONTEXT_LENGTH,
                "chunk_size": cls.CHUNK_SIZE,
                "chunk_overlap": cls.CHUNK_OVERLAP
            },
            "document": {
                "supported_formats": cls.SUPPORTED_FORMATS,
                "max_file_size": cls.MAX_FILE_SIZE,
                "max_pages": cls.MAX_PAGES
            },
            "query": {
                "max_questions": cls.MAX_QUESTIONS,
                "min_question_length": cls.MIN_QUESTION_LENGTH,
                "max_question_length": cls.MAX_QUESTION_LENGTH,
                "similarity_threshold": cls.SIMILARITY_THRESHOLD
            },
            "streamlit": {
                "page_title": cls.STREAMLIT_PAGE_TITLE,
                "page_icon": cls.STREAMLIT_PAGE_ICON,
                "layout": cls.STREAMLIT_LAYOUT,
                "initial_sidebar_state": cls.STREAMLIT_INITIAL_SIDEBAR_STATE
            }
        }
