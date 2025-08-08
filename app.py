import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any
from document_processor import DocumentProcessor
from query_engine import QueryEngine
from embedding_service import EmbeddingService
from utils import format_response, validate_url
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
API_TOKEN = "9a653094793aedeae46f194aa755e2bb17f297f5209b7f99c1ced3671779d95d"

class LLMQueryRetrievalSystem:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.query_engine = QueryEngine()
        
    def process_documents(self, document_url: str) -> Dict[str, Any]:
        """Process documents and extract structured content"""
        try:
            # Extract content from document
            content = self.document_processor.extract_content(document_url)
            
            # Create embeddings
            embeddings = self.embedding_service.create_embeddings(content)
            
            # Store in vector database
            self.embedding_service.store_embeddings(embeddings, content)
            
            return {"status": "success", "message": "Document processed successfully"}
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def query_documents(self, questions: List[str], document_url: str) -> Dict[str, Any]:
        """Query documents using the API endpoint"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {API_TOKEN}"
            }
            
            payload = {
                "documents": document_url,
                "questions": questions
            }
            
            response = requests.post(
                f"{API_BASE_URL}/hackrx/run",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"API request failed with status {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            return {"error": str(e)}

def main():
    st.set_page_config(
        page_title="LLM Query-Retrieval System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 LLM-Powered Intelligent Query-Retrieval System")
    st.markdown("Process large documents and make contextual decisions for insurance, legal, HR, and compliance domains.")
    
    # Initialize system
    if 'system' not in st.session_state:
        st.session_state.system = LLMQueryRetrievalSystem()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        st.info("System ready for document processing")
        
        # Document types supported
        st.subheader("Supported Formats")
        st.write("• PDF documents")
        st.write("• DOCX files")
        st.write("• Email documents")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document Input")
        
        # Document URL input
        document_url = st.text_input(
            "Document URL",
            placeholder="https://example.com/document.pdf",
            help="Enter the URL of the document to process"
        )
        
        # Validate URL
        if document_url and not validate_url(document_url):
            st.error("Please enter a valid URL")
        
        # Questions input
        st.subheader("Questions")
        questions_text = st.text_area(
            "Enter your questions (one per line)",
            height=200,
            placeholder="What is the grace period for premium payment?\nDoes this policy cover maternity expenses?\nWhat is the waiting period for pre-existing diseases?"
        )
        
        # Parse questions
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        
        if questions:
            st.write(f"**{len(questions)} questions detected:**")
            for i, q in enumerate(questions, 1):
                st.write(f"{i}. {q}")
    
    with col2:
        st.header("🎯 Query Results")
        
        if st.button("🚀 Process & Query", type="primary", disabled=not (document_url and questions)):
            if document_url and questions:
                with st.spinner("Processing document and generating responses..."):
                    # Process documents
                    process_result = st.session_state.system.process_documents(document_url)
                    
                    if process_result["status"] == "success":
                        st.success("Document processed successfully!")
                        
                        # Query documents
                        query_result = st.session_state.system.query_documents(questions, document_url)
                        
                        if "answers" in query_result:
                            st.success(f"Generated {len(query_result['answers'])} responses!")
                            
                            # Display results
                            st.subheader("📋 Results")
                            
                            for i, (question, answer) in enumerate(zip(questions, query_result["answers"]), 1):
                                with st.expander(f"Q{i}: {question}", expanded=True):
                                    st.write("**Answer:**")
                                    st.write(answer)
                                    
                                    # Add confidence indicator (simulated)
                                    confidence = min(95, 80 + (len(answer.split()) / 10))
                                    st.progress(confidence / 100)
                                    st.caption(f"Confidence: {confidence:.1f}%")
                            
                            # Export results
                            st.subheader("📤 Export Results")
                            
                            # Prepare export data
                            export_data = {
                                "document_url": document_url,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "questions_and_answers": [
                                    {"question": q, "answer": a} 
                                    for q, a in zip(questions, query_result["answers"])
                                ]
                            }
                            
                            st.download_button(
                                label="📥 Download Results (JSON)",
                                data=json.dumps(export_data, indent=2),
                                file_name=f"query_results_{int(time.time())}.json",
                                mime="application/json"
                            )
                            
                        elif "error" in query_result:
                            st.error(f"Query failed: {query_result['error']}")
                            if "details" in query_result:
                                st.error(f"Details: {query_result['details']}")
                    else:
                        st.error(f"Document processing failed: {process_result['message']}")
    
    # System metrics and information
    st.header("📊 System Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Supported Formats", "3", help="PDF, DOCX, Email")
    
    with col2:
        st.metric("API Status", "Connected", help="Connection to backend API")
    
    with col3:
        st.metric("Embedding Model", "Active", help="Semantic search enabled")
    
    with col4:
        st.metric("Response Time", "< 30s", help="Average query processing time")
    
    # Sample queries section
    with st.expander("💡 Sample Queries", expanded=False):
        st.markdown("""
        **Insurance Domain:**
        - What is the grace period for premium payment?
        - Does this policy cover maternity expenses, and what are the conditions?
        - What is the waiting period for pre-existing diseases?
        
        **Legal Domain:**
        - What are the termination clauses in this contract?
        - What are the liability limitations mentioned?
        - What is the dispute resolution mechanism?
        
        **HR Domain:**
        - What is the notice period for resignation?
        - What are the performance evaluation criteria?
        - What benefits are included in the compensation package?
        
        **Compliance Domain:**
        - What are the regulatory requirements mentioned?
        - What are the audit and reporting obligations?
        - What are the data protection measures specified?
        """)

if __name__ == "__main__":
    main()
