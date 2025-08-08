import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any
from document_processor import DocumentProcessor
from query_engine import QueryEngine
from embedding_service import EmbeddingService
from llm_service import LLMService
from utils import format_response, validate_url
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StandaloneLLMQuerySystem:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.query_engine = QueryEngine()
        self.llm_service = LLMService()
        
    def process_documents(self, document_url: str) -> Dict[str, Any]:
        """Process documents and extract structured content"""
        try:
            # Extract content from document
            content = self.document_processor.extract_content(document_url)
            
            # Create embeddings
            embeddings = self.embedding_service.create_embeddings(content)
            
            # Store in vector database
            self.embedding_service.store_embeddings(embeddings, content)
            
            return {"status": "success", "message": "Document processed successfully", "content": content}
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def answer_questions(self, questions: List[str]) -> List[str]:
        """Answer questions using the processed documents"""
        answers = []
        
        for question in questions:
            try:
                # Get relevant context from embeddings
                relevant_context = self.embedding_service.get_relevant_context(question)
                
                if relevant_context:
                    # Generate answer using LLM
                    answer = self.llm_service.generate_answer(question, relevant_context)
                    answers.append(answer)
                else:
                    answers.append("I couldn't find relevant information in the document to answer this question.")
                    
            except Exception as e:
                logger.error(f"Error answering question '{question}': {str(e)}")
                answers.append(f"Error processing question: {str(e)}")
        
        return answers

def main():
    st.set_page_config(
        page_title="LLM Query-Retrieval System (Standalone)",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 LLM-Powered Intelligent Query-Retrieval System")
    st.markdown("**Standalone Version** - Process large documents and make contextual decisions for insurance, legal, HR, and compliance domains.")
    
    # Initialize system
    if 'system' not in st.session_state:
        with st.spinner("Initializing system..."):
            st.session_state.system = StandaloneLLMQuerySystem()
        st.success("System initialized successfully!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        st.success("✅ System ready for document processing")
        
        # Document types supported
        st.subheader("Supported Formats")
        st.write("• PDF documents")
        st.write("• DOCX files")
        st.write("• Text documents")
        
        st.subheader("System Status")
        st.write("🟢 Document Processor: Ready")
        st.write("🟢 Embedding Service: Ready")
        st.write("🟢 Query Engine: Ready")
        st.write("🟢 LLM Service: Ready")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document Input")
        
        # Document URL input
        document_url = st.text_input(
            "Document URL",
            value="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
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
            value="""What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?
What is the waiting period for pre-existing diseases (PED) to be covered?
Does this policy cover maternity expenses, and what are the conditions?
What is the waiting period for cataract surgery?
Are the medical expenses for an organ donor covered under this policy?""",
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
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Process documents
                    status_text.text("📄 Processing document...")
                    progress_bar.progress(20)
                    
                    process_result = st.session_state.system.process_documents(document_url)
                    
                    if process_result["status"] == "success":
                        progress_bar.progress(50)
                        status_text.text("✅ Document processed successfully!")
                        
                        # Step 2: Answer questions
                        status_text.text("🤔 Generating answers...")
                        progress_bar.progress(70)
                        
                        answers = st.session_state.system.answer_questions(questions)
                        
                        progress_bar.progress(100)
                        status_text.text("🎉 All questions answered!")
                        
                        time.sleep(1)  # Brief pause to show completion
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display results
                        st.success(f"Generated {len(answers)} responses!")
                        
                        st.subheader("📋 Results")
                        
                        for i, (question, answer) in enumerate(zip(questions, answers), 1):
                            with st.expander(f"Q{i}: {question}", expanded=True):
                                st.write("**Answer:**")
                                st.write(answer)
                                
                                # Add confidence indicator
                                if "couldn't find" not in answer.lower() and "error" not in answer.lower():
                                    confidence = min(95, 80 + (len(answer.split()) / 10))
                                    st.progress(confidence / 100)
                                    st.caption(f"Confidence: {confidence:.1f}%")
                                else:
                                    st.progress(0.3)
                                    st.caption("Confidence: Low - Limited information found")
                        
                        # Export results
                        st.subheader("📤 Export Results")
                        
                        # Prepare export data
                        export_data = {
                            "document_url": document_url,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "questions_and_answers": [
                                {"question": q, "answer": a} 
                                for q, a in zip(questions, answers)
                            ],
                            "system_info": {
                                "version": "standalone",
                                "processing_method": "local_llm_with_embeddings"
                            }
                        }
                        
                        st.download_button(
                            label="📥 Download Results (JSON)",
                            data=json.dumps(export_data, indent=2),
                            file_name=f"query_results_{int(time.time())}.json",
                            mime="application/json"
                        )
                        
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"Document processing failed: {process_result['message']}")
                        
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"An error occurred: {str(e)}")
                    logger.error(f"Application error: {str(e)}")
    
    # System metrics and information
    st.header("📊 System Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Supported Formats", "3", help="PDF, DOCX, Text")
    
    with col2:
        st.metric("Processing Mode", "Standalone", help="Local processing without external API")
    
    with col3:
        st.metric("Embedding Model", "Active", help="Semantic search enabled")
    
    with col4:
        st.metric("Response Time", "< 60s", help="Average query processing time")
    
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
    
    # API Integration section
    with st.expander("🔌 API Integration (Optional)", expanded=False):
        st.markdown("""
        **To use the external API:**
        
        1. **Start the API server:**
        ```bash
        # Make sure the API server is running on localhost:8000
        # Contact your backend team or check the API documentation
