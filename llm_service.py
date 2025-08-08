import openai
from typing import List, Dict, Any, Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import time

logger = logging.getLogger(__name__)

class GPTService:
    """GPT-4 service for generating intelligent answers"""
    
    def __init__(self):
        self.client = None
        self.model = "gpt-4"
        self.max_tokens = 1000
        self.temperature = 0.1  # Low temperature for consistent, factual responses
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Set your OpenAI API key
        self.api_key = "your-openai-api-key"
    
    async def test_connection(self):
        """Test GPT-4 connection"""
        try:
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
            
            # Test with a simple request
            loop = asyncio.get_event_loop()
            
            def _test():
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Hello, are you working?"}],
                    max_tokens=10
                )
                return response.choices[0].message.content
            
            test_response = await loop.run_in_executor(self.executor, _test)
            logger.info(f"GPT-4 connection test successful: {test_response}")
            
        except Exception as e:
            logger.error(f"GPT-4 connection test failed: {str(e)}")
            raise
    
    async def generate_answer(
        self, 
        question: str, 
        context: List[Dict[str, Any]], 
        document_url: str
    ) -> str:
        """Generate an intelligent answer using GPT-4"""
        try:
            start_time = time.time()
            
            # Prepare context text
            context_text = self._prepare_context(context)
            
            # Create system prompt
            system_prompt = self._create_system_prompt()
            
            # Create user prompt
            user_prompt = self._create_user_prompt(question, context_text, document_url)
            
            # Generate response using GPT-4
            loop = asyncio.get_event_loop()
            
            def _generate():
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    presence_penalty=0.1,
                    frequency_penalty=0.1
                )
                return response.choices[0].message.content
            
            answer = await loop.run_in_executor(self.executor, _generate)
            
            processing_time = int((time.time() - start_time) * 1000)
            logger.info(f"Generated answer in {processing_time}ms")
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def _prepare_context(self, context: List[Dict[str, Any]]) -> str:
        """Prepare context text from search results"""
        if not context:
            return "No relevant context found in the document."
        
        context_parts = []
        for i, item in enumerate(context[:5], 1):  # Limit to top 5 results
            text = item.get("text", "").strip()
            score = item.get("score", 0)
            
            if text:
                context_parts.append(f"Context {i} (relevance: {score:.3f}):\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for GPT-4"""
        return """You are an expert document analyst specializing in insurance, legal, HR, and compliance domains. Your task is to provide accurate, detailed answers based on the provided document context.

Guidelines:
1. Answer questions directly and accurately based ONLY on the provided context
2. If information is not available in the context, clearly state this
3. For insurance documents, pay attention to coverage details, waiting periods, exclusions, and conditions
4. For legal documents, focus on clauses, terms, obligations, and rights
5. For HR documents, highlight policies, procedures, and employee rights
6. For compliance documents, emphasize requirements, regulations, and obligations
7. Always cite specific information from the context when possible
8. If asked about time periods, amounts, or percentages, provide exact figures from the document
9. Structure your response clearly with relevant details
10. If the context contains conflicting information, mention this explicitly

Remember: Be precise, factual, and helpful. Do not make assumptions beyond what's stated in the context."""
    
    def _create_user_prompt(self, question: str, context: str, document_url: str) -> str:
        """Create user prompt with question and context"""
        return f"""Document URL: {document_url}

Question: {question}

Relevant Context from Document:
{context}

Please provide a comprehensive answer to the question based on the context provided above. If the context doesn't contain sufficient information to answer the question, please state this clearly."""
    
    async def generate_summary(self, text_chunks: List[str], document_type: str = "document") -> str:
        """Generate a summary of the document"""
        try:
            # Combine text chunks (limit total length)
            combined_text = " ".join(text_chunks)[:8000]  # Limit to ~8k characters
            
            system_prompt = f"""You are an expert at summarizing {document_type}s. Create a concise but comprehensive summary that captures the key points, important details, and main themes."""
            
            user_prompt = f"Please provide a detailed summary of this {document_type}:\n\n{combined_text}"
            
            loop = asyncio.get_event_loop()
            
            def _generate_summary():
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=500,
                    temperature=0.2
                )
                return response.choices[0].message.content
            
            summary = await loop.run_in_executor(self.executor, _generate_summary)
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    async def extract_key_information(self, text_chunks: List[str], domain: str) -> Dict[str, Any]:
        """Extract key information based on domain"""
        try:
            combined_text = " ".join(text_chunks)[:8000]
            
            domain_prompts = {
                "insurance": "Extract key insurance information: coverage details, premiums, deductibles, exclusions, waiting periods, claim procedures, and policy terms.",
                "legal": "Extract key legal information: parties involved, obligations, rights, termination clauses, liability, dispute resolution, and governing law.",
                "hr": "Extract key HR information: job roles, responsibilities, compensation, benefits, leave policies, performance criteria, and termination procedures.",
                "compliance": "Extract key compliance information: regulatory requirements, audit procedures, reporting obligations, penalties, and compliance standards."
            }
            
            system_prompt = f"You are an expert in {domain} documents. {domain_prompts.get(domain, 'Extract key information from the document.')}"
            
            user_prompt = f"Extract and structure the key information from this {domain} document:\n\n{combined_text}"
            
            loop = asyncio.get_event_loop()
            
            def _extract():
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=800,
                    temperature=0.1
                )
                return response.choices[0].message.content
            
            extracted_info = await loop.run_in_executor(self.executor, _extract)
            
            return {
                "domain": domain,
                "extracted_information": extracted_info.strip(),
                "extraction_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error extracting key information: {str(e)}")
            return {"error": str(e)}
