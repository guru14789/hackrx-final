"""
Simple FastAPI server to simulate the hackrx API endpoint
Run this if you want to test the original API integration
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HackRX API Server", version="1.0.0")

# Expected token
EXPECTED_TOKEN = "9a653094793aedeae46f194aa755e2bb17f297f5209b7f99c1ced3671779d95d"

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

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

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, token: str = Depends(verify_token)):
    """
    Process documents and answer questions
    This is a mock implementation - replace with actual logic
    """
    try:
        logger.info(f"Processing {len(request.questions)} questions for document: {request.documents}")
        
        # Mock responses - replace with actual document processing
        mock_answers = [
            "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
            "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
            "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months.",
            "The policy has a specific waiting period of two (2) years for cataract surgery.",
            "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person."
        ]
        
        # Return mock answers (truncate if more questions than mock answers)
        answers = mock_answers[:len(request.questions)]
        
        # If more questions than mock answers, add generic responses
        while len(answers) < len(request.questions):
            answers.append("Information not found in the provided document.")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API server is running"}

if __name__ == "__main__":
    print("Starting HackRX API Server...")
    print("Server will be available at: http://localhost:8000")
    print("API endpoint: http://localhost:8000/api/v1/hackrx/run")
    print("Health check: http://localhost:8000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
