import re
import json
from typing import Dict, Any, List
from urllib.parse import urlparse

def validate_url(url: str) -> bool:
    """Validate if the provided string is a valid URL"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def format_response(response_data: Dict[str, Any]) -> str:
    """Format API response for display"""
    if 'answers' in response_data:
        formatted = []
        for i, answer in enumerate(response_data['answers'], 1):
            formatted.append(f"{i}. {answer}")
        return "\n\n".join(formatted)
    elif 'error' in response_data:
        return f"Error: {response_data['error']}"
    else:
        return json.dumps(response_data, indent=2)

def extract_key_phrases(text: str) -> List[str]:
    """Extract key phrases from text"""
    # Simple key phrase extraction
    phrases = []
    
    # Look for phrases in quotes
    quoted_phrases = re.findall(r'"([^"]*)"', text)
    phrases.extend(quoted_phrases)
    
    # Look for capitalized terms (likely important concepts)
    capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    phrases.extend(capitalized_terms)
    
    # Remove duplicates and short phrases
    unique_phrases = list(set([p for p in phrases if len(p) > 3]))
    
    return unique_phrases[:10]  # Return top 10

def calculate_confidence_score(answer: str, query: str) -> float:
    """Calculate confidence score for an answer"""
    # Simple confidence calculation based on answer length and keyword overlap
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())
    
    # Calculate word overlap
    overlap = len(query_words.intersection(answer_words))
    total_query_words = len(query_words)
    
    # Base confidence on overlap and answer length
    overlap_score = overlap / total_query_words if total_query_words > 0 else 0
    length_score = min(1.0, len(answer.split()) / 50)  # Normalize by expected answer length
    
    # Combine scores
    confidence = (overlap_score * 0.6 + length_score * 0.4) * 100
    
    return min(95, max(60, confidence))  # Clamp between 60-95%

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized[:100]  # Limit length

def chunk_text_by_sentences(text: str, max_chunk_size: int = 1000) -> List[str]:
    """Chunk text by sentences while respecting size limits"""
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_numerical_values(text: str) -> List[Dict[str, Any]]:
    """Extract numerical values and their context from text"""
    patterns = [
        (r'(\d+)\s*(percent|%)', 'percentage'),
        (r'(\d+)\s*(days?|months?|years?)', 'time_period'),
        (r'(\d+(?:,\d{3})*(?:\.\d{2})?)', 'number'),
        (r'(\d+)\s*(lakhs?|crores?)', 'currency_indian')
    ]
    
    values = []
    for pattern, value_type in patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            values.append({
                'value': match.group(1),
                'type': value_type,
                'context': text[max(0, match.start()-50):match.end()+50].strip()
            })
    
    return values
