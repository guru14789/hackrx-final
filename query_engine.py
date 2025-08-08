from typing import List, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)

class QueryEngine:
    """Handles query processing and response generation"""
    
    def __init__(self):
        self.query_patterns = {
            'coverage': r'(cover|coverage|covered|include|included)',
            'waiting_period': r'(waiting period|wait|waiting time)',
            'grace_period': r'(grace period|grace time)',
            'conditions': r'(condition|conditions|requirement|requirements)',
            'exclusions': r'(exclusion|exclusions|not covered|excluded)',
            'benefits': r'(benefit|benefits|advantage|advantages)',
            'limits': r'(limit|limits|limitation|limitations|cap|maximum)',
            'definitions': r'(define|definition|means|meaning)'
        }
    
    def classify_query(self, query: str) -> List[str]:
        """Classify query type based on patterns"""
        query_lower = query.lower()
        classifications = []
        
        for category, pattern in self.query_patterns.items():
            if re.search(pattern, query_lower):
                classifications.append(category)
        
        return classifications if classifications else ['general']
    
    def extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query"""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that',
            'what', 'how', 'when', 'where', 'why', 'does', 'do', 'can', 'will'
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms
    
    def generate_search_queries(self, original_query: str) -> List[str]:
        """Generate multiple search queries for better retrieval"""
        queries = [original_query]
        
        # Extract key terms
        key_terms = self.extract_key_terms(original_query)
        
        # Create variations
        if len(key_terms) >= 2:
            # Combine key terms
            queries.append(' '.join(key_terms[:3]))
            
            # Create specific searches
            for term in key_terms[:2]:
                queries.append(f"policy {term}")
                queries.append(f"coverage {term}")
        
        return queries
    
    def rank_responses(self, responses: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank responses based on relevance to query"""
        query_terms = set(self.extract_key_terms(query))
        
        for response in responses:
            content_terms = set(self.extract_key_terms(response.get('content', '')))
            
            # Calculate term overlap
            overlap = len(query_terms.intersection(content_terms))
            total_terms = len(query_terms.union(content_terms))
            
            # Calculate relevance score
            relevance_score = overlap / total_terms if total_terms > 0 else 0
            response['relevance_score'] = relevance_score
        
        # Sort by relevance score
        return sorted(responses, key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    def extract_specific_information(self, content: str, query_type: str) -> Dict[str, Any]:
        """Extract specific information based on query type"""
        content_lower = content.lower()
        
        if query_type == 'waiting_period':
            # Look for time periods
            time_patterns = [
                r'(\d+)\s*(month|months|year|years|day|days)',
                r'(thirty|sixty|ninety|twelve|twenty-four|thirty-six)\s*(month|months|day|days)',
                r'(\d+)\s*-\s*(month|months|year|years|day|days)'
            ]
            
            for pattern in time_patterns:
                matches = re.findall(pattern, content_lower)
                if matches:
                    return {'time_periods': matches, 'type': 'waiting_period'}
        
        elif query_type == 'coverage':
            # Look for coverage statements
            coverage_patterns = [
                r'(covered|includes|covers)\s+([^.]+)',
                r'(policy covers|coverage includes)\s+([^.]+)'
            ]
            
            for pattern in coverage_patterns:
                matches = re.findall(pattern, content_lower)
                if matches:
                    return {'coverage_items': matches, 'type': 'coverage'}
        
        elif query_type == 'limits':
            # Look for numerical limits
            limit_patterns = [
                r'(\d+%)\s+of\s+([^.]+)',
                r'maximum\s+of\s+([^.]+)',
                r'limited\s+to\s+([^.]+)'
            ]
            
            for pattern in limit_patterns:
                matches = re.findall(pattern, content_lower)
                if matches:
                    return {'limits': matches, 'type': 'limits'}
        
        return {'type': 'general', 'content': content}
    
    def format_response(self, query: str, relevant_content: List[Dict[str, Any]]) -> str:
        """Format the final response based on query and content"""
        if not relevant_content:
            return "I couldn't find specific information to answer your query in the provided document."
        
        # Classify query
        query_types = self.classify_query(query)
        
        # Get the most relevant content
        best_content = relevant_content[0]['content']
        
        # Extract specific information if possible
        for query_type in query_types:
            specific_info = self.extract_specific_information(best_content, query_type)
            if specific_info['type'] != 'general':
                # Format based on extracted information
                if query_type == 'waiting_period' and 'time_periods' in specific_info:
                    periods = specific_info['time_periods']
                    return f"Based on the policy document, the waiting period is {periods[0][0]} {periods[0][1]}."
                
                elif query_type == 'coverage' and 'coverage_items' in specific_info:
                    items = specific_info['coverage_items']
                    return f"Yes, the policy covers: {items[0][1]}."
        
        # Default formatting - return the most relevant content
        return best_content
