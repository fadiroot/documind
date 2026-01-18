"""Retrieval service for RAG operations."""
from typing import List, Dict, Any, Optional
from core.services.retrieval.embedding_service import EmbeddingService
from core.services.retrieval.vectorstore_service import VectorStoreService
from core.utils.logger import logger


class RetrievalService:
    """Service for retrieving relevant documents for RAG."""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vectorstore_service = VectorStoreService()
    
    def retrieve(self, query: str, top_k: int = 5, filters: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters
        
        Returns:
            List of relevant documents with metadata
        """
        # Create query embedding
        query_embedding = self.embedding_service.create_embedding(query)
        if not query_embedding:
            logger.error("Failed to create query embedding")
            return []
        
        # Search vector store
        results = self.vectorstore_service.search(
            query_vector=query_embedding,
            top_k=top_k,
            filters=filters
        )
        
        # Format results with consistent structure
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.get("id"),
                "content": result.get("content", ""),
                "document_name": result.get("document_name", ""),
                "page_number": result.get("page_number"),
                "chunk_index": result.get("chunk_index"),
                "score": result.get("@search.score", 0.0)
            })
        
        return formatted_results
