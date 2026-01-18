"""Vector store service for Azure AI Search."""
from typing import List, Optional, Dict, Any
from azure.search.documents.models import VectorizedQuery
from core.utils.azure_utils import get_search_client
from core.utils.logger import logger
from app.config import settings


class VectorStoreService:
    """Service for managing vector store operations with Azure AI Search."""
    
    def __init__(self):
        self.search_client = get_search_client()
        self.index_name = settings.AZURE_AI_SEARCH_INDEX_NAME or settings.AZURE_SEARCH_INDEX_NAME
    
    def upload_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Upload documents to the vector store.
        
        Args:
            documents: List of document dictionaries with id, content, contentVector, etc.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.search_client:
            logger.error("Search client not available")
            return False
        
        try:
            result = self.search_client.upload_documents(documents=documents)
            logger.info(f"Uploaded {len(documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Error uploading documents: {str(e)}")
            return False
    
    def search(self, query_vector: List[float], top_k: int = 5, filters: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: Optional OData filter expression
        
        Returns:
            List of search results
        """
        if not self.search_client:
            logger.error("Search client not available")
            return []
        
        try:
            # Create vector query using VectorizedQuery
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="contentVector"
            )
            
            # Build search parameters
            search_params = {
                "search_text": None,
                "vector_queries": [vector_query],
                "top": top_k,
                "include_total_count": True,
                "select": ["id", "content", "document_name", "page_number", "chunk_index"]
            }
            
            if filters:
                search_params["filter"] = filters
            
            results = self.search_client.search(**search_params)
            
            return [dict(result) for result in results]
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            document_id: ID of the document to delete
        
        Returns:
            True if successful, False otherwise
        """
        if not self.search_client:
            logger.error("Search client not available")
            return False
        
        try:
            self.search_client.delete_documents(documents=[{"id": document_id}])
            logger.info(f"Deleted document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
