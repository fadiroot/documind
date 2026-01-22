"""Search service for querying Azure AI Search index."""
from typing import List, Optional, Dict, Any
from azure.search.documents.models import VectorizedQuery
from azure.core.exceptions import ServiceRequestError, HttpResponseError
from core.utils.azure_utils import get_search_client
from core.utils.logger import logger
from app.config import settings


class SearchService:
    """Service for searching documents in Azure AI Search index."""
    
    def __init__(self):
        self.search_client = get_search_client()
        self.index_name = settings.AZURE_AI_SEARCH_INDEX_NAME or settings.AZURE_SEARCH_INDEX_NAME
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: Optional OData filter expression
        
        Returns:
            List of search results with metadata
        """
        if not self.search_client:
            logger.error("Search client not available - check Azure AI Search configuration")
            return []
        
        try:
            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="contentVector"
            )
            
            # Build search parameters - use only fields that exist in the index
            search_params = {
                "search_text": None,
                "vector_queries": [vector_query],
                "top": top_k,
                "include_total_count": True,
                "select": [
                    "id",
                    "content",
                    "source_document",
                    "document_name",
                    "document_title",
                    "article_reference",
                    "article_number",
                    "legal_part_name",
                    "legal_chapter_name",
                    "paragraph_number",
                    "clause_number",
                    "procedure_name",
                    "procedure_step",
                    "policy_name",
                    "annex_name",
                    "rank",
                    "grade",
                    "category_class",
                    "group",
                    "cadre_classification",
                    "metadata_item_number",
                    "metadata_item_type",
                    "metadata_item_title",
                    "metadata_section_title",
                    "metadata_resource_path",
                    "category",
                    "target_audience",
                    "keywords",
                    "page_number",
                    "chunk_index",
                    "token_count"
                ]
            }
            
            if filters:
                search_params["filter"] = filters
            
            results = self.search_client.search(**search_params)
            
            # Convert Azure Search results to dictionaries
            formatted_results = []
            
            try:
                for result in results:
                    result_dict = {}
                    
                    # Extract data from Azure Search result
                    if isinstance(result, dict):
                        result_dict = result.copy()
                    else:
                        # Try attribute access
                        result_dict['id'] = getattr(result, 'id', None)
                        result_dict['content'] = getattr(result, 'content', '')
                        
                        # Get score
                        score = getattr(result, '@search.score', None) or getattr(result, 'score', None)
                        result_dict['@search.score'] = score if score is not None else 0.0
                        
                        # Extract metadata fields - use actual index field names
                        metadata_fields = [
                            'source_document', 'document_name', 'document_title',
                            'article_reference', 'article_number',
                            'legal_part_name', 'legal_chapter_name',
                            'paragraph_number', 'clause_number',
                            'procedure_name', 'procedure_step', 'policy_name', 'annex_name',
                            'rank', 'grade', 'category_class', 'group', 'cadre_classification',
                            'metadata_item_number', 'metadata_item_type', 'metadata_item_title',
                            'metadata_section_title', 'metadata_resource_path',
                            'category', 'target_audience', 'keywords',
                            'page_number', 'chunk_index', 'token_count'
                        ]
                        
                        for field in metadata_fields:
                            if hasattr(result, '__getitem__'):
                                try:
                                    value = result.get(field) if hasattr(result, 'get') else result[field]
                                    result_dict[field] = value
                                except (KeyError, TypeError):
                                    result_dict[field] = getattr(result, field, None)
                            else:
                                result_dict[field] = getattr(result, field, None)
                    
                    if result_dict.get('content'):
                        formatted_results.append(result_dict)
                        
            except Exception as e:
                logger.error(f"Error processing search results: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            
            return formatted_results
        except ServiceRequestError as e:
            # Network/DNS errors
            error_msg = str(e)
            if "Failed to resolve" in error_msg or "name resolution" in error_msg.lower():
                logger.error(f"Azure AI Search endpoint cannot be resolved. Check network connectivity and endpoint URL: {self.index_name}")
                logger.error(f"Endpoint: {settings.AZURE_AI_SEARCH_ENDPOINT or settings.AZURE_SEARCH_ENDPOINT}")
            else:
                logger.error(f"Azure AI Search network error: {error_msg}")
            return []
        except HttpResponseError as e:
            # HTTP errors (auth, bad request, etc.)
            logger.error(f"Azure AI Search HTTP error: {e.status_code} - {e.message}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching Azure AI Search: {str(e)}")
            return []
    
    def search_by_filter(self, filter_expression: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Search documents by filter expression.
        
        Args:
            filter_expression: OData filter expression
            top_k: Maximum number of results
        
        Returns:
            List of matching documents
        """
        return self.search(query_vector=[0.0] * 1536, top_k=top_k, filters=filter_expression)
