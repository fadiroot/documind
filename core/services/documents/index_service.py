"""Azure AI Search index management service."""
from typing import Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SimpleField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    HnswParameters,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField
)
from app.config import settings
from core.utils.logger import logger


class IndexService:
    """Service for managing Azure AI Search index."""
    
    def __init__(self):
        self.endpoint = settings.AZURE_AI_SEARCH_ENDPOINT or settings.AZURE_SEARCH_ENDPOINT
        self.api_key = settings.AZURE_AI_SEARCH_API_KEY or settings.AZURE_SEARCH_KEY
        self.index_name = settings.AZURE_AI_SEARCH_INDEX_NAME or settings.AZURE_SEARCH_INDEX_NAME
        self.index_client: Optional[SearchIndexClient] = None
        
        if self.endpoint and self.api_key:
            credential = AzureKeyCredential(self.api_key)
            self.index_client = SearchIndexClient(
                endpoint=self.endpoint,
                credential=credential
            )
        else:
            logger.warning("Azure AI Search credentials not configured for index management")
    
    def create_index(self, vector_dimension: int = 3072) -> bool:
        """
        Create or update the search index with the specified schema.
        
        Args:
            vector_dimension: Dimension of the embedding vectors (default: 3072 for text-embedding-3-large)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.index_client:
            logger.error("Index client not available")
            return False
        
        try:
            fields = [
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    searchable=False,
                    filterable=False,
                    sortable=False,
                    facetable=False
                ),
                SearchField(
                    name="content",
                    type=SearchFieldDataType.String,
                    searchable=True,
                    filterable=False,
                    sortable=False,
                    facetable=False,
                    analyzer_name="ar.microsoft"
                ),
                SimpleField(
                    name="document_name",
                    type=SearchFieldDataType.String,
                    searchable=False,
                    filterable=True,
                    sortable=True,
                    facetable=True
                ),
                SimpleField(
                    name="page_number",
                    type=SearchFieldDataType.Int32,
                    searchable=False,
                    filterable=True,
                    sortable=True,
                    facetable=True
                ),
                SimpleField(
                    name="chunk_index",
                    type=SearchFieldDataType.Int32,
                    searchable=False,
                    filterable=True,
                    sortable=True,
                    facetable=False
                ),
                SimpleField(
                    name="token_count",
                    type=SearchFieldDataType.Int32,
                    searchable=False,
                    filterable=True,
                    sortable=True,
                    facetable=False
                ),
                SearchField(
                    name="contentVector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=vector_dimension,
                    vector_search_profile_name="vector-profile"
                )
            ]
            
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="vector-profile",
                        algorithm_configuration_name="hnsw-config"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="hnsw-config",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters=HnswParameters(
                            m=4,
                            ef_construction=400,
                            ef_search=500,
                            metric="cosine"
                        )
                    )
                ]
            )
            
            # Create semantic configuration
            semantic_config = SemanticConfiguration(
                name="semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="content"),
                    content_fields=[
                        SemanticField(field_name="content")
                    ],
                    keywords_fields=[
                        SemanticField(field_name="document_name")
                    ]
                )
            )
            
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_configurations=[semantic_config]
            )
            
            self.index_client.create_or_update_index(index)
            logger.info(f"Index '{self.index_name}' created/updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return False
    
    def delete_index(self) -> bool:
        """
        Delete the search index.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.index_client:
            logger.error("Index client not available")
            return False
        
        try:
            self.index_client.delete_index(self.index_name)
            logger.info(f"Index '{self.index_name}' deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            return False
    
    def get_index(self) -> Optional[SearchIndex]:
        """
        Get the current index schema.
        
        Returns:
            SearchIndex object or None if not found
        """
        if not self.index_client:
            logger.error("Index client not available")
            return None
        
        try:
            return self.index_client.get_index(self.index_name)
        except Exception as e:
            logger.error(f"Error getting index: {str(e)}")
            return None
    
    def index_exists(self) -> bool:
        """
        Check if the index exists.
        
        Returns:
            True if index exists, False otherwise
        """
        if not self.index_client:
            return False
        
        try:
            self.index_client.get_index(self.index_name)
            return True
        except Exception:
            return False
