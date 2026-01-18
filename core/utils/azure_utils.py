"""Azure-specific utility functions."""
from typing import Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from app.config import settings
from core.utils.logger import logger


def get_search_client() -> Optional[SearchClient]:
    """
    Create and return Azure AI Search client.
    
    Returns:
        SearchClient instance or None if configuration is missing
    """
    endpoint = settings.AZURE_AI_SEARCH_ENDPOINT or settings.AZURE_SEARCH_ENDPOINT
    api_key = settings.AZURE_AI_SEARCH_API_KEY or settings.AZURE_SEARCH_KEY
    index_name = settings.AZURE_AI_SEARCH_INDEX_NAME or settings.AZURE_SEARCH_INDEX_NAME
    
    if not endpoint or not api_key:
        logger.warning("Azure AI Search credentials not configured")
        return None
    
    credential = AzureKeyCredential(api_key)
    client = SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=credential
    )
    return client


def get_document_intelligence_client():
    """
    Create and return Azure Document Intelligence client.
    
    Returns:
        DocumentAnalysisClient instance or None if configuration is missing
    """
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
        
        if not settings.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or not settings.AZURE_DOCUMENT_INTELLIGENCE_KEY:
            logger.warning("Azure Document Intelligence credentials not configured")
            return None
        
        credential = AzureKeyCredential(settings.AZURE_DOCUMENT_INTELLIGENCE_KEY)
        client = DocumentIntelligenceClient(
            endpoint=settings.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
            credential=credential
        )
        return client
    except ImportError:
        logger.error("Azure Document Intelligence SDK not available")
        return None
