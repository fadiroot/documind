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
        logger.warning(f"  Endpoint: {endpoint or 'NOT SET'}")
        logger.warning(f"  API Key: {'SET' if api_key else 'NOT SET'}")
        logger.warning(f"  Index Name: {index_name or 'NOT SET'}")
        return None
    
    # Validate endpoint format
    if not endpoint.startswith(('https://', 'http://')):
        logger.error(f"Invalid Azure AI Search endpoint format: {endpoint}")
        logger.error("Endpoint should start with https:// or http://")
        return None
    
    try:
        credential = AzureKeyCredential(api_key)
        client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential
        )
        logger.debug(f"Azure AI Search client initialized for endpoint: {endpoint}, index: {index_name}")
        return client
    except Exception as e:
        logger.error(f"Failed to create Azure AI Search client: {str(e)}")
        return None


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
