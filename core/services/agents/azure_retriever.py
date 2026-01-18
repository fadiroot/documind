"""Azure AI Search retriever wrapper for LangChain."""
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from core.services.retrieval.retrieval_service import RetrievalService


class AzureAISearchRetriever(BaseRetriever):
    """
    LangChain retriever wrapper for Azure AI Search.
    This allows us to use LangChain's RAG chains with Azure AI Search.
    """
    
    def __init__(self, retrieval_service: RetrievalService, top_k: int = 5):
        # Initialize parent first
        super().__init__()
        # Use object.__setattr__ to bypass Pydantic validation for custom attributes
        object.__setattr__(self, 'retrieval_service', retrieval_service)
        object.__setattr__(self, 'top_k', top_k)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents from Azure AI Search."""
        results = self.retrieval_service.retrieve(query, top_k=self.top_k)  # type: ignore
        
        documents = []
        for result in results:
            doc = Document(
                page_content=result.get('content', ''),
                metadata={
                    'id': result.get('id'),
                    'document_name': result.get('document_name', ''),
                    'page_number': result.get('page_number'),
                    'chunk_index': result.get('chunk_index'),
                    'score': result.get('score', 0.0)
                }
            )
            documents.append(doc)
        return documents
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async retrieve relevant documents."""
        return self._get_relevant_documents(query)
