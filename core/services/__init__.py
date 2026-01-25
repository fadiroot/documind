"""Core services package - production-ready, organized by domain.

Main Services:
- AgentService: Question answering with RAG
- PDFService: Document processing and chunking
- IndexService: Azure AI Search index management
- StorageService: Document indexing and storage
- auth_service: Authentication and authorization

Usage:
    from core.services import AgentService, PDFService, IndexService

    # For question answering
    agent = AgentService()
    result = agent.answer_question("What is...?")

    # For document processing
    pdf_service = PDFService()
    chunks = pdf_service.chunk_pdf_with_metadata(pdf_bytes)

    # For index management
    index_service = IndexService()
    index_service.create_index()
"""
# Agent services
from core.services.agents import AgentService

# Retrieval services (internal use - not exported)
from core.services.retrieval import EmbeddingService

# Indexing services
from core.services.indexing import IndexService, StorageService

# Document services
from core.services.documents import PDFService, DocumentChunker, DocumentChunk

# Auth services
from core.services.auth import auth_service

__all__ = [
    # Main Services (Public API)
    "AgentService",
    "PDFService",
    "IndexService",
    "StorageService",
    "auth_service",
    # Document Processing
    "DocumentChunker",
    "DocumentChunk",
    # Utilities
    "EmbeddingService",
]
