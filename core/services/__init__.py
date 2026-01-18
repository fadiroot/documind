"""Services package - organized by domain."""
# Agent services
from core.services.agents import (
    AgentService,
    AgentChain,
    AzureAISearchRetriever,
    ConversationMemory,
    create_user_info_tool,
)

# Retrieval services
from core.services.retrieval import (
    RetrievalService,
    EmbeddingService,
    VectorStoreService,
)

# Document services
from core.services.documents import (
    PDFService,
    IndexService,
)

# Auth services
from core.services.auth import (
    auth_service,
)

__all__ = [
    # Agents
    "AgentService",
    "AgentChain",
    "AzureAISearchRetriever",
    "ConversationMemory",
    "create_user_info_tool",
    # Retrieval
    "RetrievalService",
    "EmbeddingService",
    "VectorStoreService",
    # Documents
    "PDFService",
    "IndexService",
    # Auth
    "auth_service",
]
