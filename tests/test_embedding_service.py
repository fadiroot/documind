"""Tests for embedding service."""
import pytest
from core.services.retrieval.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Test cases for EmbeddingService."""
    
    def test_embedding_service_initialization(self):
        """Test embedding service can be initialized."""
        service = EmbeddingService()
        assert service is not None
    
    # Add more tests as needed
    # Note: Actual embedding tests would require OpenAI API key
