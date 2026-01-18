"""Embedding service for creating vector embeddings."""
from typing import List, Optional
from openai import AzureOpenAI
from app.config import settings
from core.utils.logger import logger


class EmbeddingService:
    """Service for creating embeddings using Azure OpenAI."""
    
    def __init__(self):
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            logger.warning("Azure OpenAI credentials not configured")
            self.client = None
        else:
            self.client = AzureOpenAI(
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION or "2024-02-01",
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
            )
        self.model = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME or "text-embedding-3-large"
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector or None if client not available
        """
        if not self.client:
            logger.error("OpenAI client not available")
            return None
        
        # Validate input
        if not text:
            logger.error("Invalid input: text is None or empty")
            return None
        
        if not isinstance(text, str):
            logger.error(f"Invalid input: text must be a string, got {type(text)}")
            return None
        
        text = text.strip()
        if not text:
            logger.error("Invalid input: text is empty after stripping")
            return None
        
        try:
            logger.debug(f"Creating embedding for text (length: {len(text)}, model: {self.model})")
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            logger.error(f"Model: {self.model}, Text length: {len(text) if text else 0}")
            return None
    
    def create_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Create embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        if not self.client:
            logger.error("OpenAI client not available")
            return [None] * len(texts)
        
        # Validate and filter inputs
        if not texts:
            return []
        
        # Filter out empty or invalid texts
        valid_texts = [text.strip() for text in texts if text and isinstance(text, str) and text.strip()]
        if not valid_texts:
            logger.error("No valid texts provided for embedding")
            return [None] * len(texts)
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=valid_texts
            )
            # Map results back to original positions (None for invalid inputs)
            result_map = {text: item.embedding for text, item in zip(valid_texts, response.data)}
            results = []
            for text in texts:
                if text and isinstance(text, str) and text.strip():
                    results.append(result_map.get(text.strip()))
                else:
                    results.append(None)
            return results
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            return [None] * len(texts)
