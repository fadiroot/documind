"""Configuration management for the application."""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    API_TITLE: str = "RAG Application"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Azure Document Intelligence
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT: Optional[str] = None
    AZURE_DOCUMENT_INTELLIGENCE_KEY: Optional[str] = None
    
    # Azure Cognitive Search (legacy names)
    AZURE_SEARCH_ENDPOINT: Optional[str] = None
    AZURE_SEARCH_KEY: Optional[str] = None
    AZURE_SEARCH_INDEX_NAME: str = "documents-index"
    
    # Azure AI Search
    AZURE_AI_SEARCH_ENDPOINT: Optional[str] = None
    AZURE_AI_SEARCH_API_KEY: Optional[str] = None
    AZURE_AI_SEARCH_INDEX_NAME: Optional[str] = None
    
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_API_VERSION: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = None
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: Optional[str] = None
    
    # Azure Project / AI Agents
    AZURE_PROJECT_ENDPOINT: Optional[str] = None
    AZURE_PROJECT_API_KEY: Optional[str] = None
    AZURE_PROJECT_NAME: Optional[str] = None
    AZURE_AI_AGENT_ID: Optional[str] = None  # e.g., "asst_FpXxhFbEWYZftZh9PmeVh9n2"
    
    # Azure Identity
    AZURE_SUBSCRIPTION_ID: Optional[str] = None
    AZURE_RESOURCE_GROUP_NAME: Optional[str] = None
    AZURE_TENANT_ID: Optional[str] = None
    AZURE_CLIENT_ID: Optional[str] = None
    AZURE_CLIENT_SECRET: Optional[str] = None
    
    # OpenAI (non-Azure)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@postgres:5432/ragdb"
    
    # Application
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore any extra env vars not defined here


settings = Settings()
