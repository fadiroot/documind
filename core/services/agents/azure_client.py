"""Azure OpenAI client factory using app settings (API key auth)."""
from typing import Optional

from langchain_openai import AzureChatOpenAI

from app.config import settings


def get_azure_llm() -> Optional[AzureChatOpenAI]:
    """Get LangChain AzureChatOpenAI for AgentService/AgentChain."""
    if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
        return None

    return AzureChatOpenAI(
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME or "gpt-4o",
        api_version=settings.AZURE_OPENAI_API_VERSION or "2024-02-01",
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,  # pyright: ignore[reportArgumentType]
        temperature=0.3,
        streaming=True,
    )
