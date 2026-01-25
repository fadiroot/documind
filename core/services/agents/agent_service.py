"""Main service for question answering using RAG."""
from typing import Dict, Any, Optional

from langchain_openai import AzureChatOpenAI

from core.services.retrieval.retrieval_service import RetrievalService
from core.services.memory.conversation_memory import ConversationMemory
from core.services.agents.agent_chain import AgentChain
from core.services.prompts.prompt_builder import PromptBuilder
from core.models.user import UserMetadata
from core.utils.logger import logger
from app.config import settings


class AgentService:
    """
    RAG question answering service with streaming.
    
    Example:
        service = AgentService()
        
        # Streaming
        for event in service.stream("ما هي الإجازات؟", user, "session_123"):
            if event["type"] == "answer_chunk":
                print(event["content"], end="")
    """
    
    def __init__(self, min_retrieval_score: float = 0.3):
        """Initialize service with Azure OpenAI."""
        self.llm = self._init_llm()
        self.agent_chain: Optional[AgentChain] = None
        if self.llm:
            self.agent_chain = AgentChain(
                llm=self.llm, # type: ignore
                retrieval_service=RetrievalService(
                    min_score_threshold=min_retrieval_score,
                    enable_reranking=False
                ),
                conversation_memory=ConversationMemory(
                    max_recent_exchanges=3,
                    enable_summarization=False
                ),
                prompt_builder=PromptBuilder(),
                min_retrieval_score=min_retrieval_score
            )
    
    def _init_llm(self) -> Optional[AzureChatOpenAI]:
        """Initialize Azure OpenAI LLM."""
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            logger.error("Azure OpenAI not configured")
            return None
        
        try:
            return AzureChatOpenAI(
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME or "gpt-4o", # type: ignore
                api_version=settings.AZURE_OPENAI_API_VERSION or "2024-02-01", # type: ignore
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT, # type: ignore
                api_key=settings.AZURE_OPENAI_API_KEY, # type: ignore
                temperature=0.3,
                streaming=True
            )
        except Exception as e:
            logger.error(f"Failed to init Azure OpenAI: {str(e)}")
            return None
    
    def stream(
        self,
        question: str,
        user: Optional[UserMetadata] = None,
        session_id: Optional[str] = None,
    ):
        """
        Stream answer in real-time (primary method).
        
        Yields: status, answer_start, answer_chunk, answer_end, complete, error
        """
        if not self.agent_chain:
            yield {"type": "error", "content": "Service not initialized"}
            return
        
        try:
            self.agent_chain.set_user(user)
            for event in self.agent_chain.stream({"input": question, "session_id": session_id}):
                yield event
        except Exception as e:
            logger.error(f"Stream error: {str(e)}", exc_info=True)
            yield {"type": "error", "content": f"Error: {str(e)}"}
    
