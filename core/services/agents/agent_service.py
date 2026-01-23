"""Main service for question answering using RAG (Retrieval-Augmented Generation)."""
from typing import Dict, Any, Optional

from langchain_openai import AzureChatOpenAI

from core.services.retrieval.retrieval_service import RetrievalService
from core.services.memory.conversation_memory import ConversationMemory
from core.services.agents.agent_chain import AgentChain
from core.services.prompts.prompt_builder import PromptBuilder
from core.services.tools.tool_executor import ToolExecutor
from core.models.user import UserMetadata
from core.utils.logger import logger
from app.config import settings


class AgentService:
    """
    Main service for question answering with RAG and conversation management.
    
    Coordinates document retrieval, LLM generation, and user-specific tools.
    
    Example:
        ```python
        agent = AgentService()
        result = agent.answer_question(
            question="ما هي الإجازات المتاحة؟",
            user_metadata=user_metadata,
            session_id="user_123"
        )
        print(result["answer"])
        print(result["sources"])
        ```
    """
    
    def __init__(self, min_retrieval_score: float = 0.3):
        """
        Initialize the AgentService.
        
        Args:
            min_retrieval_score: Minimum relevance score for retrieved documents (0.0-1.0).
                                Lower values include more documents but may reduce quality.
        """
        # Initialize LLM
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            logger.warning("Azure OpenAI credentials not configured")
            self.llm = None
        else:
            self.llm = AzureChatOpenAI(
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME or "gpt-4o",
                api_version=settings.AZURE_OPENAI_API_VERSION or "2024-02-01",
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,  # type: ignore
                temperature=0.3,
                streaming=True
            )
        
        # Initialize services
        self.retrieval_service = RetrievalService(
            min_score_threshold=min_retrieval_score,
            enable_reranking=False  # Can be enabled later
        )
        self.conversation_memory = ConversationMemory(
            max_recent_exchanges=3,
            enable_summarization=False  # Can be enabled later
        )
        self.prompt_builder = PromptBuilder()
        self.tool_executor = ToolExecutor(max_iterations=3)
        
        # Initialize agent chain
        self.agent_chain: Optional[AgentChain] = None
        if self.llm:
            self.agent_chain = AgentChain(
                llm=self.llm,
                retrieval_service=self.retrieval_service,
                conversation_memory=self.conversation_memory,
                prompt_builder=self.prompt_builder,
                tool_executor=self.tool_executor,
                min_retrieval_score=min_retrieval_score
            )
    
    def answer_question(
        self,
        question: str,
        user_metadata: Optional[UserMetadata] = None,
        session_id: Optional[str] = None,
        category: Optional[str] = None,  # Future: for filtering
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG with document retrieval and LLM generation.
        
        Args:
            question: User's question in Arabic or English
            user_metadata: User metadata for personalized answers (optional)
            session_id: Session ID for conversation history (optional)
            category: Document category filter (reserved for future use)
        
        Returns:
            Dictionary containing:
                - answer (str): Generated answer with citations
                - sources (List[Dict]): Source documents with metadata
                - user_info_used (bool): Whether user-specific tools were used
                - session_id (str): Session identifier
                - retrieval_score (float): Average relevance score
                - error (str): Error message if any
        
        Example:
            ```python
            result = agent.answer_question(
                question="ما هي أنواع الإجازات؟",
                session_id="user_123"
            )
            if not result.get("error"):
                print(result["answer"])
            ```
        """
        if not self.agent_chain or not self.agent_chain.tools_chain:
            logger.error("Agent chain not available")
            return {
                "answer": "Agent not properly initialized. Please check configuration.",
                "sources": [],
                "error": "chain_initialization_failed"
            }
        
        try:
            # Set user metadata for tool access
            self.agent_chain.set_user_metadata(user_metadata)
            
            # Process question through chain
            chain_input = {"input": question, "session_id": session_id}
            result = self.agent_chain.invoke(chain_input)
            
            # Format response
            return {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "user_info_used": result.get("user_info_used", False),
                "session_id": session_id,
                "retrieval_score": result.get("retrieval_score", 0.0),
                "error": result.get("error")
            }
        
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def stream_answer(
        self,
        question: str,
        user_metadata: Optional[UserMetadata] = None,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
    ):
        """
        Stream answer chunks for a question.
        
        Args:
            question: User's question in Arabic or English
            user_metadata: User metadata for personalized answers (optional)
            session_id: Session ID for conversation history (optional)
            category: Document category filter (reserved for future use)
        
        Yields:
            String chunks of the answer
        """
        if not self.agent_chain or not self.agent_chain.tools_chain:
            yield "Agent not properly initialized. Please check configuration."
            return
        
        try:
            # Set user metadata for tool access
            self.agent_chain.set_user_metadata(user_metadata)
            
            # Process question through chain to get full answer
            chain_input = {"input": question, "session_id": session_id}
            result = self.agent_chain.invoke(chain_input)
            
            # Stream the answer in chunks
            answer = result.get("answer", "")
            if answer:
                # Simple chunking by words for streaming effect
                words = answer.split()
                for word in words:
                    yield word + " "
        except Exception as e:
            logger.error(f"Error streaming answer: {str(e)}")
            yield f"Error processing question: {str(e)}"
    
    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history for a specific session."""
        return self.conversation_memory.clear_session(session_id)
