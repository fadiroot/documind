"""Agent service using LangChain RAG chains with LCEL and tools."""
from typing import List, Dict, Any, Optional, Iterator

from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document

from core.services.retrieval.retrieval_service import RetrievalService
from core.services.agents.azure_retriever import AzureAISearchRetriever
from core.services.agents.conversation_memory import ConversationMemory
from core.services.agents.agent_chain import AgentChain
from core.models.user import UserMetadata
from core.utils.logger import logger
from app.config import settings


class AgentService:
    """Simple RAG service using LangChain RAG chains with tools for question answering."""
    
    def __init__(self):
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
        
        self.retrieval_service = RetrievalService()
        self.retriever = AzureAISearchRetriever(self.retrieval_service, top_k=5)
        self.conversation_memory = ConversationMemory()
        
        # Initialize chain
        self.agent_chain: Optional[AgentChain] = None
        if self.llm:
            self.agent_chain = AgentChain(
                llm=self.llm,
                retriever=self.retriever,
                conversation_memory=self.conversation_memory
            )
        
        logger.info("Agent service initialized")
    
    def answer_question(
        self, 
        question: str, 
        context: Optional[List[str]] = None,
        category: Optional[str] = None,
        user_metadata: Optional[UserMetadata] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using LangChain RAG chain with tools.
        The agent will decide if user info is needed based on the question.
        
        Args:
            question: User question
            context: Optional context IDs to filter search (not yet implemented)
            category: Optional category filter (e.g., "legal", "financial", "technical")
            user_metadata: Optional user metadata (stored for tool access)
            session_id: Optional session ID for conversation memory (Azure thread)
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.agent_chain or not self.agent_chain.tools_chain:
            logger.error("Agent chain not available")
            return {
                "answer": "Agent not properly initialized. Please check configuration.",
                "sources": [],
                "error": "Chain initialization failed"
            }
        
        try:
            # Set user metadata for tool access
            self.agent_chain.set_user_metadata(user_metadata)
            
            # Prepare input
            chain_input = {"input": question, "session_id": session_id}
            if category:
                chain_input["input"] = f"[Category: {category}] {question}"
            
            # Use tools chain which will decide if user info is needed
            result = self.agent_chain.invoke(chain_input)
            
            # Extract answer and user_info_used flag
            if isinstance(result, dict):
                answer = result.get("answer", "")
                user_info_used = result.get("user_info_used", False)
            else:
                answer = str(result)
                user_info_used = False
            
            # Get source documents for citation (using original question)
            source_docs = self.retriever._get_relevant_documents(question)
            
            # Format sources from LangChain documents
            sources = self._format_source_documents(source_docs)
            
            # Calculate confidence from source scores
            confidence = self._calculate_confidence_from_docs(source_docs)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "user_info_used": user_info_used,
                "session_id": session_id
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
        category: Optional[str] = None,
        user_metadata: Optional[UserMetadata] = None,
        session_id: Optional[str] = None
    ) -> Iterator[str]:
        """
        Stream answer tokens as they're generated.
        
        Args:
            question: User question
            category: Optional category filter (e.g., "legal", "financial", "technical")
            user_metadata: Optional user metadata (stored for tool access)
            session_id: Optional session ID for conversation memory (Azure thread)
        
        Yields:
            Answer chunks as strings
        """
        if not self.agent_chain or not self.agent_chain.tools_chain:
            yield "Service not properly initialized."
            return
        
        try:
            # Set user metadata for tool access
            self.agent_chain.set_user_metadata(user_metadata)
            
            # Prepare input
            chain_input = {"input": question, "session_id": session_id}
            if category:
                chain_input["input"] = f"[Category: {category}] {question}"
            
            # Stream answer (simulate streaming by word-by-word)
            result = self.agent_chain.invoke(chain_input)
            answer = result.get("answer", "") if isinstance(result, dict) else str(result)
            words = answer.split()
            
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
        except Exception as e:
            logger.error(f"Error streaming answer: {str(e)}")
            yield f"Error: {str(e)}"
    
    def _format_source_documents(self, source_docs: List[Document]) -> List[Dict[str, Any]]:
        """Format LangChain documents to source format."""
        sources = []
        for doc in source_docs:
            content = doc.page_content
            truncated_content = content[:500] + "..." if len(content) > 500 else content
            
            sources.append({
                "id": doc.metadata.get('id', ''),
                "content": truncated_content,
                "document_name": doc.metadata.get('document_name', 'Unknown'),
                "page_number": doc.metadata.get('page_number'),
                "chunk_index": doc.metadata.get('chunk_index'),
                "score": doc.metadata.get('score', 0.0)
            })
        return sources
    
    def _calculate_confidence_from_docs(self, source_docs: List[Document]) -> Optional[float]:
        """Calculate confidence score from top source documents."""
        if not source_docs:
            return 0.0
        
        # Get scores from top 3 documents
        scores = [doc.metadata.get('score', 0.0) for doc in source_docs[:3]]
        if not scores:
            return None
        
        avg_score = sum(scores) / len(scores)
        
        # Normalize if score is > 1.0 (assumes percentage-based scores)
        if avg_score > 1.0:
            avg_score = avg_score / 100.0
        
        # Clamp between 0.0 and 1.0
        return min(1.0, max(0.0, avg_score))
    
    def clear_conversation(self, session_id: str) -> bool:
        """
        Clear conversation history for a specific session.
        
        Args:
            session_id: Session ID to clear
        
        Returns:
            True if session was cleared, False if session didn't exist
        """
        return self.conversation_memory.clear_session(session_id)
