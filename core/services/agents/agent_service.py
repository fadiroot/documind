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
        self.agent_chain: Optional[AgentChain] = None
        if self.llm:
            self.agent_chain = AgentChain(
                llm=self.llm,
                retriever=self.retriever,
                conversation_memory=self.conversation_memory
            )
        
    
    def answer_question(
        self, 
        question: str, 
        context: Optional[List[str]] = None,
        category: Optional[str] = None,
        user_metadata: Optional[UserMetadata] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Answer a question using LangChain RAG chain with tools."""
        if not self.agent_chain or not self.agent_chain.tools_chain:
            logger.error("Agent chain not available")
            return {
                "answer": "Agent not properly initialized. Please check configuration.",
                "sources": [],
                "error": "Chain initialization failed"
            }
        
        try:
            self.agent_chain.set_user_metadata(user_metadata)
            
            chain_input = {"input": question, "session_id": session_id}
            if category:
                chain_input["input"] = f"[Category: {category}] {question}"
            
            result = self.agent_chain.invoke(chain_input)
            
            if isinstance(result, dict):
                answer = result.get("answer", "")
                user_info_used = result.get("user_info_used", False)
            else:
                answer = str(result)
                user_info_used = False
            
            source_docs = self.retriever._get_relevant_documents(question)
            sources = self._format_source_documents(source_docs)
            
            return {
                "answer": answer,
                "sources": sources,
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
        """Stream answer tokens as they're generated."""
        if not self.agent_chain or not self.agent_chain.tools_chain:
            yield "Service not properly initialized."
            return
        
        try:
            self.agent_chain.set_user_metadata(user_metadata)
            
            chain_input = {"input": question, "session_id": session_id}
            if category:
                chain_input["input"] = f"[Category: {category}] {question}"
            
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
    
    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history for a specific session."""
        return self.conversation_memory.clear_session(session_id)
