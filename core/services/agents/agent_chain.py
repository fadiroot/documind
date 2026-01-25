"""AgentChain orchestrates RAG pipeline with streaming."""
from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from core.services.retrieval.retrieval_service import RetrievalService
from core.services.retrieval.retrieval_result import RetrievalResult
from core.services.prompts.prompt_builder import PromptBuilder
from core.services.memory.conversation_memory import ConversationMemory
from core.services.errors.error_handler import ErrorHandler
from core.services.agents.question_router_agent import QuestionRouterAgent
from core.services.utils.metadata_utils import build_resource_path
from core.models.user import UserMetadata
from core.utils.logger import logger


class AgentChain:
    """
    RAG pipeline orchestrator with streaming.
    
    Flow: Question → Router → Retrieve → Generate → Stream
    """
    
    def __init__(
        self,
        llm: AzureChatOpenAI,
        retrieval_service: RetrievalService,
        conversation_memory: ConversationMemory,
        prompt_builder: Optional[PromptBuilder] = None,
        min_retrieval_score: float = 0.3
    ):
        self.llm = llm
        self.retrieval_service = retrieval_service
        self.conversation_memory = conversation_memory
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.min_retrieval_score = min_retrieval_score
        self.user: Optional[UserMetadata] = None
        self.router = QuestionRouterAgent(llm) if llm else None
    
    def set_user(self, user: Optional[UserMetadata]):
        """Set user for personalization."""
        self.user = user
    
    def stream(self, input_dict: Dict[str, Any]):
        """
        Stream answer generation (primary method).
        
        Yields: status, answer_start, answer_chunk, answer_end, complete, error
        """
        question = input_dict.get("input", "")
        session_id = input_dict.get("session_id")
        
        if not question:
            yield {"type": "error", "content": "No question", "error": "empty"}
            return
        
        language = self.prompt_builder.detect_language(question)
        is_arabic = "arabic" in language.lower()
        lang = "arabic" if is_arabic else "english"
        needs_docs = self._needs_documents(question, lang)
        
        if needs_docs:
            yield from self._answer_with_docs(question, lang, session_id)
        else:
            yield from self._answer_general(question, lang, session_id)
    
    def _needs_documents(self, question: str, lang: str) -> bool:
        """Check if question needs document retrieval."""
        if not self.router:
            return True
        
        try:
            decision = self.router.should_retrieve_documents(question, lang)
            return decision.get("needs_retrieval", True)
        except Exception as e:
            logger.warning(f"Router error: {str(e)}")
            return True
    
    def _answer_general(self, question: str, lang: str, session_id: Optional[str]):
        """Answer general questions without document retrieval."""
        yield {"type": "status", "content": "Processing..."}
        
        history = self.conversation_memory.get_summary(session_id)
        
        if lang == "arabic":
            system = "أنت مساعد ذكي متخصص في الوثائق القانونية والتنظيمية السعودية. كن ودوداً ومهذباً."
            user = f"{history}\n\nالسؤال: {question}\n\nأجب بشكل طبيعي." if history and history != "No previous conversation." else f"السؤال: {question}"
        else:
            system = "You are a helpful AI assistant for Saudi Arabian legal documents. Be friendly and polite."
            user = f"{history}\n\nQuestion: {question}\n\nAnswer naturally." if history and history != "No previous conversation." else f"Question: {question}"
        
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        
        # Add user information
        user_info = self._format_user_info()
        if user_info:
            messages[0].content = f"{user_info}\n\n{messages[0].content}"
        
        try:
            yield {"type": "answer_start"}
            full_answer = ""
            
            for chunk in self.llm.stream(messages):
                content = self._extract(chunk)
                if content:
                    full_answer += content
                    yield {"type": "answer_chunk", "content": content}
            
            yield {"type": "answer_end"}
            
            self.conversation_memory.add_exchange(session_id, question, full_answer)
            
            yield {
                "type": "complete",
                "sources": [],
                "retrieval_score": 0.0
            }
        except Exception as e:
            logger.error(f"General answer error: {str(e)}")
            yield {"type": "error", "content": str(e)}
    
    def _answer_with_docs(self, question: str, lang: str, session_id: Optional[str]):
        """Answer questions using retrieved documents (RAG)."""
        try:
            yield {"type": "status", "content": "Retrieving documents..."}
            docs = self.retrieval_service.retrieve(
                query=question,
                top_k=5,
                min_score=self.min_retrieval_score
            )
            
            if not docs.has_results():
                logger.warning("No documents found")
                fallback = ErrorHandler.handle_no_documents(question, lang)
                yield {"type": "answer_start"}
                yield {"type": "answer_chunk", "content": fallback}
                yield {"type": "answer_end"}
                yield {"type": "complete", "sources": [], "error": "no_documents"}
                return
            
            yield {"type": "status", "content": "Generating answer..."}
            history = self.conversation_memory.get_summary(session_id)
            system_prompt, user_prompt = self.prompt_builder.build_full_prompt(
                question=question,
                docs=docs.documents,
                history_summary=history
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            full_answer = yield from self._generate(messages)
            
            if full_answer:
                self.conversation_memory.add_exchange(session_id, question, full_answer)
            
            yield {
                "type": "complete",
                "sources": self._build_sources(docs),
                "retrieval_score": docs.get_average_score()
            }
        
        except Exception as e:
            logger.error(f"RAG error: {str(e)}", exc_info=True)
            fallback = ErrorHandler.handle_llm_error(e, question, lang)
            yield {"type": "error", "content": fallback, "error": str(e)}
    
    def _generate(self, messages: List):
        """Generate answer with user data included in context."""
        full_answer = ""
        
        # Add user information to system prompt
        user_info = self._format_user_info()
        if user_info:
            if messages and isinstance(messages[0], SystemMessage):
                messages[0].content = f"{user_info}\n\n{messages[0].content}"
            else:
                messages.insert(0, SystemMessage(content=user_info))
        
        yield {"type": "answer_start"}
        for chunk in self.llm.stream(messages):
            content = self._extract(chunk)
            if content:
                full_answer += content
                yield {"type": "answer_chunk", "content": content}
        yield {"type": "answer_end"}
        
        return full_answer
    
    def _format_user_info(self) -> str:
        """Format user metadata as context string."""
        if not self.user:
            return ""
        
        user = self.user
        info_parts = [
            "User Information:",
            f"- Full Name: {user.full_name}",
            f"- Cadre: {user.cadre}",
            f"- Current Rank: {user.current_rank or 'N/A'}",
            f"- Years in Rank: {user.years_in_rank or 'N/A'}",
            f"- Administration: {user.administration}"
        ]
        
        if user.job_title:
            info_parts.append(f"- Job Title: {user.job_title}")
        
        if user.expected_filter:
            info_parts.append(f"- Expected Filter: {user.expected_filter}")
        
        return "\n".join(info_parts)
    
    def _extract(self, response: Any) -> str:
        """Extract text from LLM response."""
        if hasattr(response, 'content'):
            return str(response.content) if response.content else ""
        return str(response)
    
    def _build_sources(self, docs: RetrievalResult) -> List[Dict[str, Any]]:
        """Build source list from retrieval results."""
        sources = []
        for doc, score, meta in zip(docs.documents, docs.scores, docs.metadata):
            sources.append({
                "id": meta.get("id", ""),
                "document_name": meta.get("document_name", "Unknown"),
                "page_number": meta.get("page_number"),
                "score": score,
                "resource_path": build_resource_path(meta),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        return sources
