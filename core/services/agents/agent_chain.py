"""AgentChain orchestrates RAG pipeline with streaming."""
from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import AzureChatOpenAI

from core.services.retrieval.retrieval_service import RetrievalService
from core.services.retrieval.retrieval_result import RetrievalResult
from core.services.prompts.prompt_builder import PromptBuilder
from core.services.tools.tool_executor import ToolExecutor
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
        tool_executor: Optional[ToolExecutor] = None,
        min_retrieval_score: float = 0.3
    ):
        self.llm = llm
        self.retrieval_service = retrieval_service
        self.conversation_memory = conversation_memory
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.tool_executor = tool_executor or ToolExecutor()
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
                "user_info_used": False,
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
                yield {"type": "complete", "user_info_used": False, "sources": [], "error": "no_documents"}
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
            
            user_info_used, full_answer = yield from self._generate(messages)
            
            if full_answer:
                self.conversation_memory.add_exchange(session_id, question, full_answer)
            
            yield {
                "type": "complete",
                "user_info_used": user_info_used,
                "sources": self._build_sources(docs),
                "retrieval_score": docs.get_average_score()
            }
        
        except Exception as e:
            logger.error(f"RAG error: {str(e)}", exc_info=True)
            fallback = ErrorHandler.handle_llm_error(e, question, lang)
            yield {"type": "error", "content": fallback, "error": str(e)}
    
    def _generate(self, messages: List):
        """Generate answer with optional tool calls."""
        user_info_used = False
        full_answer = ""
        
        if self.user:
            from core.services.agents.agent_tools import create_user_info_tool
            tool = create_user_info_tool(self.user)
            llm_with_tools = self.llm.bind_tools([tool])
            
            yield {"type": "answer_start"}
            final_message = None
            
            for chunk in llm_with_tools.stream(messages):
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    final_message = chunk
                
                content = self._extract(chunk)
                if content:
                    full_answer += content
                    yield {"type": "answer_chunk", "content": content}
            
            if final_message and hasattr(final_message, 'tool_calls') and final_message.tool_calls:
                result = self.tool_executor.execute(
                    llm_response=final_message,
                    user_metadata=self.user,
                    llm=self.llm
                )
                user_info_used = result.success and len(result.tools_called) > 0
                
                messages.append(AIMessage(content=full_answer, tool_calls=final_message.tool_calls))
                for i, (call, res) in enumerate(zip(final_message.tool_calls, result.results)):
                    messages.append(ToolMessage(content=res, tool_call_id=call.get("id", f"call_{i}")))
                
                full_answer = ""
                for chunk in self.llm.stream(messages):
                    content = self._extract(chunk)
                    if content:
                        full_answer += content
                        yield {"type": "answer_chunk", "content": content}
            
            yield {"type": "answer_end"}
        else:
            yield {"type": "answer_start"}
            for chunk in self.llm.stream(messages):
                content = self._extract(chunk)
                if content:
                    full_answer += content
                    yield {"type": "answer_chunk", "content": content}
            yield {"type": "answer_end"}
        
        return user_info_used, full_answer
    
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
