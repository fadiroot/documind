"""AgentChain orchestrates the RAG pipeline: retrieval → prompt → LLM → validation."""
from typing import Dict, Any, Optional, List
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage
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
    Refactored AgentChain with clean separation of concerns.
    
    Responsibilities:
    - Orchestrate retrieval → prompt → LLM → validation pipeline
    - Handle tool execution loop
    - Manage conversation state
    - Error recovery
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
        self.current_user_metadata: Optional[UserMetadata] = None
        # Initialize question router agent
        self.question_router = QuestionRouterAgent(llm) if llm else None
        self._setup_chain()
    
    def set_user_metadata(self, user_metadata: Optional[UserMetadata]):
        """Set the current user metadata for tool access."""
        self.current_user_metadata = user_metadata
    
    def _setup_chain(self):
        """Set up LangChain RAG chain using LCEL."""
        if not self.llm:
            logger.error("LLM not available, chain not initialized")
            self.tools_chain = None
            return
        
        try:
            self.tools_chain = RunnableLambda(self._process_question)
            self.tools_chain = self.tools_chain.with_config(run_name="process_question")
        except Exception as e:
            logger.error(f"Error setting up RAG chain: {str(e)}")
            self.tools_chain = None
    
    def _process_question(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a question through the complete pipeline.
        
        Pipeline:
        1. Retrieve documents (with thresholds)
        2. Build prompts (system + context + user)
        3. Execute LLM (with tools if needed)
        4. Save conversation
        """
        question = input_dict.get("input", "")
        session_id = input_dict.get("session_id")
        
        if not question:
            return {
                "answer": "No question provided.",
                "user_info_used": False,
                "error": "empty_question"
            }
        
        # Detect language
        language = self.prompt_builder.detect_language(question)
        language_code = "arabic" if "arabic" in language.lower() else "english"
        
        # Step 0: Use router agent to determine if retrieval is needed
        needs_retrieval = True
        router_decision = None
        
        if self.question_router:
            try:
                router_decision = self.question_router.should_retrieve_documents(question, language_code)
                needs_retrieval = router_decision.get("needs_retrieval", True)
            except Exception as e:
                logger.warning(f"Error in question router agent: {str(e)}, defaulting to retrieval")
                needs_retrieval = True
        
        # If router says no retrieval needed, handle as general question
        if not needs_retrieval:
            return self._handle_general_question(question, language_code, session_id, router_decision)
        
        try:
            # Step 1: Retrieve documents
            retrieval_result = self.retrieval_service.retrieve(
                query=question,
                top_k=5,
                min_score=self.min_retrieval_score
            )
            
            # Handle retrieval failures
            if not retrieval_result.has_results():
                logger.warning(f"No documents retrieved. Total found: {retrieval_result.total_found}, Filtered: {retrieval_result.filtered_count}")
                fallback = ErrorHandler.handle_no_documents(question, language_code)
                return {
                    "answer": fallback,
                    "user_info_used": False,
                    "sources": [],
                    "error": "no_documents"
                }
            
            # Check confidence - but be lenient since retrieval already filtered
            avg_score = retrieval_result.get_average_score()
            # Only warn if score is very low (< 0.2), but still proceed
            if avg_score < 0.2 and len(retrieval_result.documents) > 0:
                logger.warning(f"Low average score: {avg_score:.4f}, but proceeding with {len(retrieval_result.documents)} documents")
            
            # Step 2: Get conversation summary (not full history)
            history_summary = self.conversation_memory.get_summary(session_id)
            
            # Step 3: Build prompts
            system_prompt, full_prompt = self.prompt_builder.build_full_prompt(
                question=question,
                docs=retrieval_result.documents,
                history_summary=history_summary
            )
            
            # Step 4: Execute LLM with tools if needed
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=full_prompt)
            ]
            
            # Setup tools if user metadata available
            llm_with_tools = self.llm
            if self.current_user_metadata:
                from core.services.agents.agent_tools import create_user_info_tool
                user_info_tool = create_user_info_tool(self.current_user_metadata)
                llm_with_tools = self.llm.bind_tools([user_info_tool])
            
            # Get initial LLM response
            response = llm_with_tools.invoke(messages)
            
            # Step 5: Handle tool calls if any
            tool_result = self.tool_executor.execute(
                llm_response=response,
                user_metadata=self.current_user_metadata,
                llm=self.llm
            )
            
            user_info_used = tool_result.success and len(tool_result.tools_called) > 0
            
            # If tools were called, get final response
            if tool_result.results:
                tool_results_text = "\n".join(tool_result.results)
                follow_up_prompt = f"{full_prompt}\n\nTool Results:\n{tool_results_text}"
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=follow_up_prompt)
                ]
                response = self.llm.invoke(messages)
            
            # Extract answer
            answer = self._extract_content(response)
            
            # Step 6: Save conversation
            self.conversation_memory.add_exchange(
                session_id=session_id,
                question=question,
                answer=answer
            )
            
            return {
                "answer": answer,
                "user_info_used": user_info_used,
                "sources": self._format_sources(retrieval_result),
                "retrieval_score": avg_score
            }
        
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            fallback = ErrorHandler.handle_llm_error(e, question, language_code)
            return {
                "answer": fallback,
                "user_info_used": False,
                "sources": [],
                "error": str(e)
            }
    
    def _handle_general_question(
        self,
        question: str,
        language_code: str,
        session_id: Optional[str],
        router_decision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle general questions without document retrieval."""
        # Get conversation history summary
        history_summary = self.conversation_memory.get_summary(session_id)
        
        # Build a general system prompt
        if language_code == "arabic":
            system_prompt = """أنت مساعد ذكي متخصص في الإجابة على الأسئلة حول الوثائق القانونية والتنظيمية السعودية.

يمكنك الإجابة على:
- التحيات والأسئلة العامة
- الأسئلة حول الوثائق القانونية والتنظيمية (بعد البحث في قاعدة المعرفة)
- أسئلة حول حقوق الموظفين والهندسة
- الأسئلة حول المحادثات السابقة (لديك إمكانية الوصول إلى ملخص المحادثة السابقة)

كن ودوداً ومهذباً في الرد على التحيات والأسئلة العامة.
إذا سألك المستخدم عن سؤاله السابق أو المحادثة السابقة، استخدم ملخص المحادثة المتاح للإجابة."""
            
            # Build user prompt with history if available
            if history_summary and history_summary != "No previous conversation.":
                user_prompt = f"{history_summary}\n\nالسؤال: {question}\n\nأجب بشكل طبيعي وودود. إذا كان السؤال يتعلق بالمحادثة السابقة، استخدم المعلومات المتاحة أعلاه."
            else:
                user_prompt = f"السؤال: {question}\n\nأجب بشكل طبيعي وودود."
        else:
            system_prompt = """You are a helpful AI assistant specialized in answering questions about Saudi Arabian legal and regulatory documents.

You can answer:
- Greetings and general questions
- Questions about legal and regulatory documents (after searching the knowledge base)
- Questions about employee rights and engineering
- Questions about previous conversations (you have access to conversation history summary)

Be friendly and polite when responding to greetings and general questions.
If the user asks about their previous question or conversation history, use the available conversation summary to answer."""
            
            # Build user prompt with history if available
            if history_summary and history_summary != "No previous conversation.":
                user_prompt = f"{history_summary}\n\nQuestion: {question}\n\nAnswer naturally and friendly. If the question relates to previous conversation, use the information available above."
            else:
                user_prompt = f"Question: {question}\n\nAnswer naturally and friendly."
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            answer = self._extract_content(response)
            
            # Save conversation
            self.conversation_memory.add_exchange(
                session_id=session_id,
                question=question,
                answer=answer
            )
            
            return {
                "answer": answer,
                "user_info_used": False,
                "sources": [],
                "retrieval_score": 0.0,
                "router_decision": router_decision
            }
        except Exception as e:
            logger.error(f"Error handling general question: {str(e)}")
            # Fallback response
            if language_code == "arabic":
                fallback_answer = "مرحباً! أنا مساعد ذكي متخصص في الإجابة على الأسئلة حول الوثائق القانونية والتنظيمية السعودية. كيف يمكنني مساعدتك اليوم؟"
            else:
                fallback_answer = "Hello! I'm an AI assistant specialized in answering questions about Saudi Arabian legal and regulatory documents. How can I help you today?"
            
            return {
                "answer": fallback_answer,
                "user_info_used": False,
                "sources": [],
                "retrieval_score": 0.0,
                "error": str(e),
                "router_decision": router_decision
            }
    
    def _extract_content(self, response: Any) -> str:
        """Extract content from LLM response."""
        if hasattr(response, 'content'):
            content = response.content
            return str(content) if content else ""
        return str(response)
    
    def _format_sources(self, retrieval_result: RetrievalResult) -> List[Dict[str, Any]]:
        """Format sources from retrieval result."""
        sources = []
        for doc, score, metadata in zip(
            retrieval_result.documents,
            retrieval_result.scores,
            retrieval_result.metadata
        ):
            sources.append({
                "id": metadata.get("id", ""),
                "document_name": metadata.get("document_name", "Unknown"),
                "page_number": metadata.get("page_number"),
                "score": score,
                "resource_path": self._build_resource_path(metadata),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        return sources
    
    def _build_resource_path(self, metadata: dict) -> str:
        """Build resource path from metadata using shared utility."""
        return build_resource_path(metadata)
    
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the chain with input."""
        if not self.tools_chain:
            return {
                "answer": "Chain not available",
                "user_info_used": False,
                "error": "chain_not_initialized"
            }
        return self.tools_chain.invoke(input_dict)
