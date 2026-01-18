"""Agent chain setup and processing logic."""
from typing import Dict, Any, Optional

from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage

from core.services.agents.azure_retriever import AzureAISearchRetriever
from core.services.agents.conversation_memory import ConversationMemory
from core.services.agents.agent_tools import create_user_info_tool
from core.models.user import UserMetadata
from core.utils.logger import logger


class AgentChain:
    """Manages the LangChain RAG chain with tools."""
    
    def __init__(
        self,
        llm: AzureChatOpenAI,
        retriever: AzureAISearchRetriever,
        conversation_memory: ConversationMemory
    ):
        self.llm = llm
        self.retriever = retriever
        self.conversation_memory = conversation_memory
        self.tools_chain = None
        self.current_user_metadata: Optional[UserMetadata] = None
        self._setup_chain()
    
    def set_user_metadata(self, user_metadata: Optional[UserMetadata]):
        """Set the current user metadata for tool access."""
        self.current_user_metadata = user_metadata
    
    def _build_tool_instruction(self) -> str:
        """Build tool usage instructions."""
        return """
Available tool:
- get_user_info: Use ONLY for questions about the LOGGED-IN USER'S actual account/profile.
  DO NOT use for follow-up questions referencing scenarios from conversation history.
"""
    
    def _build_base_prompt(
        self, 
        context: str, 
        chat_history: str, 
        question: str, 
        response_language: str,
        tool_instruction: str = ""
    ) -> str:
        """Build the main prompt with all context."""
        return f"""You are a helpful AI assistant that answers questions based on retrieved documents.{tool_instruction}

Document Context:
{context}

Conversation History:
{chat_history}

Question: {question}

Instructions:
1. Answer based on the conversation history and document context.
2. If this is a follow-up question, use the scenario from the previous conversation.
3. Format your answer with clear headings, bullet points, and proper spacing.
4. RESPOND ENTIRELY IN {response_language.upper()} - match the question's language exactly."""
    
    def _build_final_prompt(
        self,
        context: str,
        chat_history: str,
        question: str,
        tool_results: list[str],
        response_language: str
    ) -> str:
        """Build the final prompt after tool execution."""
        return f"""Based on the following information, provide a well-organized answer.

Document Context: {context}

Conversation History: {chat_history}

Tool Results:
{chr(10).join(tool_results)}

Question: {question}

IMPORTANT: If this question references a scenario from conversation history, answer based on THAT scenario, not the logged-in user's profile.

Format your answer clearly with headings, bullet points, and proper spacing.
RESPOND ENTIRELY IN {response_language.upper()}."""
    
    def _setup_chain(self):
        """Set up LangChain RAG chain with tools using LCEL."""
        if not self.llm:
            logger.error("LLM not available, chain not initialized")
            return
        
        try:
            chain_ref = self
            
            def process_with_tools(input_dict: Dict[str, Any]) -> Dict[str, Any]:
                """Process question and let LLM decide if tools are needed."""
                question = input_dict.get("input", "")
                session_id = input_dict.get("session_id")
                user_info_used = False
                
                thread_id = chain_ref.conversation_memory.get_or_create_thread(session_id)
                chat_history = chain_ref.conversation_memory.get_chat_history(thread_id, session_id)
                
                is_arabic = any('\u0600' <= char <= '\u06FF' for char in question)
                response_language = "Arabic (العربية)" if is_arabic else "English"
                
                docs = chain_ref.retriever._get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in docs[:5]])
                
                user_info_tool = None
                llm_with_tools = chain_ref.llm
                if chain_ref.current_user_metadata:
                    user_info_tool = create_user_info_tool(chain_ref.current_user_metadata)
                    llm_with_tools = chain_ref.llm.bind_tools([user_info_tool])
                
                tool_instruction = chain_ref._build_tool_instruction() if user_info_tool else ""
                
                prompt_text = chain_ref._build_base_prompt(
                    context=context,
                    chat_history=chat_history,
                    question=question,
                    response_language=response_language,
                    tool_instruction=tool_instruction
                )
                
                messages = [HumanMessage(content=prompt_text)]
                response = llm_with_tools.invoke(messages)
                
                has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls
                
                if has_tool_calls and user_info_tool:
                    tool_results = []
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get("name", "")
                        if tool_name == "get_user_info":
                            result = user_info_tool.invoke({})
                            tool_results.append(f"User Info: {result}")
                            user_info_used = True
                    
                    final_prompt = chain_ref._build_final_prompt(
                        context=context,
                        chat_history=chat_history,
                        question=question,
                        tool_results=tool_results,
                        response_language=response_language
                    )
                    
                    if not chain_ref.llm:
                        return {"answer": "LLM not available", "user_info_used": False}
                    
                    final_response = chain_ref.llm.invoke([HumanMessage(content=final_prompt)])
                    answer = chain_ref._extract_content(final_response)
                    
                    chain_ref.conversation_memory.save_message(thread_id, "user", question, session_id)
                    chain_ref.conversation_memory.save_message(thread_id, "assistant", answer, session_id)
                    
                    return {"answer": answer, "user_info_used": user_info_used}
                
                answer = chain_ref._extract_content(response)
                
                chain_ref.conversation_memory.save_message(thread_id, "user", question, session_id)
                chain_ref.conversation_memory.save_message(thread_id, "assistant", answer, session_id)
                
                return {"answer": answer, "user_info_used": user_info_used}
            
            # Create chain
            self.tools_chain = RunnableLambda(process_with_tools)
        except Exception as e:
            logger.error(f"Error setting up RAG chain: {str(e)}")
            self.tools_chain = None
    
    def _extract_content(self, response: Any) -> str:
        """Extract content from LLM response."""
        if hasattr(response, 'content'):
            content = response.content
            return str(content) if content else ""
        return str(response)
    
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the chain with input."""
        if not self.tools_chain:
            return {"answer": "Chain not available", "user_info_used": False}
        return self.tools_chain.invoke(input_dict)
