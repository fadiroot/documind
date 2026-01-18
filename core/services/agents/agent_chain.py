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
                
                # Get or create Azure thread for conversation memory
                thread_id = chain_ref.conversation_memory.get_or_create_thread(session_id)
                chat_history = chain_ref.conversation_memory.get_chat_history(thread_id)
                
                # Detect question language (simple heuristic)
                is_arabic = any('\u0600' <= char <= '\u06FF' for char in question)
                response_language = "Arabic (العربية)" if is_arabic else "English"
                
                # Get documents first (always retrieve for context)
                docs = chain_ref.retriever._get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in docs[:5]])
                
                # Create user info tool with current metadata and bind to LLM
                user_info_tool = None
                llm_with_tools = chain_ref.llm
                if chain_ref.current_user_metadata:
                    user_info_tool = create_user_info_tool(chain_ref.current_user_metadata)
                    tools = [user_info_tool]
                    llm_with_tools = chain_ref.llm.bind_tools(tools)
                
                # Build prompt - LLM decides if user info is needed via tool
                tool_instruction = ""
                if user_info_tool:
                    tool_instruction = "\nAvailable tool:\n- get_user_info: Use when questions ask about personal information (salary, rank, position, cadre, benefits, etc.)\n"
                
                prompt_text = f"""You are a helpful AI assistant that answers questions based on retrieved documents.{tool_instruction}

Document Context:
{context}

Conversation History:
{chat_history}

Current Question: {question}

IMPORTANT LANGUAGE INSTRUCTION:
- The question is in {response_language}
- You MUST respond in the SAME LANGUAGE as the question
- If the question is in Arabic, respond entirely in Arabic
- If the question is in English, respond entirely in English
- Do not mix languages in your response

Instructions:
1. Consider the conversation history when answering - the user may be referring to previous questions or context.
2. Analyze the question and decide if user information is needed. If the question is about personal matters, use the get_user_info tool.
3. Format your answer in a clear, organized structure:
   - Use clear headings and subheadings
   - Use bullet points or numbered lists for multiple items
   - Group related information together
   - Keep paragraphs concise and focused
   - Use proper spacing between sections
4. Make the answer easy to read and navigate.
5. RESPOND IN {response_language.upper()} - Match the question's language exactly."""
                
                # Call LLM with tools - it will decide if tools are needed
                messages = [HumanMessage(content=prompt_text)]
                response = llm_with_tools.invoke(messages)
                
                # If LLM wants to use tools, execute them
                if hasattr(response, 'tool_calls') and response.tool_calls and user_info_tool:
                    tool_results = []
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get("name", "")
                        if tool_name == "get_user_info":
                            result = user_info_tool.invoke({})
                            tool_results.append(f"User Info: {result}")
                            user_info_used = True
                    
                    # Call LLM again with tool results
                    # Detect language again for final prompt
                    is_arabic = any('\u0600' <= char <= '\u06FF' for char in question)
                    response_language = "Arabic (العربية)" if is_arabic else "English"
                    
                    final_prompt = f"""Based on the following information, provide a well-organized and structured answer.

Document Context:
{context}

Conversation History:
{chat_history}

Tool Results:
{chr(10).join(tool_results)}

Question: {question}

CRITICAL LANGUAGE REQUIREMENT:
- The question is in {response_language}
- You MUST respond ENTIRELY in {response_language.upper()}
- If question is Arabic, answer in Arabic only
- If question is English, answer in English only
- Do NOT mix languages

Answer Formatting Requirements:
1. Consider the conversation history - the user may be continuing a previous discussion
2. Structure your answer with clear sections and subsections
3. Use headings (## or ###) to organize major topics
4. Use bullet points (-) or numbered lists (1., 2., 3.) for multiple items
5. Group related information together logically
6. Keep paragraphs concise (2-3 sentences max)
7. Use proper spacing between sections for readability
8. Highlight important information when relevant
9. End with a brief summary or next steps if applicable

Provide a clear, accurate, and well-organized answer IN {response_language.upper()}:"""
                    
                    if not chain_ref.llm:
                        return {"answer": "LLM not available", "user_info_used": False}
                    
                    final_response = chain_ref.llm.invoke([HumanMessage(content=final_prompt)])
                    answer = chain_ref._extract_content(final_response)
                    
                    # Save conversation to Azure thread
                    if thread_id:
                        chain_ref.conversation_memory.save_message(thread_id, "user", question)
                        chain_ref.conversation_memory.save_message(thread_id, "assistant", answer)
                    
                    return {"answer": answer, "user_info_used": user_info_used}
                
                # No tools called - direct answer
                answer = chain_ref._extract_content(response)
                
                # Save conversation to Azure thread
                if thread_id:
                    chain_ref.conversation_memory.save_message(thread_id, "user", question)
                    chain_ref.conversation_memory.save_message(thread_id, "assistant", answer)
                
                return {"answer": answer, "user_info_used": user_info_used}
            
            # Create chain
            self.tools_chain = RunnableLambda(process_with_tools)
            
            logger.info("RAG chain with tools initialized successfully")
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
