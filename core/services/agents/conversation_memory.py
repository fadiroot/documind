"""Conversation memory management using Azure AI Agents threads."""
from typing import Dict, Optional, Any

try:
    from azure.ai.projects import AIProjectClient  # type: ignore
    from azure.identity import DefaultAzureCredential  # type: ignore
    from azure.ai.agents.models import ListSortOrder  # type: ignore
    AZURE_AI_PROJECTS_AVAILABLE = True
except ImportError:
    AIProjectClient = None  # type: ignore
    DefaultAzureCredential = None  # type: ignore
    ListSortOrder = None  # type: ignore
    AZURE_AI_PROJECTS_AVAILABLE = False

from core.utils.logger import logger
from app.config import settings


class ConversationMemory:
    """Manages conversation memory using Azure AI Agents threads."""
    
    def __init__(self):
        self.azure_project_client: Optional[Any] = None
        self.azure_agent_id: Optional[str] = None
        self.session_thread_map: Dict[str, str] = {}
        self.in_memory_history: Dict[str, list[Dict[str, str]]] = {}
        
        if AZURE_AI_PROJECTS_AVAILABLE and settings.AZURE_PROJECT_ENDPOINT and settings.AZURE_AI_AGENT_ID:
            try:
                credential = DefaultAzureCredential()  # type: ignore
                self.azure_project_client = AIProjectClient(  # type: ignore
                    credential=credential,
                    endpoint=settings.AZURE_PROJECT_ENDPOINT
                )
                self.azure_agent_id = settings.AZURE_AI_AGENT_ID
            except Exception as e:
                logger.warning(f"Failed to initialize Azure AI Agents: {str(e)}. Thread management disabled. Using in-memory fallback.")
                self.azure_project_client = None
        elif not AZURE_AI_PROJECTS_AVAILABLE:
            logger.warning("Azure AI Projects package not available")
      
    def get_or_create_thread(self, session_id: Optional[str]) -> Optional[str]:
        """Get or create Azure AI Agents thread for a session."""
        if not session_id or not self.azure_project_client or not self.azure_agent_id:
            logger.info(f"[THREAD] Session {session_id}: No Azure thread (fallback to in-memory)")
            return None
        
        try:
            if session_id in self.session_thread_map:
                thread_id = self.session_thread_map[session_id]
                logger.info(f"[THREAD] Session {session_id}: Using existing thread {thread_id}")
                return thread_id
            
            thread = self.azure_project_client.agents.threads.create()
            self.session_thread_map[session_id] = thread.id
            logger.info(f"[THREAD] Session {session_id}: Created new thread {thread.id}")
            return thread.id
        except Exception as e:
            error_msg = str(e)
            if "expired" not in error_msg.lower() and "AADSTS7000222" not in error_msg and "authentication failed" not in error_msg.lower():
                logger.error(f"Error creating/getting Azure thread: {error_msg}")
            return None
    
    def get_chat_history(self, thread_id: Optional[str], session_id: Optional[str] = None) -> str:
        """Get chat history from Azure AI Agents thread, falls back to in-memory history."""
        history_parts = []
        
        if thread_id and self.azure_project_client and ListSortOrder:
            try:
                messages = self.azure_project_client.agents.messages.list(
                    thread_id=thread_id,
                    order=ListSortOrder.ASCENDING
                )
                for message in list(messages)[-10:]:
                    if message.text_messages:
                        role = message.role
                        text = message.text_messages[-1].text.value
                        prefix = "السؤال السابق" if role == "user" else "الإجابة السابقة"
                        history_parts.append(f"{prefix}: {text}")
            except Exception as e:
                logger.error(f"Error retrieving chat history: {str(e)}")
        
        if not history_parts and session_id and session_id in self.in_memory_history:
            for msg in self.in_memory_history[session_id][-10:]:
                role = msg.get("role", "")
                text = msg.get("content", "")
                prefix = "السؤال السابق" if role == "user" else "الإجابة السابقة"
                history_parts.append(f"{prefix}: {text}")
        
        if history_parts:
            formatted_history = "\n".join(history_parts)
            return f"{formatted_history}\n\nاستخدم هذا السياق لفهم الإشارات في السؤال الحالي."
        
        return "No previous conversation history."
    
    def save_message(self, thread_id: Optional[str], role: str, content: str, session_id: Optional[str] = None) -> bool:
        """Save message to Azure thread, falls back to in-memory storage."""
        if thread_id and self.azure_project_client:
            try:
                self.azure_project_client.agents.messages.create(
                    thread_id=thread_id,
                    role=role,
                    content=content
                )
            except Exception as e:
                logger.error(f"Error saving message to thread: {str(e)}")
        
        if session_id:
            if session_id not in self.in_memory_history:
                self.in_memory_history[session_id] = []
            self.in_memory_history[session_id].append({"role": role, "content": content})
            if len(self.in_memory_history[session_id]) > 20:
                self.in_memory_history[session_id] = self.in_memory_history[session_id][-20:]
        
        return True
    
    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        cleared = False
        if session_id in self.session_thread_map:
            del self.session_thread_map[session_id]
            cleared = True
        if session_id in self.in_memory_history:
            del self.in_memory_history[session_id]
            cleared = True
        return cleared
