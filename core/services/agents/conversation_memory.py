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
        
        if AZURE_AI_PROJECTS_AVAILABLE and settings.AZURE_PROJECT_ENDPOINT and settings.AZURE_AI_AGENT_ID:
            try:
                self.azure_project_client = AIProjectClient(
                    credential=DefaultAzureCredential(),  # type: ignore
                    endpoint=settings.AZURE_PROJECT_ENDPOINT
                )  # type: ignore
                self.azure_agent_id = settings.AZURE_AI_AGENT_ID
                logger.info(f"Azure AI Agents initialized with agent ID: {self.azure_agent_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure AI Agents: {str(e)}")
                self.azure_project_client = None
        else:
            if not AZURE_AI_PROJECTS_AVAILABLE:
                logger.warning("Azure AI Projects package not available")
            else:
                logger.warning("Azure AI Agents not configured (AZURE_PROJECT_ENDPOINT or AZURE_AI_AGENT_ID missing)")
    
    def get_or_create_thread(self, session_id: Optional[str]) -> Optional[str]:
        """Get or create Azure AI Agents thread for a session."""
        if not session_id or not self.azure_project_client or not self.azure_agent_id:
            return None
        
        try:
            # Check if thread already exists for this session
            if session_id in self.session_thread_map:
                return self.session_thread_map[session_id]
            
            # Create new thread
            thread = self.azure_project_client.agents.threads.create()
            self.session_thread_map[session_id] = thread.id
            logger.info(f"Created new Azure thread {thread.id} for session {session_id}")
            return thread.id
        except Exception as e:
            logger.error(f"Error creating/getting Azure thread: {str(e)}")
            return None
    
    def get_chat_history(self, thread_id: Optional[str]) -> str:
        """Get chat history from Azure AI Agents thread."""
        if not thread_id or not self.azure_project_client:
            return "No previous conversation history."
        
        try:
            messages = self.azure_project_client.agents.messages.list(
                thread_id=thread_id,
                order=ListSortOrder.ASCENDING # type: ignore
            )
            
            history_parts = []
            for message in list(messages)[-10:]:  # Keep last 10 messages (5 turns)
                if message.text_messages:
                    role = message.role
                    text = message.text_messages[-1].text.value
                    history_parts.append(f"{role.capitalize()}: {text}")
            
            return "\n".join(history_parts) if history_parts else "No previous conversation history."
        except Exception as e:
            logger.error(f"Error retrieving chat history from thread: {str(e)}")
            return "No previous conversation history."
    
    def save_message(self, thread_id: Optional[str], role: str, content: str) -> bool:
        """Save a message to Azure AI Agents thread."""
        if not thread_id or not self.azure_project_client:
            return False
        
        try:
            self.azure_project_client.agents.messages.create(
                thread_id=thread_id,
                role=role,
                content=content
            )
            return True
        except Exception as e:
            logger.error(f"Error saving message to thread: {str(e)}")
            return False
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a specific session by removing thread mapping.
        
        Args:
            session_id: Session ID to clear
        
        Returns:
            True if session was cleared, False if session didn't exist
        """
        if session_id in self.session_thread_map:
            del self.session_thread_map[session_id]
            logger.info(f"Cleared conversation thread mapping for session: {session_id}")
            return True
        return False
