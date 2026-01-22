"""Improved conversation memory with summarization support."""
from typing import Dict, Optional, Any
from dataclasses import dataclass

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


@dataclass
class ConversationSummary:
    """Conversation summary for context."""
    summary: str
    recent_questions: list[str]
    recent_answers: list[str]
    total_exchanges: int


class ConversationMemory:
    """
    Improved conversation memory with summarization.
    
    Design principles:
    - Stateless: Legal answers don't need long conversation context
    - Efficient: Summarize instead of full history
    - Sliding window: Keep only recent exchanges
    """
    
    def __init__(self, max_recent_exchanges: int = 3, enable_summarization: bool = False):
        self.azure_project_client: Optional[Any] = None
        self.azure_agent_id: Optional[str] = None
        self.session_thread_map: Dict[str, str] = {}
        self.in_memory_history: Dict[str, list[Dict[str, str]]] = {}
        self.summaries: Dict[str, ConversationSummary] = {}
        self.max_recent_exchanges = max_recent_exchanges
        self.enable_summarization = enable_summarization
        
        if AZURE_AI_PROJECTS_AVAILABLE and settings.AZURE_PROJECT_ENDPOINT and settings.AZURE_AI_AGENT_ID:
            try:
                credential = DefaultAzureCredential()  # type: ignore
                self.azure_project_client = AIProjectClient(  # type: ignore
                    credential=credential,
                    endpoint=settings.AZURE_PROJECT_ENDPOINT
                )
                self.azure_agent_id = settings.AZURE_AI_AGENT_ID
            except Exception as e:
                logger.warning(f"Failed to initialize Azure AI Agents: {str(e)}. Using in-memory fallback.")
                self.azure_project_client = None
        elif not AZURE_AI_PROJECTS_AVAILABLE:
            logger.warning("Azure AI Projects package not available")
    
    def get_or_create_thread(self, session_id: Optional[str]) -> Optional[str]:
        """Get or create Azure AI Agents thread for a session."""
        if not session_id or not self.azure_project_client or not self.azure_agent_id:
            return None
        
        try:
            if session_id in self.session_thread_map:
                return self.session_thread_map[session_id]
            
            thread = self.azure_project_client.agents.threads.create()
            self.session_thread_map[session_id] = thread.id
            return thread.id
        except Exception as e:
            logger.error(f"Error creating/getting Azure thread: {str(e)}")
            return None
    
    def get_summary(self, session_id: Optional[str]) -> Optional[str]:
        """
        Get conversation summary instead of full history.
        
        Returns:
            Summary string or None if no conversation exists
        """
        if not session_id:
            return None
        
        # Check if we have a summary
        if session_id in self.summaries:
            summary_obj = self.summaries[session_id]
            if summary_obj.summary:
                return summary_obj.summary
        
        # Build summary from recent exchanges
        recent = self._get_recent_exchanges(session_id, self.max_recent_exchanges)
        if not recent:
            return None
        
        # Extract questions from recent exchanges (stored as {"role": "user", "content": question})
        questions = [msg.get('content', '') for msg in recent if msg.get('role') == 'user']
        if questions:
            # Return last 2 questions as summary
            last_questions = questions[-2:] if len(questions) >= 2 else questions
            summary = f"Previous questions: {'; '.join(last_questions)}"
            return summary
        
        return None
    
    def add_exchange(
        self,
        session_id: Optional[str],
        question: str,
        answer: str
    ) -> None:
        """
        Add a Q&A exchange and update summary.
        
        Args:
            session_id: Session identifier
            question: User question
            answer: Assistant answer
        """
        if not session_id:
            return
        
        # Save to Azure thread if available
        thread_id = self.get_or_create_thread(session_id)
        if thread_id and self.azure_project_client:
            try:
                self.azure_project_client.agents.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=question
                )
                self.azure_project_client.agents.messages.create(
                    thread_id=thread_id,
                    role="assistant",
                    content=answer
                )
            except Exception as e:
                logger.warning(f"Error saving to Azure thread: {str(e)}, using in-memory only")
        
        # Save to in-memory (always, even if Azure fails)
        if session_id not in self.in_memory_history:
            self.in_memory_history[session_id] = []
        
        self.in_memory_history[session_id].append({"role": "user", "content": question})
        self.in_memory_history[session_id].append({"role": "assistant", "content": answer})
        
        # Keep only recent exchanges (sliding window)
        max_messages = self.max_recent_exchanges * 2  # Each exchange = 2 messages (user + assistant)
        if len(self.in_memory_history[session_id]) > max_messages:
            self.in_memory_history[session_id] = self.in_memory_history[session_id][-max_messages:]
        
        # Update summary
        self._update_summary(session_id)
    
    def _get_recent_exchanges(self, session_id: str, count: int) -> list[Dict[str, str]]:
        """Get recent exchanges from memory."""
        if session_id not in self.in_memory_history:
            return []
        
        history = self.in_memory_history[session_id]
        # Get last 'count' exchanges (each exchange = 2 messages: user + assistant)
        num_messages = count * 2
        return history[-num_messages:] if len(history) > num_messages else history
    
    def _update_summary(self, session_id: str) -> None:
        """Update conversation summary."""
        if session_id not in self.in_memory_history:
            return
        
        history = self.in_memory_history[session_id]
        max_messages = self.max_recent_exchanges * 2
        recent = history[-max_messages:] if len(history) > max_messages else history
        
        questions = []
        answers = []
        
        for msg in recent:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                questions.append(content)
            elif role == "assistant":
                answers.append(content)
        
        # Build summary from recent questions
        summary_text = ""
        if questions:
            # Get last 2 questions, truncate if too long
            recent_questions = questions[-2:] if len(questions) >= 2 else questions
            question_summaries = []
            for q in recent_questions:
                if len(q) > 50:
                    question_summaries.append(q[:50] + "...")
                else:
                    question_summaries.append(q)
            summary_text = f"Previous questions: {'; '.join(question_summaries)}"
        
        self.summaries[session_id] = ConversationSummary(
            summary=summary_text,
            recent_questions=questions,
            recent_answers=answers,
            total_exchanges=len(questions)
        )
    
    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        cleared = False
        
        if session_id in self.session_thread_map:
            del self.session_thread_map[session_id]
            cleared = True
        
        if session_id in self.in_memory_history:
            del self.in_memory_history[session_id]
            cleared = True
        
        if session_id in self.summaries:
            del self.summaries[session_id]
            cleared = True
        
        return cleared
    
    # Legacy method for backward compatibility
    def get_chat_history(self, thread_id: Optional[str], session_id: Optional[str] = None) -> str:
        """Legacy method: returns summary instead of full history."""
        summary = self.get_summary(session_id)
        return summary or "No previous conversation history."
