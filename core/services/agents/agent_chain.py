"""Agent chain - streams from LangChain agent with retrieve_tool."""
from typing import Any, Dict, Optional

from core.services.memory.conversation_memory import ConversationMemory
from core.models.user import UserMetadata
from core.utils.logger import logger


class AgentChain:
    def __init__(
        self,
        conversation_memory: ConversationMemory,
        min_retrieval_score: float = 0.3,
    ):
        self.conversation_memory = conversation_memory
        self.min_retrieval_score = min_retrieval_score
        self.user: Optional[UserMetadata] = None
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            from core.services.agents.agent import create_documind_agent
            self._agent = create_documind_agent(min_retrieval_score=self.min_retrieval_score)
        return self._agent

    def set_user(self, user: Optional[UserMetadata]):
        self.user = user

    def stream(self, input_dict: Dict[str, Any]):
        question = input_dict.get("input", "")
        session_id = input_dict.get("session_id")

        if not question:
            yield {"type": "error", "content": "No question", "error": "empty"}
            return

        try:
            agent = self._get_agent()
            
            # Build messages with conversation history
            messages: list[Dict[str, str]] = []
            
            # Get recent conversation history from memory
            if session_id and session_id in self.conversation_memory.in_memory_history:
                history = self.conversation_memory.in_memory_history[session_id]
                # Include all recent exchanges (already limited by sliding window in memory)
                for msg in history:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current question
            messages.append({"role": "user", "content": question})
            
            inputs: Any = {"messages": messages}

            yield {"type": "status", "content": "Processing..."}
            yield {"type": "answer_start"}

            full_answer = ""
            for chunk in agent.stream(inputs, stream_mode="updates"):
                if isinstance(chunk, dict):
                    for node_updates in chunk.values():
                        if isinstance(node_updates, dict) and "messages" in node_updates:
                            for msg in node_updates.get("messages", []):
                                c = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
                                if c:
                                    content = str(c)
                                    full_answer += content
                                    yield {"type": "answer_chunk", "content": content}
                        else:
                            c = getattr(node_updates, "content", None)
                            if c:
                                content = str(c)
                                full_answer += content
                                yield {"type": "answer_chunk", "content": content}

            yield {"type": "answer_end"}
            self.conversation_memory.add_exchange(session_id, question, full_answer)
            yield {"type": "complete", "sources": [], "retrieval_score": 0.0}

        except Exception as e:
            logger.error(f"Agent stream error: {str(e)}", exc_info=True)
            yield {"type": "error", "content": str(e), "error": str(e)}
