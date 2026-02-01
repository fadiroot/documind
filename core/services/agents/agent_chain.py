"""Agent chain - streams from LangChain agent with retrieve_tool."""
import json
from typing import Any, Dict, List, Optional, Tuple

from core.services.memory.conversation_memory import ConversationMemory
from core.models.user import UserMetadata
from core.utils.logger import logger


def _parse_answer_and_sources(raw: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse JSON response [{"content": "...", "resource": "..."}] into formatted text and sources.
    Returns (formatted_answer, sources). On parse failure returns (raw, []).
    """
    raw = raw.strip()
    if not raw:
        return "", []

    stripped = raw
    if stripped.startswith("```"):
        for marker in ("```json", "```"):
            if stripped.startswith(marker):
                stripped = stripped[len(marker):].strip()
                break
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()

    try:
        data = json.loads(stripped)
        if not isinstance(data, list):
            return raw, []

        parts: List[str] = []
        sources: List[Dict[str, Any]] = []

        seen_resources: set[str] = set()
        for item in data:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or item.get("text", "")
            resource = item.get("resource") or item.get("source", "")
            if content:
                if resource and resource not in seen_resources:
                    seen_resources.add(resource)
                    sources.append({"document_name": resource, "resource": resource})
                if resource:
                    parts.append(f"{content}\n\n[{resource}]")
                else:
                    parts.append(str(content))

        if parts:
            formatted = "\n\n".join(parts)
            return formatted, sources
        return raw, []
    except json.JSONDecodeError:
        return raw, []


def _extract_ai_content(chunk: Any) -> Optional[str]:
    """Extract text content from a stream chunk if it's from the AI model."""
    if chunk is None:
        return None
    content = getattr(chunk, "content", None) or (chunk.get("content") if isinstance(chunk, dict) else None)
    if not content:
        return None
    msg_type = getattr(chunk, "type", None) or (chunk.get("type") if isinstance(chunk, dict) else None)
    if msg_type in ("human", "tool", "system"):
        return None
    return content if isinstance(content, str) else str(content)


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
            try:
                stream = agent.stream(inputs, stream_mode="messages")
            except (TypeError, ValueError):
                stream = agent.stream(inputs, stream_mode="updates")

            for item in stream:
                if isinstance(item, tuple) and len(item) >= 1:
                    chunk = item[0]
                else:
                    chunk = item

                if isinstance(chunk, dict):
                    for node_updates in chunk.values():
                        if isinstance(node_updates, dict) and "messages" in node_updates:
                            for msg in node_updates.get("messages", []):
                                content = _extract_ai_content(msg)
                                if content:
                                    full_answer += content
                        else:
                            content = _extract_ai_content(node_updates)
                            if content:
                                full_answer += content
                else:
                    content = _extract_ai_content(chunk)
                    if content:
                        full_answer += content

            formatted_answer, sources = _parse_answer_and_sources(full_answer)
            for i in range(0, len(formatted_answer), 512):
                yield {"type": "answer_chunk", "content": formatted_answer[i : i + 512]}

            yield {"type": "answer_end"}
            self.conversation_memory.add_exchange(session_id, question, formatted_answer)
            retrieval_score = 1.0 if sources else 0.0
            yield {"type": "complete", "sources": sources, "retrieval_score": retrieval_score}

        except Exception as e:
            logger.error(f"Agent stream error: {str(e)}", exc_info=True)
            yield {"type": "error", "content": str(e), "error": str(e)}
