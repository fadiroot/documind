import json
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
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
from core.models.answer import AnswerResponse, AnswerItem
from core.utils.logger import logger


class AgentChain:
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
        self._prompts_dir = Path(__file__).parent.parent / "prompts" / "templates"
    
    def set_user(self, user: Optional[UserMetadata]):
        self.user = user
    
    def stream(self, input_dict: Dict[str, Any]):
        question = input_dict.get("input", "")
        session_id = input_dict.get("session_id")
        
        if not question:
            yield {"type": "error", "content": "No question", "error": "empty"}
            return
        
        needs_docs = self._needs_documents(question)
        
        if needs_docs:
            yield from self._answer_with_docs(question, session_id)
        else:
            yield from self._answer_general(question, session_id)
    
    def _needs_documents(self, question: str) -> bool:
        if not self.router:
            return True
        
        try:
            decision = self.router.should_retrieve_documents(question)
            return decision.get("needs_retrieval", True)
        except Exception as e:
            logger.warning(f"Router error: {str(e)}")
            return True
    
    def _load_prompt(self, filename: str) -> str:
        prompt_path = self._prompts_dir / filename
        try:
            content = prompt_path.read_text(encoding="utf-8")
            # Extract content after YAML frontmatter (after ---\n---\n)
            if "---\n" in content:
                parts = content.split("---\n", 2)
                if len(parts) >= 3:
                    return parts[2].strip()
            return content.strip()
        except Exception as e:
            logger.warning(f"Failed to load prompt {filename}: {str(e)}")
            return ""
    
    def _answer_general(self, question: str, session_id: Optional[str]):
        yield {"type": "status", "content": "Processing..."}
        
        history = self.conversation_memory.get_summary(session_id)
        system = self._load_prompt("general_system_prompt.promptly")
        
        if history and history != "No previous conversation.":
            user = f"{history}\n\nQuestion: {question}\n\nAnswer naturally."
        else:
            user = f"Question: {question}"
        
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
                "sources": [],
                "retrieval_score": 0.0
            }
        except Exception as e:
            logger.error(f"General answer error: {str(e)}")
            yield {"type": "error", "content": str(e)}
    
    def _answer_with_docs(self, question: str, session_id: Optional[str]):
        try:
            yield {"type": "status", "content": "Retrieving documents..."}
            docs = self.retrieval_service.retrieve(
                query=question,
                top_k=5,
                min_score=self.min_retrieval_score
            )
            
            if not docs.has_results():
                logger.warning("No documents found")
                fallback = ErrorHandler.handle_no_documents(question)
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
            
            result = yield from self._generate(messages)
            
            if result and result.get("content"):
                self.conversation_memory.add_exchange(session_id, question, result["content"])
            
            complete_data = {
                "type": "complete",
                "sources": self._build_sources(docs),
                "retrieval_score": docs.get_average_score()
            }
            
            if result and result.get("structured"):
                complete_data["structured_response"] = result["structured"]
                complete_data["resources"] = result.get("resources", [])
                
                try:
                    structured_data = result["structured"]
                    if structured_data and "answers" in structured_data:
                        answer_response = AnswerResponse(**structured_data)
                        print("\n" + "="*80)
                        print("AnswerResponse after streaming:")
                        print("="*80)
                        print(answer_response.model_dump_json(indent=2))
                        print("="*80 + "\n")
                except Exception as e:
                    logger.warning(f"Failed to print AnswerResponse: {str(e)}")
            
            yield complete_data
        
        except Exception as e:
            logger.error(f"RAG error: {str(e)}", exc_info=True)
            fallback = ErrorHandler.handle_llm_error(e, question)
            yield {"type": "error", "content": fallback, "error": str(e)}
    
    def _generate(self, messages: List):
        full_answer = ""
        
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
        
        parsed_response = self._parse_json_response(full_answer)
        return parsed_response
    
    def _parse_json_response(self, raw_response: str) -> Dict[str, Any]:
        try:
            json_text = raw_response.strip()
            
            # Remove markdown code blocks if present
            json_text = re.sub(r'^```(?:json)?\s*\n', '', json_text, flags=re.MULTILINE)
            json_text = re.sub(r'\n```\s*$', '', json_text, flags=re.MULTILINE)
            json_text = json_text.strip()
            
            data = json.loads(json_text)
            
            if isinstance(data, list):
                answer_items = []
                for item in data:
                    if isinstance(item, dict) and "content" in item and "resource" in item:
                        answer_items.append(AnswerItem(
                            content=item["content"],
                            resource=item["resource"]
                        ))
                
                if answer_items:
                    answer_response = AnswerResponse(answers=answer_items)
                    return {
                        "content": "\n\n".join([item.content for item in answer_items]),
                        "resources": [item.resource for item in answer_items],
                        "structured": answer_response.model_dump()
                    }
            
            logger.warning("Failed to parse JSON response, using raw content")
            return {
                "content": raw_response,
                "resources": [],
                "structured": None
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {str(e)}, using raw content")
            return {
                "content": raw_response,
                "resources": [],
                "structured": None
            }
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return {
                "content": raw_response,
                "resources": [],
                "structured": None
            }
    
    def _format_user_info(self) -> str:
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
        if hasattr(response, 'content'):
            return str(response.content) if response.content else ""
        return str(response)
    
    def _build_sources(self, docs: RetrievalResult) -> List[Dict[str, Any]]:
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
