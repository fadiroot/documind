from typing import Dict, Any
import json
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from core.utils.logger import logger


class QuestionRouterAgent:
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self._prompts_dir = Path(__file__).parent.parent / "prompts" / "templates"
    
    def should_retrieve_documents(self, question: str) -> Dict[str, Any]:
        if not question or not question.strip():
            return {
                "needs_retrieval": False,
                "reason": "empty_question",
                "confidence": 1.0,
                "agent_reasoning": None
            }
        
        system_prompt = self._load_prompt("question_router_prompt.promptly")
        user_prompt = f"Question: {question}\n\nDetermine if this question needs document retrieval from the knowledge base. Respond in JSON format only."
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            
            llm_content = response.content if hasattr(response, 'content') else str(response)
            if isinstance(llm_content, list):
                llm_content = " ".join(str(item) for item in llm_content)
            llm_content_str = str(llm_content).strip()
            
            json_start = llm_content_str.find('{')
            json_end = llm_content_str.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = llm_content_str[json_start:json_end]
                try:
                    result = json.loads(json_str)
                    return {
                        "needs_retrieval": bool(result.get("needs_retrieval", True)),
                        "reason": str(result.get("reason", "llm_determined")),
                        "confidence": float(result.get("confidence", 0.7)),
                        "agent_reasoning": llm_content_str
                    }
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON from LLM response: {json_str}")
            
            llm_lower = llm_content_str.lower()
            if any(word in llm_lower for word in ['no', 'not', "doesn't", "don't", 'لا', 'ليس', 'false']):
                return {
                    "needs_retrieval": False,
                    "reason": "llm_determined_no_retrieval",
                    "confidence": 0.7,
                    "agent_reasoning": llm_content_str
                }
            else:
                return {
                    "needs_retrieval": True,
                    "reason": "llm_determined_retrieval_needed",
                    "confidence": 0.7,
                    "agent_reasoning": llm_content_str
                }
                
        except Exception as e:
            logger.error(f"Error in QuestionRouterAgent: {str(e)}")
            return {
                "needs_retrieval": True,
                "reason": "error_fallback",
                "confidence": 0.5,
                "agent_reasoning": f"Error occurred: {str(e)}"
            }
    
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
