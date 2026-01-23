"""LangChain agent to determine if a question needs document retrieval."""
from typing import Dict, Any
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from core.utils.logger import logger


class QuestionRouterAgent:
    """
    LangChain agent that determines if a question needs document retrieval.
    
    Uses LLM reasoning to make intelligent decisions about
    whether to retrieve documents or answer directly.
    """
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    def should_retrieve_documents(self, question: str, language: str = "english") -> Dict[str, Any]:
        """
        Determine if a question needs document retrieval.
        
        Args:
            question: The user's question
            language: Language of the question (arabic/english)
            
        Returns:
            Dictionary with:
            - needs_retrieval (bool): Whether to retrieve documents
            - reason (str): Reason for the decision
            - confidence (float): Confidence score (0-1)
            - agent_reasoning (str): LLM's reasoning (if available)
        """
        if not question or not question.strip():
            return {
                "needs_retrieval": False,
                "reason": "empty_question",
                "confidence": 1.0,
                "agent_reasoning": None
            }
        
        # Build system prompt based on language
        if language.lower() == "arabic":
            system_prompt = """أنت وكيل ذكي مهمتك تحديد ما إذا كان السؤال يحتاج إلى البحث في قاعدة المعرفة من الوثائق القانونية والتنظيمية.

السؤال يحتاج إلى استرجاع الوثائق إذا كان:
- يتعلق بالمواد القانونية، الأنظمة، اللوائح، السياسات
- يسأل عن حقوق الموظفين، الرواتب، الإجازات، التعيينات، الترقي
- يتطلب معلومات محددة من الوثائق الرسمية

السؤال لا يحتاج إلى استرجاع الوثائق إذا كان:
- تحية بسيطة: مرحبا، السلام عليكم، أهلاً
- سؤال عام عنك: من أنت، ما اسمك، ماذا تفعل
- سؤال قصير جداً (1-3 أحرف)

أجب بصيغة JSON فقط بهذا الشكل:
{
  "needs_retrieval": true/false,
  "reason": "سبب القرار",
  "confidence": 0.0-1.0
}"""
        else:
            system_prompt = """You are an intelligent agent whose task is to determine if a question needs to search the knowledge base of legal and regulatory documents.

A question NEEDS retrieval if it:
- Relates to legal articles, regulations, policies, procedures
- Asks about employee rights, salaries, leave entitlements, appointments, promotions
- Requires specific information from official documents

A question DOESN'T need retrieval if it:
- Is a simple greeting: hello, hi, greetings
- Is a general question about you: who are you, what is your name, what can you do
- Is very short (1-3 characters)

Respond ONLY in JSON format:
{
  "needs_retrieval": true/false,
  "reason": "reason for decision",
  "confidence": 0.0-1.0
}"""
        
        user_prompt = f"Question: {question}\n\nDetermine if this question needs document retrieval from the knowledge base. Respond in JSON format only."
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            # Get LLM response
            response = self.llm.invoke(messages)
            
            # Extract content from response
            llm_content = response.content if hasattr(response, 'content') else str(response)
            if isinstance(llm_content, list):
                llm_content = " ".join(str(item) for item in llm_content)
            llm_content_str = str(llm_content).strip()
            
            # Try to parse JSON from response
            # Look for JSON block in the response
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
            
            # Fallback: try to infer from text response
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
            # Fallback: default to retrieval needed
            return {
                "needs_retrieval": True,
                "reason": "error_fallback",
                "confidence": 0.5,
                "agent_reasoning": f"Error occurred: {str(e)}"
            }
