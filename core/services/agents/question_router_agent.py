"""LangChain agent to determine if a question needs document retrieval."""
from typing import Dict, Any, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from core.utils.logger import logger


@tool
def needs_document_retrieval(question: str) -> Dict[str, Any]:
    """
    Determine if a question requires document retrieval from the knowledge base.
    
    Questions that NEED retrieval:
    - Questions about legal articles, regulations, policies
    - Questions about employee rights, salaries, leave entitlements
    - Questions about specific documents, procedures, or manuals
    - Questions containing keywords like: مادة, article, حقوق, rights, راتب, salary, إجازة, leave
    
    Questions that DON'T need retrieval:
    - Greetings: مرحبا, hello, hi, how are you
    - General questions: من أنت, who are you, what can you do
    - Very short questions (1-3 characters)
    
    Args:
        question: The user's question
        
    Returns:
        Dictionary with 'needs_retrieval' (bool) and 'reason' (str)
    """
    question_lower = question.lower().strip()
    question_clean = question.strip()
    
    # Very short questions are likely greetings
    if len(question_clean) <= 3:
        return {
            "needs_retrieval": False,
            "reason": "too_short",
            "confidence": 0.9
        }
    
    # Document-related keywords (Arabic and English)
    document_keywords = [
        # Arabic
        'مادة', 'باب', 'فصل', 'حق', 'حقوق', 'راتب', 'إجازة', 'تعيين', 'ترقية',
        'مهندس', 'موظف', 'عامل', 'متعاقد', 'قانون', 'نظام', 'لائحة', 'دليل',
        'مستند', 'وثيقة', 'إجراء', 'سياسة', 'قاعدة', 'تنظيم',
        # English
        'article', 'law', 'regulation', 'policy', 'right', 'salary', 'leave',
        'employee', 'engineer', 'contractor', 'appointment', 'promotion',
        'document', 'procedure', 'manual', 'guide', 'regulation', 'rule'
    ]
    
    # Check for document keywords
    has_document_keywords = any(
        keyword.lower() in question_lower or keyword in question_clean
        for keyword in document_keywords
    )
    
    if has_document_keywords:
        return {
            "needs_retrieval": True,
            "reason": "contains_document_keywords",
            "confidence": 0.95
        }
    
    # Greeting patterns
    greeting_patterns = [
        'مرحبا', 'السلام', 'أهلا', 'هاي', 'صباح', 'مساء', 'كيف حالك',
        'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon'
    ]
    
    is_greeting = any(
        greeting in question_lower[:20]  # Check first 20 chars
        for greeting in greeting_patterns
    )
    
    if is_greeting:
        return {
            "needs_retrieval": False,
            "reason": "greeting",
            "confidence": 0.85
        }
    
    # General question patterns
    general_patterns = [
        'من أنت', 'ما اسمك', 'كيف حالك', 'ماذا تفعل', 'ماذا يمكنك',
        'who are you', 'what is your name', 'what can you do', 'what do you do'
    ]
    
    is_general = any(
        pattern in question_lower
        for pattern in general_patterns
    )
    
    if is_general:
        return {
            "needs_retrieval": False,
            "reason": "general_question",
            "confidence": 0.8
        }
    
    # Default: assume it needs retrieval if unclear
    return {
        "needs_retrieval": True,
        "reason": "default_assumption",
        "confidence": 0.6
    }


class QuestionRouterAgent:
    """
    LangChain agent that determines if a question needs document retrieval.
    
    Uses LLM reasoning with a tool to make intelligent decisions about
    whether to retrieve documents or answer directly.
    """
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.tool = needs_document_retrieval
        self._setup_agent()
    
    def _setup_agent(self):
        """Set up the agent with the routing tool."""
        self.agent_llm = self.llm.bind_tools([self.tool])
    
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

استخدم الأداة المتاحة لتحديد ما إذا كان السؤال يحتاج إلى استرجاع الوثائق.

السؤال يحتاج إلى استرجاع الوثائق إذا كان:
- يتعلق بالمواد القانونية، الأنظمة، اللوائح
- يسأل عن حقوق الموظفين، الرواتب، الإجازات
- يحتوي على كلمات مثل: مادة، حقوق، راتب، إجازة، قانون، نظام

السؤال لا يحتاج إلى استرجاع الوثائق إذا كان:
- تحية: مرحبا، السلام عليكم
- سؤال عام: من أنت، ما اسمك
- سؤال قصير جداً (1-3 أحرف)

أجب بشكل واضح ومباشر."""
        else:
            system_prompt = """You are an intelligent agent whose task is to determine if a question needs to search the knowledge base of legal and regulatory documents.

Use the available tool to determine if the question needs document retrieval.

A question NEEDS retrieval if it:
- Relates to legal articles, regulations, policies
- Asks about employee rights, salaries, leave entitlements
- Contains keywords like: article, rights, salary, leave, law, regulation

A question DOESN'T need retrieval if it:
- Is a greeting: hello, hi, how are you
- Is a general question: who are you, what is your name
- Is very short (1-3 characters)

Answer clearly and directly."""
        
        user_prompt = f"Question: {question}\n\nDetermine if this question needs document retrieval from the knowledge base."
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            # Get LLM response with tool call
            response = self.agent_llm.invoke(messages)
            
            # Check if tool was called
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                if tool_call.get('name') == 'needs_document_retrieval':
                    # Extract tool result
                    tool_args = tool_call.get('args', {})
                    result = self.tool.invoke(tool_args)
                    
                    return {
                        "needs_retrieval": result.get("needs_retrieval", True),
                        "reason": result.get("reason", "unknown"),
                        "confidence": result.get("confidence", 0.5),
                        "agent_reasoning": response.content if hasattr(response, 'content') else None
                    }
            
            # If no tool call, use LLM's direct response to infer
            llm_content = response.content if hasattr(response, 'content') else str(response)
            # Handle case where content might be a list
            if isinstance(llm_content, list):
                llm_content = " ".join(str(item) for item in llm_content)
            llm_content_str = str(llm_content)
            llm_lower = llm_content_str.lower()
            
            # Try to infer from LLM response
            if any(word in llm_lower for word in ['no', 'not', 'doesn\'t', 'don\'t', 'لا', 'ليس']):
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
            # Fallback: use tool directly without LLM
            result = self.tool.invoke({"question": question})
            return {
                "needs_retrieval": result.get("needs_retrieval", True),
                "reason": result.get("reason", "fallback"),
                "confidence": result.get("confidence", 0.5),
                "agent_reasoning": f"Error occurred: {str(e)}"
            }
