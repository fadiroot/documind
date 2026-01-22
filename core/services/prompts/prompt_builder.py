"""Prompt builder for RAG system - formats context and questions."""
from typing import List, Optional, Tuple
from langchain_core.documents import Document

from core.services.utils.metadata_utils import build_resource_path


class PromptBuilder:
    """Simple prompt builder - formats context and question."""
    
    def build_system_prompt(self, language: str = "english") -> str:
        """Build system prompt."""
        if "arabic" in language.lower():
            return """أنت مساعد متخصص في الإجابة على الأسئلة حول الوثائق القانونية والتنظيمية السعودية.

القواعد:
- أجب بناءً على السياق المقدم فقط
- كل بيان واقعي يجب أن يكون مصحوباً بمرجع
- إذا لم يكن هناك سياق، قل أن المعلومات غير متاحة"""
        else:
            return """You are a helpful AI assistant specialized in answering questions about Saudi Arabian legal and regulatory documents.

Rules:
- Answer based ONLY on the provided context
- Every factual statement MUST include a citation
- If no context, state that information is not available"""
    
    def build_context_prompt(self, docs: List[Document]) -> str:
        """Build context from retrieved documents."""
        if not docs:
            return "No relevant documents were found."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content
            metadata = doc.metadata or {}
            resource_path = self._get_resource_path(metadata)
            
            context_parts.append(
                f"[Reference {i}]\nDocument: {resource_path}\n"
                f"{'─' * 50}\n{content}\n{'─' * 50}"
            )
        
        return "\n\n".join(context_parts)
    
    def build_user_prompt(self, question: str, history_summary: Optional[str] = None) -> str:
        """Build user prompt."""
        if history_summary and history_summary != "No previous conversation.":
            return f"{history_summary}\n\nQuestion: {question}\n\nAnswer based on the provided context with citations."
        return f"Question: {question}\n\nAnswer based on the provided context with citations."
    
    def build_full_prompt(
        self,
        question: str,
        docs: List[Document],
        history_summary: Optional[str] = None
    ) -> Tuple[str, str]:
        """Build complete prompt: (system_prompt, full_prompt)."""
        is_arabic = any('\u0600' <= char <= '\u06FF' for char in question)
        language = "arabic" if is_arabic else "english"
        
        system_prompt = self.build_system_prompt(language)
        context = self.build_context_prompt(docs)
        user_prompt = self.build_user_prompt(question, history_summary)
        
        full_prompt = f"{context}\n\n{user_prompt}"
        return system_prompt, full_prompt
    
    def detect_language(self, text: str) -> str:
        """Detect language."""
        is_arabic = any('\u0600' <= char <= '\u06FF' for char in text)
        return "Arabic (العربية)" if is_arabic else "English"
    
    def _get_resource_path(self, metadata: dict) -> str:
        """Get resource path from metadata using shared utility."""
        return build_resource_path(metadata)
