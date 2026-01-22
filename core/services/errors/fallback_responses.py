"""Fallback responses for error scenarios."""
from typing import Dict


class FallbackResponses:
    """Predefined fallback responses for common error scenarios."""
    
    ARABIC_RESPONSES = {
        "no_documents": (
            "عذراً، لم أتمكن من العثور على معلومات محددة حول هذا الموضوع في الوثائق المتاحة حالياً. "
            "يُرجى التأكد من أن الوثائق ذات الصلة قد تم فهرستها في النظام، أو محاولة إعادة صياغة السؤال بشكل مختلف."
        ),
        "low_confidence": (
            "عذراً، المعلومات المتاحة حول هذا الموضوع محدودة الثقة. "
            "يُرجى إعادة صياغة السؤال أو التحقق من الوثائق المتاحة."
        ),
        "retrieval_error": (
            "حدث خطأ أثناء البحث في قاعدة المعرفة. يُرجى المحاولة مرة أخرى لاحقاً."
        ),
        "llm_error": (
            "حدث خطأ أثناء معالجة السؤال. يُرجى المحاولة مرة أخرى."
        ),
        "citation_error": (
            "عذراً، لم أتمكن من التحقق من المراجع في الإجابة. يُرجى المحاولة مرة أخرى."
        ),
        "timeout": (
            "انتهت مهلة المعالجة. يُرجى المحاولة مرة أخرى مع سؤال أقصر أو أكثر تحديداً."
        )
    }
    
    ENGLISH_RESPONSES = {
        "no_documents": (
            "Sorry, I could not find specific information about this topic in the available documents. "
            "Please ensure that relevant documents have been indexed in the system, or try rephrasing your question."
        ),
        "low_confidence": (
            "Sorry, the available information about this topic has limited confidence. "
            "Please rephrase your question or verify the available documents."
        ),
        "retrieval_error": (
            "An error occurred while searching the knowledge base. Please try again later."
        ),
        "llm_error": (
            "An error occurred while processing your question. Please try again."
        ),
        "citation_error": (
            "Sorry, I could not verify the citations in the answer. Please try again."
        ),
        "timeout": (
            "Processing timeout. Please try again with a shorter or more specific question."
        )
    }
    
    @classmethod
    def get_response(cls, error_type: str, language: str = "arabic") -> str:
        """
        Get fallback response for error type.
        
        Args:
            error_type: Type of error (no_documents, low_confidence, etc.)
            language: Response language (arabic or english)
        
        Returns:
            Fallback response text
        """
        responses = cls.ARABIC_RESPONSES if language.lower() == "arabic" else cls.ENGLISH_RESPONSES
        return responses.get(error_type, responses.get("llm_error", "An error occurred."))
    
    @classmethod
    def detect_language(cls, text: str) -> str:
        """Detect language from text."""
        is_arabic = any('\u0600' <= char <= '\u06FF' for char in text)
        return "arabic" if is_arabic else "english"
