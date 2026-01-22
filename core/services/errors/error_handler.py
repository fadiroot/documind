"""Error handling utilities."""
from typing import Optional, Callable, Any
from functools import wraps

from core.services.errors.fallback_responses import FallbackResponses
from core.utils.logger import logger


class ErrorHandler:
    """Centralized error handling with fallback responses."""
    
    @staticmethod
    def handle_retrieval_error(
        error: Exception,
        question: str,
        language: str = "arabic"
    ) -> str:
        """Handle retrieval errors with fallback response."""
        logger.error(f"Retrieval error: {str(error)}")
        return FallbackResponses.get_response("retrieval_error", language)
    
    @staticmethod
    def handle_llm_error(
        error: Exception,
        question: str,
        language: str = "arabic"
    ) -> str:
        """Handle LLM errors with fallback response."""
        logger.error(f"LLM error: {str(error)}")
        return FallbackResponses.get_response("llm_error", language)
    
    @staticmethod
    def handle_citation_error(
        error: Exception,
        question: str,
        language: str = "arabic"
    ) -> str:
        """Handle citation validation errors with fallback response."""
        logger.error(f"Citation error: {str(error)}")
        return FallbackResponses.get_response("citation_error", language)
    
    @staticmethod
    def handle_no_documents(
        question: str,
        language: str = "arabic"
    ) -> str:
        """Handle case when no documents are retrieved."""
        logger.warning(f"No documents retrieved for question: {question[:100]}")
        return FallbackResponses.get_response("no_documents", language)
    
    @staticmethod
    def handle_low_confidence(
        question: str,
        avg_score: float,
        language: str = "arabic"
    ) -> str:
        """Handle case when retrieval confidence is low."""
        logger.warning(f"Low confidence ({avg_score:.2f}) for question: {question[:100]}")
        return FallbackResponses.get_response("low_confidence", language)
    
    @staticmethod
    def safe_execute(
        func: Callable,
        error_type: str = "llm_error",
        default_return: Any = None,
        language: str = "arabic"
    ) -> Any:
        """
        Safely execute a function with error handling.
        
        Args:
            func: Function to execute
            error_type: Type of error for fallback response
            default_return: Default return value on error
            language: Language for fallback response
        
        Returns:
            Function result or default_return
        """
        try:
            return func()
        except Exception as e:
            logger.error(f"Error in safe_execute: {str(e)}")
            if isinstance(default_return, str):
                return FallbackResponses.get_response(error_type, language)
            return default_return
