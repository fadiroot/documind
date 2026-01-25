"""Question data models."""
from pydantic import BaseModel
from typing import Optional, List


class QuestionRequest(BaseModel):
    """Question request model."""
    question: str
    context_ids: Optional[List[str]] = None
    max_results: int = 5
    category: Optional[str] = None  # الفئة المستهدفة - e.g., "legal", "financial", "technical"


class QuestionResponse(BaseModel):
    """Question response model."""
    answer: str
    sources: List[dict]
    confidence: Optional[float] = None
